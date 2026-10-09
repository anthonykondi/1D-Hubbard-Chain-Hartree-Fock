using Random
using LinearAlgebra
import PythonPlot as plt
using Dates

println("Finished Importing")

# ========================================================================================
# Defining Utility Functions
# ========================================================================================

delta(i, j) = (i == j) ? 1 : 0    # Kronecker delta function


function random_unitary(n)        # Stole this from somewhere
    Z = randn(ComplexF64, n, n)
    Q, R = qr(Z)
    return Matrix(Q) * Diagonal(diag(R) ./ abs.(diag(R)))
end


"""
Fill the first N_e diagonal components with 1s (full occupation).
"""
function generate_diag_1rdm(N, N_e)
    P = zeros(2*N, 2*N)
    for i in 1:N_e
        P[i, i] = 1
    end
    return P
end


"""
Randomly generate a valid 1 body reduced density matrix for the 
N-site 1-d Hubbard chain with N_e electrons.
"""
function generate_rand_1rdm(N, N_e)
    D = generate_diag_1rdm(N, N_e)
    U = random_unitary(2*N)
    P = U * D * U'
    return P
end


function neel_seed(N, N_e; amplitude = 0.4)
    n̄ = N_e / (2N)
    a = min(amplitude, n̄, 1 - n̄)
    P = zeros(2N, 2N)
    for i in 1:N
        n_up = isodd(i) ? n̄ + a : n̄ - a
        P[i, i]     = n_up
        P[i+N, i+N] = 2n̄ - n_up
    end
    return P
end


# ========================================================================================
# Hartree-Fock machinery (Real space)
# ========================================================================================

function hopping_matrix_real_space(N, t)
    T_term = zeros(ComplexF64, 2*N, 2*N)

    for i in 1:2*N          # rows
        for j in 1:2*N      # columns
            # same spin 
            same_spin = (i <= N) == (j <= N)

            # corresponding lattice sites
            site_i = mod1(i, N)
            site_j = mod1(j, N)

            # nearest neighbour condition
            nearest_neighbours = (
                (mod1(site_i + 1, N) == site_j) ||
                (mod1(site_i - 1, N) == site_j)
            )

            if same_spin && nearest_neighbours
                T_term[i, j] = -t
            end
        end
    end
    return T_term
end


function HF_Hubbard_Hamiltonian_real_space(N, T, U, P)
    H = copy(T)

    for i in 1:2*N, j in 1:2*N                # 2-fold loop
        site_i = mod1(i, N)
        site_j = mod1(j, N)
        site_i == site_j || continue          # V is on-site only

        if fld(i - 1, N) == fld(j - 1, N)     # same spin (and same site => i == j)
            ī = i <= N ? i + N : i - N        # same site, opposite spin
            H[i, j] += U * real(P[ī, ī])      # Hartree: opposite-spin density
        else
            H[i, j] -= U * P[i, j]            # Fock: spin-flip element
        end
    end

    return H
end


"""Construct the hopping matrix (it is diagonal in k space)""" 
function hopping_matrix_k_space(N, t)
    T_term = zeros(ComplexF64, 2*N, 2*N)
    
    for i in 1:2*N
        m = mod1(i, N)
        T_term[i, i] = -2 * t * cos(2 * pi * m / N)
    end
    
    return T_term
end


function HF_Hubbard_Hamiltonian_k_space(N, T, U, P)    # Ivan said this part can be improved with sparce matrices (for RHF)
    H = copy(T)

    for i in 1:2*N, j in 1:2*N                # O(N^3)
        k_i = mod1(i, N)
        k_j = mod1(j, N)
        spin_i = fld(i - 1, N)
        spin_j = fld(j - 1, N)

        if spin_i == spin_j                   # same spin (Hartree term)
            spin_opp = 1 - spin_i             # 0 -> 1, 1 -> 0 (spin flip)
            Hartree = 0

            for s in 1:N
                skk_idx = mod1(s + k_j - k_i, N) 
                idx_1 = s + spin_opp * N
                idx_2 = skk_idx + spin_opp * N
                Hartree += P[idx_1, idx_2]
            end
            H[i, j] += (U / N) * Hartree
        
        else                                  # opposite spin (Fock term) 
            Fock = 0
            
            for s in 1:N
                skk_idx = mod1(s + k_j - k_i, N) 
                idx_1 = s + spin_i * N
                idx_2 = skk_idx + spin_j * N
                Fock += P[idx_1, idx_2]
            end
            H[i, j] -= (U / N) * Fock
        
        end
    end

    return H
end


# numerically stable Fermi function (avoids exp overflow (exp gets too big))
fermi(x) = x > 0 ? exp(-x) / (1 + exp(-x)) : 1 / (1 + exp(x))


# using a bisection method (n_avg is monotonic in μ so this method always converges)
function find_chemical_potential(N_e, eigen_energies, kT; tol=1e-13, max_iter=200)
    n_avg(μ) = sum(fermi((e - μ) / kT) for e in eigen_energies)

    # i'm not exactly sure why we would take these bounds that Claude gave me
    lo = minimum(eigen_energies) - 40*kT
    hi = maximum(eigen_energies) + 40*kT

    for _ in 1:max_iter
        μ = (lo + hi) / 2
        if n_avg(μ) < N_e
            lo = μ
        else
            hi = μ
        end
        hi - lo < tol && break
    end
    return (lo + hi) / 2
end


function SCF_HF(N, N_e, t, U, P_0, ϵ, max_iter; real_space=true, kT=0)
    if real_space
        T = hopping_matrix_real_space(N, t)
    else
        T = hopping_matrix_k_space(N, t)
    end

    P_k_tilde = P_0

    for k in 1:max_iter
        println("Iteration $k")
        
        # construct k_th Hamiltonian 
        if real_space
            H_k_tilde = HF_Hubbard_Hamiltonian_real_space(N, T, U, P_k_tilde)
        else
            H_k_tilde = HF_Hubbard_Hamiltonian_k_space(N, T, U, P_k_tilde)
        end

        # diagonalize k_th Hamiltonian
        eigen_energies, orbitals = eigen(Hermitian(H_k_tilde))

        # keeping first N_e eigenvectors (lowest energy orbitals)
        if kT == 0
            C_occ = zeros(2*N, 2*N)
            for i in 1:N_e
                C_occ[i, i] = 1
            end
        else
            μ = find_chemical_potential(N_e, eigen_energies, kT)
            C_occ = diagm(fermi.(-(1 / kT) .* (μ .- eigen_energies)))
        end

        # constructing temporary next density matrix using occupied orbitals O(N^3)
        P_k₊ = orbitals * C_occ * orbitals' 

        # checking convergence
        difference = norm(P_k₊ - P_k_tilde)
        println("difference = $difference")
        if difference < ϵ
            return P_k₊
        end

        # compute temporary next hamiltonian
        if real_space
            H_k₊ = HF_Hubbard_Hamiltonian_real_space(N, T, U, P_k₊)
        else
            H_k₊ = HF_Hubbard_Hamiltonian_k_space(N, T, U, P_k₊)
        end

        # computing s and c coefficients for optimal mixing scheme
        s = real(2 * tr(H_k_tilde * (P_k₊ - P_k_tilde)))
        c = real(tr((H_k₊ - H_k_tilde) * (P_k₊ - P_k_tilde)))

        # building k+1_th optimally mixed 1rdm
        λ = (c <= -s / 2) ? 1 : (-s / (2 * c))
        P_k₊_tilde = (1 - λ) * P_k_tilde + λ * P_k₊

        P_k_tilde = P_k₊_tilde
    end

    @warn "SCF did not converge"
    return P_k_tilde
end


# ========================================================================================
# RHF Functions (using GHF framework [doesn't work that well])
# ========================================================================================

function generate_rand_restricted_1rdm(N, N_e)
    nocc = Int(N_e / 2)                       # orbitals per spin
    V = random_unitary(N)                # N×N, not 2N×2N
    Pσ = V[:, 1:nocc] * V[:, 1:nocc]'    # rank-nocc projector
    P = zeros(ComplexF64, 2N, 2N)
    P[1:N, 1:N] = Pσ
    P[N+1:2N, N+1:2N] = Pσ
    return P
end


"""Forces P to have the diagonal stripe form (for RHF, 2 site cells)"""
function restrict(P, N)
    Pavg = (P[1:N, 1:N] + P[N+1:2N, N+1:2N]) / 2
    Pr = zeros(eltype(P), 2N, 2N)
    Pr[1:N, 1:N] = Pavg
    Pr[N+1:2N, N+1:2N] = Pavg
    return Pr
end


function SCF_RHF(N, N_e, t, U, P_0, ϵ, max_iter; kT=0)
    T = hopping_matrix_k_space(N, t)
    P_k_tilde = P_0

    for k in 1:max_iter
        println("Iteration $k")
        
        # construct k_th Hamiltonian 
        H_k_tilde = HF_Hubbard_Hamiltonian_k_space(N, T, U, P_k_tilde)

        # diagonalize k_th Hamiltonian
        eigen_energies, orbitals = eigen(Hermitian(H_k_tilde))

        # keeping first N_e eigenvectors (lowest energy orbitals)
        if kT == 0
            C_occ = zeros(2*N, 2*N)
            for i in 1:N_e
                C_occ[i, i] = 1
            end
        else
            μ = find_chemical_potential(N_e, eigen_energies, kT)
            C_occ = diagm(fermi.(-(1 / kT) .* (μ .- eigen_energies)))
        end

        # constructing temporary next density matrix using occupied orbitals O(N^3)
        P_k₊ = orbitals * C_occ * orbitals'
        P_k₊ = restrict(P_k₊, N)    # applying RHF restriction

        # checking convergence
        difference = norm(P_k₊ - P_k_tilde)
        println("difference = $difference")
        if difference < ϵ
            return P_k₊
        end

        # compute temporary next hamiltonian
        H_k₊ = HF_Hubbard_Hamiltonian_k_space(N, T, U, P_k₊)

        # computing s and c coefficients for optimal mixing scheme
        s = real(2 * tr(H_k_tilde * (P_k₊ - P_k_tilde)))
        c = real(tr((H_k₊ - H_k_tilde) * (P_k₊ - P_k_tilde)))

        # building k+1_th optimally mixed 1rdm
        λ = (c <= -s / 2) ? 1 : (-s / (2 * c))
        P_k₊_tilde = (1 - λ) * P_k_tilde + λ * P_k₊   # still in the restricted form

        P_k_tilde = P_k₊_tilde
    end

    @warn "SCF did not converge"
    return P_k_tilde
end


function magnetization_alternating_sites_RHF(N, N_e, Ut_ratio_max, n_samples, ϵ, max_iter; kT=0)
    ut_vals = [(i / (n_samples - 1)) * Ut_ratio_max for i in 0:(n_samples - 1)]
    diffs = Float64[]

    for ut in ut_vals
        # the choice of ansatz here is actually really crucial (also only works for half filling)
        P_0 = generate_rand_restricted_1rdm(N, N_e)
        P_k_out = SCF_RHF(N, N_e, 1, ut, P_0, ϵ, max_iter, kT=kT)
        P_out = k_to_real_P(P_k_out, N)
        
        # calculating alternating site magnetization
        mag = zeros(3)
        for i in 1:N 
            Sx = real(P_out[i, i + N]) * (-1)^(i - 1)
            Sy = -imag(P_out[i, i + N]) * (-1)^(i - 1)
            Sz = 0.5 * real(P_out[i, i] - P_out[i + N, i + N]) * (-1)^(i - 1)
            mag += [Sx, Sy, Sz]
        end
        mag ./= N 

        push!(diffs, dot(mag, mag))
    end
    return diffs
end


# ========================================================================================
# RHF Functions (2 site unit cells)
# ========================================================================================

"""
Collinear UHF block-diagonal Hamiltonian at momentum k, basis (1↑, 2↑, 1↓, 2↓).
b  : double-cell length (2x nearest-neighbour distance)
P  : k-averaged 4x4 density matrix (only the diagonal enters the Hartree term)
"""
function UHF_Hamiltonian(k, b, P, t, U)
    γ   = -(1 + exp(-im * k * b))
    off = t * γ
    H_up   = [U*real(P[3,3])  off;
              conj(off)       U*real(P[4,4])]
    H_down = [U*real(P[1,1])  off;
              conj(off)       U*real(P[2,2])]
    Z = zeros(ComplexF64, 2, 2)
    return Hermitian([H_up Z; Z H_down])
end

"""
Occupation numbers f[i,n] for the pooled spectrum evals[i,n] (k-index i, band n).
kT == 0 : fill lowest Ne states, sharing electrons equally within a degenerate shell at the cut.
kT  > 0 : Fermi-Dirac, μ found by bisection so that Σf = Ne.
Returns (f, μ).
"""
function occupations(evals, Ne, kT; tol=1e-9)
    flat = vec(evals)
    if kT == 0
        EF    = sort(flat)[Ne]
        below = flat .< EF - tol
        ties  = abs.(flat .- EF) .<= tol
        f = zeros(length(flat))
        f[below] .= 1.0
        f[ties]  .= (Ne - count(below)) / count(ties)
        return reshape(f, size(evals)), EF
    else
        fermi(E, μ) = 0.5 * (1 - tanh((E - μ) / (2kT)))   # overflow-safe
        lo, hi = minimum(flat) - 20kT, maximum(flat) + 20kT
        for _ in 1:200
            μ = (lo + hi) / 2
            sum(fermi.(flat, μ)) < Ne ? (lo = μ) : (hi = μ)
        end
        μ = (lo + hi) / 2
        return reshape(fermi.(flat, μ), size(evals)), μ
    end
end

"""
One pass: build and diagonalize H(k) for every k from the SAME P̄, fill globally,
return the per-k density matrices P(k), their average P̄_new, the H(k)'s and μ.
"""
function diagonalize_and_fill(ks, b, Pbar, t, U, Ne, kT)
    N  = length(ks)
    Hs = [UHF_Hamiltonian(k, b, Pbar, t, U) for k in ks]
    es = [eigen(H) for H in Hs]
    evals = [es[i].values[n] for i in 1:N, n in 1:4]

    f, μ = occupations(evals, Ne, kT)

    Pk = [es[i].vectors * Diagonal(f[i, :]) * es[i].vectors' for i in 1:N]
    Pbar_new = sum(Pk) / N
    return Pk, Pbar_new, Hs, μ
end

"""
HF energy per unit cell (no entropy term): (1/N) Σ_k tr[h0(k) P(k)] + U Σ_α n_α↑ n_α↓
"""
function HF_energy(ks, b, Pk, Pbar, t, U)
    N = length(ks)
    Ekin = sum(real(tr(UHF_Hamiltonian(k, b, Pbar, t, 0.0) * Pk[i]))
               for (i, k) in enumerate(ks)) / N
    Eint = U * real(Pbar[1,1]*Pbar[3,3] + Pbar[2,2]*Pbar[4,4])
    return Ekin + Eint
end

"""
UHF SCF with ODA. P_0 is a 4x4 initial (k-averaged) density matrix.
N : number of double cells, N_e : total electrons (L = 2N sites, half filling → N_e = 2N)
Returns (P̄, E, μ).
"""
function SCF_2_site_cells_HF(N, N_e, t, U, P_0; b=1.0, kT=0.0, ϵ=1e-6, max_iter=1000)
    ks = [2π * m / (N * b) for m in 1:N]          # m = 1..N

    # first step: full step from the initial guess
    Pk_t, Pbar_t, _, μ = diagonalize_and_fill(ks, b, P_0, t, U, N_e, kT)

    for iter in 1:max_iter
        # H(k) built from the current mixed P̄, then new P(k) and P̄⁺
        Pk_p, Pbar_p, Hs, μ = diagonalize_and_fill(ks, b, Pbar_t, t, U, N_e, kT)

        # convergence: distance between output and input density
        difference = norm(Pbar_p - Pbar_t)
        println("Iteration $iter: difference = $difference")
        if difference < ϵ
            E = HF_energy(ks, b, Pk_p, Pbar_p, t, U)
            return Pbar_p, E, μ
        end

        # ODA: E(λ) = E(P̃) + s λ + c λ², with ΔP = P⁺ - P̃
        ΔPk   = Pk_p .- Pk_t
        ΔPbar = Pbar_p - Pbar_t
        s = real(sum(tr(Hs[i] * ΔPk[i]) for i in 1:N)) / N
        c = U * real(ΔPbar[1,1]*ΔPbar[3,3] + ΔPbar[2,2]*ΔPbar[4,4])
        λ = c > 0 ? clamp(-s / (2c), 0.0, 1.0) : 1.0

        Pk_t   = (1 - λ) .* Pk_t .+ λ .* Pk_p
        Pbar_t = (1 - λ) * Pbar_t + λ * Pbar_p
    end

    @warn "SCF did not converge"
    Pk_f = Pk_t
    return Pbar_t, HF_energy(ks, b, Pk_f, Pbar_t, t, U), μ
end


function bands_2site(Pbar, t, U; b=1.0, nk=401)
    kb = range(-π, π; length=nk)            # k*b over one double-cell BZ
    ks = kb ./ b
    E_up = zeros(nk, 2); E_dn = zeros(nk, 2)
    for (i, k) in enumerate(ks)
        H = Matrix(UHF_Hamiltonian(k, b, Pbar, t, U))
        E_up[i, :] = eigvals(Hermitian(H[1:2, 1:2]))
        E_dn[i, :] = eigvals(Hermitian(H[3:4, 3:4]))
    end
    x = ks .* (b / 2) ./ π                  # k·a / π, with a = b/2
    return x, E_up, E_dn
end

function plot_bands_2site(Pbar, t, U, μ; b=1.0, nk=401)
    x, E_up, E_dn = bands_2site(Pbar, t, U; b, nk)
    fig = plt.figure(figsize=(6, 5)); ax = fig.add_subplot(1, 1, 1)
    for n in 1:2
        ax.plot(x, E_up[:, n], "b-";  label = n == 1 ? raw"$\uparrow$" : nothing)
        ax.plot(x, E_dn[:, n], "r--"; label = n == 1 ? raw"$\downarrow$" : nothing)
    end
    ax.axhline(μ; color="k", ls=":", lw=0.8)
    ax.set_xlabel(raw"$ka/\pi$"); ax.set_ylabel("E")
    ax.legend(); fig.tight_layout()
    return fig
end


# ========================================================================================
# Analytics Functions 
# ========================================================================================

"""Convert density matrix from k to real space"""
function k_to_real_P(P_k, N)
    P_real = zeros(ComplexF64, 2 * N, 2 * N)
    for i in 1:2*N, j in 1:2*N
        spin_i = fld(i - 1, N)
        spin_j = fld(j - 1, N)
        site_i = mod1(i, N)
        site_j = mod1(j, N)

        for m1 in 1:N, m2 in 1:N
            k1 = 2 * pi * m1 / N 
            k2 = 2 * pi * m2 / N
            P_real[i, j] += exp(im * k1 * site_i) * exp(-im * k2 * site_j) * (
                P_k[m1 + spin_i * N, m2 + spin_j * N]) / N
        end
    end
    return P_real
end


"""Convert density matrix from real to k space"""
function real_to_k_P(P_real, N)
    P_k = zeros(ComplexF64, 2 * N, 2 * N)
    for mi_idx in 1:2*N, mj_idx in 1:2*N
        spin_i = fld(mi_idx - 1, N)
        spin_j = fld(mj_idx - 1, N)
        mi = mod1(mi_idx, N)
        mj = mod1(mj_idx, N)
        ki = 2 * pi * mi / N 
        kj = 2 * pi * mj / N

        for site_i in 1:N, site_j in 1:N
            P_k[mi_idx, mj_idx] += exp(-im * ki * site_i) * exp(im * kj * site_j) * (
                P_real[site_i + spin_i * N, site_j + spin_j * N]) / N
        end
    end
    return P_k
end


"""
|Sz(A) - Sz(B)| for a range of U/t values.
Each U starts from a fresh Néel seed (to reach AFM configuration).
"""
function spin_difference_alternating_sites(N, N_e, Ut_ratio_max, n_samples, ϵ, max_iter)
    ut_vals = [(i / (n_samples - 1)) * Ut_ratio_max for i in 0:(n_samples - 1)]
    diffs = Float64[]

    for ut in ut_vals
        # the choice of ansatz here is actually really crucial (also only works for half filling)
        P_0 = neel_seed(N, N_e)     # because we are looking for AFM (to try to match Ivan's plot)
        # P_0 = generate_rand_1rdm(N, N_e)
        P_out = SCF_real_space(N, N_e, 1, ut, P_0, ϵ, max_iter)

        # calculating alternating site spind difference
        S_diff = 0
        for i in 1:N 
            S_diff += 0.5 * real(P_out[i, i] - P_out[i + N, i + N]) * (-1)^(i - 1)
        end

        push!(diffs, abs(S_diff))
    end
    return diffs
end


# here I should really just make 1 function and have a setting to switch from real to k space

function magnetization_alternating_sites(N, N_e, Ut_ratio_max, n_samples, ϵ, max_iter; real_space=true, kT=0)
    ut_vals = [(i / (n_samples - 1)) * Ut_ratio_max for i in 0:(n_samples - 1)]
    diffs = Float64[]

    for ut in ut_vals
        # the choice of ansatz here is actually really crucial (also only works for half filling)
        P_0 = neel_seed(N, N_e)
        if real_space
            P_out = SCF_HF(N, N_e, 1, ut, P_0, ϵ, max_iter, real_space=true, kT=kT)
        else
            P_k_0 = real_to_k_P(P_0, N)
            P_k_out = SCF_HF(N, N_e, 1, ut, P_k_0, ϵ, max_iter, real_space=false, kT=kT)
            P_out = k_to_real_P(P_k_out, N)
        end

        # calculating alternating site magnetization
        mag = zeros(3)
        for i in 1:N 
            Sx = real(P_out[i, i + N]) * (-1)^(i - 1)
            Sy = -imag(P_out[i, i + N]) * (-1)^(i - 1)
            Sz = 0.5 * real(P_out[i, i] - P_out[i + N, i + N]) * (-1)^(i - 1)
            mag += [Sx, Sy, Sz]
        end
        mag ./= N 

        push!(diffs, dot(mag, mag))
    end
    return diffs
end


function check_1rdm(P)     # running through the 3 properties that P should satisfy
    h_err = norm(P - P')
    p_err = norm(P - P * P)
    e_num = tr(P)

    println("h_err = $h_err")
    println("p_err = $p_err")
    println("e_num = $e_num")
end


function Sz_ev_profile(P, N)     # spin-z expectation value profile
    return [0.5 * real(P[i, i] - P[i + N, i + N]) for i in 1:N]
end


function spin_magnitude_profile(P, N)
    Sx = zeros(Float64, N)
    Sy = zeros(Float64, N)
    Sz = zeros(Float64, N)

    for i in 1:N
        up = i
        down = N + i

        Puu = P[up, up]
        Pdd = P[down, down]
        Pud = P[up, down]

        Sx[i] = real(Pud)
        Sy[i] = -imag(Pud)
        Sz[i] = 0.5 * real(Puu - Pdd)
    end
    
    return sqrt.(Sx.^2 .+ Sy.^2 .+ Sz.^2)
end


function spin_blocks(P, N; layout=:blocked)
    up, dn = layout == :blocked ? (1:N, N+1:2N) : (1:2:2N, 2:2:2N)
    return P[up, up], P[dn, dn], P[up, dn]
end


function plot_1rdm(P, N; layout=:blocked, title="")
    Pu, Pd, Pud = spin_blocks(P, N; layout)

    nu = real.(diag(Pu))
    nd = real.(diag(Pd))
    n  = nu .+ nd              # charge density  n_i
    m  = 0.5 .* (nu .- nd)     # magnetization   m_i = <S^z_i>

    fig = plt.figure(figsize=(9, 11))

    # heatmaps with a symmetric color scale centered at zero
    function hm(k, M, t)
        ax = fig.add_subplot(3, 2, k)
        c = max(maximum(abs, real.(M)), 1e-12)
        im = ax.imshow(real.(M); cmap="RdBu", vmin=-c, vmax=c)
        ax.set_title(t)
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        fig.colorbar(im; ax=ax)
    end

    hm(1, Pu,  raw"$P_{\uparrow\uparrow}$")
    hm(2, Pd,  raw"$P_{\downarrow\downarrow}$")
    hm(3, Pud, raw"$P_{\uparrow\downarrow}$ (should be ≈ 0 for collinear UHF)")

    # site densities
    ax = fig.add_subplot(3, 2, 4)
    ax.plot(1:N, n, "o-"; label=raw"$n_i$")
    ax.plot(1:N, m, "s-"; label=raw"$m_i$")
    ax.set_ylim(-0.1, 2.1)
    ax.set_xlabel("site i")
    ax.set_title("Site densities")
    ax.legend()

    # |P_ij| vs distance |i-j| (decay of correlations)
    Ptot = Pu .+ Pd
    d = [abs(i - j) for i in 1:N, j in 1:N]
    ax = fig.add_subplot(3, 2, 5)
    ax.scatter(vec(d), max.(vec(abs.(Ptot)), 1e-16); s=8)
    ax.set_yscale("log")
    ax.set_xlabel("|i − j|")
    ax.set_ylabel(raw"$|P_{ij}|$")
    ax.set_title("Decay")

    fig.suptitle(title)
    fig.tight_layout()
    return fig
end

# ========================================================================================
# Code for testing
# ========================================================================================

Random.seed!(123)

N, t, U = 40, 1.0, 4.0
N_e = 80
m0 = 0.1
P_0 = Diagonal([(1+m0)/2, (1-m0)/2, (1-m0)/2, (1+m0)/2])   # 1↑, 2↑, 1↓, 2↓

Pbar, E, μ = SCF_2_site_cells_HF(N, N_e, t, U, P_0)
plot_bands_2site(Pbar, t, U, μ)

plt.title("band structure N = $N, N_e = $N_e, t = $t, U = $U")
plt.savefig("band structure N = $N, N_e = $N_e, t = $t, U = $U.png")
plt.show()