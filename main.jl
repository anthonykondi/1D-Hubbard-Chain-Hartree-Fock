using Random
using LinearAlgebra
import PythonPlot as plt
using Dates

println("Finished Importing")

# ============================================
# Defining Utility Functions
# ============================================

delta(i, j) = (i == j) ? 1 : 0    # Kronecker delta function


function random_unitary(n)        # Stole this from somewhere
    Z = randn(ComplexF64, n, n)
    Q, R = qr(Z)
    return Matrix(Q) * Diagonal(diag(R) ./ abs.(diag(R)))
end


function log(msg)     # Chatgpt apparently said this should be better
    println(msg)
    flush(stdout)
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


# ============================================
# Hartree-Fock machinery
# ============================================

function hopping_matrix(N, t)
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


function HF_Hubbard_Hamiltonian(N, T, U, P)
    # need to explicitly set them to complex to avoid problems later
    H = copy(T)

    # CORRECTED (what I had before was wrong): 
    for i in 1:N
        H[i, i]         += U * real(P[i+N, i+N])
        H[i+N, i+N]     += U * real(P[i, i])
        H[i, i+N]       -= U * P[i, i+N]
        H[i+N, i]       -= U * P[i+N, i]
    end
    return H
end


function SCF_optimization(N, N_e, t, U, P_0, ϵ, max_iter)
    T = hopping_matrix(N, t)
    P_k_tilde = P_0

    for k in 1:max_iter
        log("Iteration $k")
        # global P_k_tilde    # setting this to global scope so it can be modified in the loop
        
        # construct k_th Hamiltonian 
        H_k_tilde = HF_Hubbard_Hamiltonian(N, T, U, P_k_tilde)
        # log("Calculated hamiltonian")

        # diagonalize k_th Hamiltonian
        eigenvalues, eigenvectors = eigen(H_k_tilde)
        # log("Diagonalized hamiltonian")

        # keeping first N_e eigenvectors (lowest energy orbitals)
        orbitals = eigenvectors[:, 1:N_e]

        # constructing temporary next density matrix using occupied orbitals O(N^3)
        P_k₊ = orbitals * orbitals' 
        # log("Constructed new density matrix")

        # checking convergence
        difference = norm(P_k₊ - P_k_tilde)
        log("difference = $difference")
        if difference < ϵ
            return P_k₊
        end

        # compute temporary next hamiltonian
        H_k₊ = HF_Hubbard_Hamiltonian(N, T, U, P_k₊)
        # log("Calculated temporary hamiltonian")

        # computing s and c coefficients for optimal mixing scheme
        s = real(2 * tr(H_k_tilde * (P_k₊ - P_k_tilde)))
        c = real(tr((H_k₊ - H_k_tilde) * (P_k₊ - P_k_tilde)))

        # building k+1_th optimally mixed 1rdm
        λ = (c <= -s / 2) ? 1 : (-s / (2 * c))
        P_k₊_tilde = (1 - λ) * P_k_tilde + λ * P_k₊
        # log("Constructed next density matrix")

        P_k_tilde = P_k₊_tilde
    end

    @warn "SCF did not converge"
    return P_k_tilde
end


# ============================================
# Analytics Functions
# ============================================

"""
|Sz(A) - Sz(B)| for a range of U/t values.
Each U starts from a fresh Néel seed (to reach AFM configuration).
"""
function spin_difference_2_sites(N, N_e, Ut_ratio_max, n_samples, ϵ, max_iter, site_A, site_B)
    ut_vals = [(i / (n_samples - 1)) * Ut_ratio_max for i in 0:(n_samples - 1)]
    diffs = Float64[]

    for ut in ut_vals
        P_0 = neel_seed(N, N_e)     # because we are looking for AFM (to try to match Ivan's plot)
        P_out = SCF_optimization(N, N_e, 1, ut, P_0, ϵ, max_iter)

        Sz_A = 0.5 * real(P_out[site_A, site_A] - P_out[site_A + N, site_A + N])
        Sz_B = 0.5 * real(P_out[site_B, site_B] - P_out[site_B + N, site_B + N])
        push!(diffs, abs(Sz_A - Sz_B))
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


# ============================================
# Code for testing
# ============================================

Random.seed!(123)

N = 30
N_e = 30
ut_max = 10
samples = 16
ϵ = 1e-6
max_iter = 200

ut_range = 0:(10/(samples - 1)):10

Sz_diff = spin_difference_2_sites(N, N_e, ut_max, samples, ϵ, max_iter, 1, 2)

plt.figure()
plt.scatter(ut_range, Sz_diff)
plt.xlabel("U/t")
plt.ylabel("|Sz(A) - Sz(B)|")
plt.title("Hubbard ring (periodic), half filling, N = $N")
plt.savefig("comparison_plot.png")
plt.show()