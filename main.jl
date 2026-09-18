# using Random
using LinearAlgebra
import PythonPlot as plt

# Random.seed!(123)

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


function HF_Hubbard_Hamiltonian(N, t, U, P)
    # need to explicitly set them to complex to avoid problems later
    T_term = zeros(ComplexF64, 2*N, 2*N)
    U_term = zeros(ComplexF64, 2*N, 2*N)

    # hopping term
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

    # on-site repulsion term
    for i in 1:2*N
        for j in 1:2*N 
            site_i = mod1(i, N)
            site_j = mod1(j, N)
            spin_i = fld(i - 1, N)
            spin_j = fld(j - 1, N)

            Hartree_term = delta(spin_i, spin_j) * P[i, j]
            Fock_term = (1 - delta(spin_i, spin_j)) * P[i, j]

            U_term[i, j] = U * delta(site_i, site_j) * (Hartree_term - Fock_term)
        end
    end

    H = T_term + U_term   # construct full HF hamiltonian
    return H
end


function SCF_optimization(N, N_e, t, U, P_0, ϵ, max_iter)
    P_k_tilde = P_0

    for k in 1:max_iter
        log("Iteration $k")
        # global P_k_tilde    # setting this to global scope so it can be modified in the loop
        
        # construct k_th Hamiltonian 
        H_k_tilde = HF_Hubbard_Hamiltonian(N, t, U, P_k_tilde)
        log("Calculated hamiltonian")

        # diagonalize k_th Hamiltonian
        eigenvalues, eigenvectors = eigen(H_k_tilde)
        log("Diagonalized hamiltonian")

        # keeping first N_e eigenvectors (lowest energy orbitals)
        orbitals = eigenvectors[:, 1:N_e]

        # constructing temporary next density matrix using occupied orbitals O(N^3)
        P_k₊ = orbitals * orbitals' 
        log("Constructed new density matrix")

        # checking convergence
        difference = norm(P_k₊ - P_k_tilde)
        log("difference = $difference")
        if difference < ϵ
            P_k_tilde = P_k₊
            break
        end

        # compute temporary next hamiltonian
        H_k₊ = HF_Hubbard_Hamiltonian(N, t, U, P_k₊)
        log("Calculated temporary hamiltonian")

        # computing s and c coefficients for optimal mixing scheme
        s = real(2 * tr(H_k_tilde * (P_k₊ - P_k_tilde)))
        c = real(tr((H_k₊ - H_k_tilde) * (P_k₊ - P_k_tilde)))

        # building k+1_th optimally mixed 1rdm
        λ = (c <= -s / 2) ? 1 : (-s / (2 * c))
        P_k₊_tilde = (1 - λ) * P_k_tilde + λ * P_k₊
        log("Constructed next density matrix")

        P_k_tilde = P_k₊_tilde
    end

    return P_k_tilde
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
    Sz = zeros(N)
    
    for i in 1:N
        n_up = real(P[i, i])
        n_down = real(P[N + i, N + i])

        Sz[i] = 0.5 * (n_up - n_down)
    end
    return Sz
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
# Analytics Functions
# ============================================

"""
Generate a plot of Sz difference between 2 fixed sites over a range of U/t values at fixed N, N_e
"""
function generate_spin_diagram_2_sites(N, N_e, Ut_ratio_max, n_samples)
    ut_vals = [(i + 1) * Ut_ratio_max for i in 0:(n_samples - 1)]
end 

# ============================================
# Setting variables
# ============================================

N = 100                               # orbitals
N_e = 40                              # electrons (N_e <= 2N)
P_0 = generate_rand_1rdm(N, N_e)      # 1rdm (initial guess)
t = 1                                 # hopping term
U = 1                                 # on site repulsion
max_iter = 100
ϵ = 1e-6


# ============================================
# Running the SCF algorithm
# ============================================

P_out = SCF_optimization(N, N_e, t, U, P_0, ϵ, max_iter)

# ============================================
# Performing Tests and Calculations
# ============================================

check_1rdm(P_out)

S = spin_magnitude_profile(P_out, N)

plt.plot(1:N, S)
plt.show()

# calculate net magnetization (EV of spin-z operator with P_out)

