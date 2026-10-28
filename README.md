# 1D Hubbard Model: Hartree–Fock Simulation

## Overview

A learning project implementing the **one-dimensional Hubbard model** (Hubbard chain / quantum wire) with periodic boundary conditions. The code uses self-consistent Hartree–Fock (SCF) to approximate the many-body ground state as a Slater determinant, exploring how electron interactions influence the system's properties.

The project is intended for experimenting with many-body theory, finite-temperature quantum mechanics, band structures, magnetic order, and the approximations needed to study interacting quantum systems.

## How It Works

1. **Initialize the system:** Generate an initial one-body reduced density matrix (1RDM), either randomly or with a chosen magnetic ordering.
2. **Construct the Hamiltonian:** Build the hopping and interaction terms in either real space or momentum space, using Hartree–Fock decoupling of the Hubbard interaction.
3. **Solve the single-particle problem:** Diagonalize the effective Hamiltonian to obtain orbital energies and eigenstates.
4. **Update the density matrix:** Occupy the lowest-energy orbitals at zero temperature, or use Fermi–Dirac occupations at finite temperature.
5. **Iterate to convergence:** Apply optimal damping to stabilize the self-consistent field iterations.
6. **Analyze the results:** Examine spin profiles, magnetic order parameters, density matrices, and electronic band structures.

## Main Functions

- `generate_rand_1rdm(...)` and `neel_seed(...)` — Generate initial density matrices with random or antiferromagnetic structure.
- `hopping_matrix_real_space(...)` and `hopping_matrix_k_space(...)` — Construct the kinetic-energy matrix in different representations.
- `HF_Hubbard_Hamiltonian_real_space(...)` and `HF_Hubbard_Hamiltonian_k_space(...)` — Build the effective Hartree–Fock Hamiltonian.
- `SCF_HF(...)` — Perform self-consistent Hartree–Fock iterations with optimal damping.
- `SCF_RHF(...)` — Explore a restricted Hartree–Fock variant.
- `SCF_2_site_cells_HF(...)` — Solve the model using a two-site unit cell and momentum-resolved Hamiltonians.
- `occupations(...)` and `find_chemical_potential(...)` — Determine orbital occupations and chemical potential.
- `bands_2site(...)` and `plot_bands_2site(...)` — Calculate and visualize electronic band structures.
- `magnetization_alternating_sites(...)` and `spin_magnitude_profile(...)` — Analyze magnetic order and local spin properties.
- `plot_1rdm(...)` and `check_1rdm(...)` — Visualize and check properties of the density matrix.

## Running the Script

**Requirements:** Julia, `PythonPlot`, and the standard Julia libraries `Random`, `LinearAlgebra`, and `Dates`.

Install the plotting package in the Julia REPL:

```julia
using Pkg
Pkg.add("PythonPlot")
```

Run the script with:

```bash
julia main.jl
```

The example at the bottom initializes a half-filled chain with \(N=40\), \(t=1\), and \(U=4\), solves the two-site-unit-cell Hartree–Fock problem, and plots the resulting band structure. System parameters and initial conditions can be adjusted in the testing section.

## Future Improvements

- Refactor repeated SCF logic into reusable components with configurable representations and approximations.
- Improve performance through sparse matrices, more efficient matrix operations, and optimized momentum-space calculations.
- Investigate convergence failures, alternative mixing schemes, and sensitivity to initial conditions.
- Explore magnetic order parameters, symmetry breaking, and phase behaviour as functions of \(U/t\), filling, and temperature.
- Compare Hartree–Fock results against exact diagonalization for small systems or other many-body approximation methods.
- Expand visualization and analysis tools for density matrices, band structures, and finite-temperature behaviour.

