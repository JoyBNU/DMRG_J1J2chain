
########################################################################################
# DMRG Program Of 1D Heisenberg Chain 
# H = -J * ΣSi·Sj(NN)
# 2025.8.15 H = -J1 * ΣSi·Sj(NN)- J2 * ΣSi·Sj(NNN) OBC
########################################################################################

# Parameters and some related functions 

import sys
    
# Enter raw parameters
N = 10          # Length of the lattice
J1 = 1           # The strength of Heisenberg interaction
J2 = 0.1
kept = 10       # Kept dimension of Hamiltonian matrix
iteration = 5   # Number of sweeps
kept_plus = 5   # Added kept dimension every sweep

m_Sz = 1        # Measure of Sz
m_entropy = 1   # Measure of the von Neumann entropy

# Check whether the entered parameters are reasonable
def check_parameters(N):
    # N should be an even number
    if N % 2 == 1:
        print("Please set L as an even number.")
        sys.exit()

