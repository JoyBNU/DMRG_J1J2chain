# -*- coding: utf-8 -*-
########################################################################################
# DMRG Program Of 1D Heisenberg Chain 
# H = -J * ΣSi·Sj(NN)
# 2025.8.19 H = -J1 * ΣSi·Sj(NN)- J2 * ΣSi·Sj(NNN) PBC
########################################################################################

# Parameters and some related functions 

import sys
import numpy as np
    
def read_params(filename="read.in"):
    with open(filename, "r") as f:
        line = f.readline().strip()
    N, J1, J2 = line.split()
    return int(N), float(J1), float(J2)


N, J1, J2 = read_params()
# J2=0.01*J2

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

