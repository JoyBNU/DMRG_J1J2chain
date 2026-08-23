# -*- coding: utf-8 -*-
# Initialize and build a two-lattice system, then increase the number of 
# particles in the system
# 2025.8.19 H = -J1 * ΣSi·Sj(NN)- J2 * ΣSi·Sj(NNN) PBC
import numpy as np
import sys

def init_increasing(N, kept, J1, J2):
    
    # The Hilbert space dimonsion of a single spin
    Dim_spin = 2    # 2S+1 .c.
    
    # The Hilbert space dimonsion of The Sys Block and The Env Block with 
    # different sites
    Dim_sys = np.zeros(N - 3, int)
    Dim_env = np.zeros(N - 3, int)
    Dim_sys[0] = Dim_spin
    Dim_env[0] = Dim_spin
    
    # Matrix representations of the Sys/Env Block Hamiltonian and the operators 
    # of marginal sites. These are what we needed to calculate the Hamiltonian 
    # matrix of the whole system after increasing sites. H__sys acts like a 
    # storage repository and we update the data at the end of each rise.
    H__sys = [None] * (N - 3)
    Su_sys = [None] * (N - 3)
    Sd_sys = [None] * (N - 3)
    Sz_sys = [None] * (N - 3)
    H__env = [None] * (N - 3)
    Su_env = [None] * (N - 3)
    Sd_env = [None] * (N - 3)
    Sz_env = [None] * (N - 3)

    Su_sys1 = [None] * (N - 3)  # first site
    Sd_sys1 = [None] * (N - 3)
    Sz_sys1 = [None] * (N - 3)

    Su_env1 = [None] * (N - 3)  # last site
    Sd_env1 = [None] * (N - 3)
    Sz_env1 = [None] * (N - 3)
   
    Su2_sys = [None] * (N - 3)
    Sd2_sys = [None] * (N - 3)
    Sz2_sys = [None] * (N - 3)
    Su2_env = [None] * (N - 3)
    Sd2_env = [None] * (N - 3)
    Sz2_env = [None] * (N - 3)

    Su_sys2 = [None] * (N - 3)  # second site
    Sd_sys2 = [None] * (N - 3)
    Sz_sys2 = [None] * (N - 3)
    Su_env2 = [None] * (N - 3)  # last second site
    Sd_env2 = [None] * (N - 3)
    Sz_env2 = [None] * (N - 3)
    
    # Create matrices for the Pauli operator and the identity operator
    Su = np.array([[0, 1], [0, 0]])
    Sd = np.array([[0, 0], [1, 0]])
    Sz = np.array([[0.5, 0], [0, -0.5]])
    Is = np.eye(Dim_spin)

    # Initialization of a two-lattice system: n_sys=1, n_env=1
    H__sys[0] = np.zeros((Dim_spin, Dim_spin))
    Su_sys[0] = Su
    Sd_sys[0] = Sd
    Sz_sys[0] = Sz
    H__env[0] = np.zeros((Dim_spin, Dim_spin))
    Su_env[0] = Su
    Sd_env[0] = Sd
    Sz_env[0] = Sz

    Su_sys1[0] = Su  # first site
    Sd_sys1[0] = Sd
    Sz_sys1[0] = Sz

    Su_env1[0] = Su  # last site
    Sd_env1[0] = Sd
    Sz_env1[0] = Sz

    Sz2_sys[0] = np.zeros((Dim_spin, Dim_spin))
    Su2_sys[0] = np.zeros((Dim_spin, Dim_spin))
    Sd2_sys[0] = np.zeros((Dim_spin, Dim_spin))

    Su_sys2[0] = np.zeros((Dim_spin, Dim_spin))  # second site
    Sd_sys2[0] = np.zeros((Dim_spin, Dim_spin))
    Sz_sys2[0] = np.zeros((Dim_spin, Dim_spin))
    Su_env2[0] = np.zeros((Dim_spin, Dim_spin))  # last second site
    Sd_env2[0] = np.zeros((Dim_spin, Dim_spin))
    Sz_env2[0] = np.zeros((Dim_spin, Dim_spin))

    # The number of times to increase sites
    time_growth = int((N - 2) / 2) 
    keptnow = kept              

    # Increasing sites
    # note that i starts from 0, so the first time is i=0
    for i in range(time_growth):  
        # The number of sys/env sites 
        N_sys = i + 1
        N_env = i + 1
    
        # Matrix representation of Hamiltonian of sys+1, 1+1, 1+env
        H_sysnew = np.kron(H__sys[N_sys - 1], Is) + J1 * (
            np.kron(Sz_sys[N_sys - 1], Sz) + 0.5 * (
                np.kron(Sd_sys[N_sys - 1], Su) + np.kron(Su_sys[N_sys - 1], Sd)
            )
        )

        if N_sys>=2:
            H_sysnew += J2 * (
                np.kron(Sz2_sys[N_sys - 1], Sz) + 0.5 * (
                    np.kron(Sd2_sys[N_sys - 1], Su) + 
                    np.kron(Su2_sys[N_sys - 1], Sd)
                )
            )
        
        H_envnew = np.kron(Is, H__env[N_env - 1]) + J1 * (
            np.kron(Sz, Sz_env[N_env - 1]) + 0.5 * (
                np.kron(Su, Sd_env[N_env - 1]) + np.kron(Sd, Su_env[N_env - 1])
            )
        )

        if N_env>=2:
            H_envnew += J2 * (
            np.kron(Sz, Sz2_env[N_env - 1]) + 0.5 * (
                np.kron(Su, Sd2_env[N_env - 1]) + 
                np.kron(Sd, Su2_env[N_env - 1])
            )
            )

        H_mid = J1 * (np.kron(Sz, Sz) + 0.5 * (np.kron(Sd, Su) + np.kron(Su, Sd))) 
        
        H_midenv = J2 * (
            np.kron(Sz, np.kron(Is,Sz_env[N_env - 1])) + 0.5 * (
                np.kron(Su, np.kron(Is,Sd_env[N_env - 1])) + np.kron(Sd, np.kron(Is,Su_env[N_env - 1]))
            )
        )

        H_midsys = J2 * (
            np.kron(np.kron(Sz_sys[N_sys - 1],Is), Sz) + 0.5 * (
                np.kron(np.kron(Sd_sys[N_sys - 1],Is), Su) + np.kron(np.kron(Su_sys[N_sys - 1],Is), Sd)
            )
        )

        # PBC Hamiltonian
        Dim_super = Dim_sys[N_sys-1] * 2 * 2 * Dim_env[N_env-1]  # 当前超块维度
        H_sn = np.zeros((Dim_super, Dim_super))
        H_sn1 = np.zeros((Dim_super, Dim_super))
        H_sn2 = np.zeros((Dim_super, Dim_super))
        
        H_sn = J1 * (np.kron(np.kron(Sz_sys1[N_sys-1], np.kron(Is, Is)), Sz_env1[N_env-1])
            + 0.5 * (np.kron(np.kron(Su_sys1[N_sys-1], np.kron(Is, Is)), Sd_env1[N_env-1])
                    + np.kron(np.kron(Sd_sys1[N_sys-1], np.kron(Is, Is)), Su_env1[N_env-1])))
        
        if N_sys >= 2 and N_env >= 1:
            H_sn1 = J2 * (np.kron(np.kron(Sz_sys2[N_sys-1],np.kron(Is,Is)), Sz_env1[N_env-1]) +\
                      0.5 * (np.kron(np.kron(Su_sys2[N_sys-1],np.kron(Is,Is)), Sd_env1[N_env-1]) +\
                              np.kron(np.kron(Sd_sys2[N_sys-1],np.kron(Is,Is)), Su_env1[N_env-1])))
        
        if N_sys >= 1 and N_env >= 2:
            H_sn2 = J2 * (np.kron(np.kron(Sz_sys1[N_sys-1],np.kron(Is,Is)), Sz_env2[N_env-1]) +\
                      0.5 * (np.kron(np.kron(Su_sys1[N_sys-1],np.kron(Is,Is)), Sd_env2[N_env-1]) +\
                              np.kron(np.kron(Sd_sys1[N_sys-1],np.kron(Is,Is)), Su_env2[N_env-1]))) 
        
        # Matrix representation of Hamiltonian of sys+1+1+env
        I_sys = np.eye(Dim_sys[N_sys - 1])                  # Identity of sys
        I_env = np.eye(Dim_env[N_env - 1])                  # Identity of env
        Dim_sysnew = Dim_sys[N_sys - 1] * Dim_spin          # Dimonsion of new sys
        Dim_envnew = Dim_env[N_env - 1] * Dim_spin          # Dimonsion of new env
        I_sysnew = np.eye(Dim_sysnew)                       # Identity of new sys
        I_envnew = np.eye(Dim_envnew)                       # Identity of new env

        H = np.kron(H_sysnew, I_envnew) + np.kron(I_sysnew, H_envnew) + np.kron(np.kron(I_sys, H_mid), I_env) + (
                np.kron(I_sys, H_midenv) + np.kron(H_midsys, I_env) + H_sn + H_sn1 + H_sn2
            )
        # print("===============================================",N_sys,N_env)
        # print("H_sysnew", H_sysnew.shape)
        # print("H_envnew", H_envnew.shape)
        # print("H_mid", H_mid.shape)
        # print("H_midenv", H_midenv.shape)
        # print("H_midsys", H_midsys.shape) 
        # print("H_sn", H_sn.shape)
        # print("H_sn1", H_sn1.shape)
        # print("H_sn2", H_sn2.shape) 
        # print("H", H.shape) 
        # Check if H is symmetrical
        is_symmetrical = np.allclose(H.T, H, atol=1e-14)
        if not is_symmetrical:
            #print(f"N_sys = {N_sys}, N_env = {N_env}：H不是对称矩阵")
            sys.exit()  # Exit code
        
        # Diagonalize the Hamiltonian matrix and find the ground state
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        Ground_energy = min(eigenvalues)
        Ground_state = eigenvectors[:, np.argmin(eigenvalues)]
        
        # Rsp * Rsp' = \rho_{sys} 
        # Rsp' * Rsp = \rho_{env}^T
        Rsp = Ground_state.reshape(Dim_sysnew, Dim_envnew)
        Rho_sysnew = np.dot(Rsp,Rsp.T)
        Rho_envnew_transpose = np.dot(Rsp.T,Rsp)
        Rho_envnew = Rho_envnew_transpose.T        
    
        # Diagonalize the reduced density matrix, and then truncate the other quantities
        # Sys        
        # Diagonalize the reduced density matrix and find the transformation matrix
        eigenvalues, eigenvectors = np.linalg.eigh(Rho_sysnew)
        idx_descending = np.argsort(-eigenvalues)
        Dim_sys[N_sys] = min(keptnow, Dim_sysnew)
        O_sysnew = eigenvectors[:, idx_descending[:Dim_sys[N_sys]]]
        
        # print(f"N_sys = {(A-B)/A}")
        # Check if O_sysnew is orthogonal
        is_orthogonal = np.allclose(np.dot(O_sysnew.T,O_sysnew), np.eye(O_sysnew.shape[1]), atol=1e-14)
        if not is_orthogonal:
            #print(f"N_sys = {N_sys}：O_sysnew不是正交矩阵")
            O_sysnew, _ = np.linalg.qr(O_sysnew)
        
        # Update quantities
        H__sys[N_sys] = np.dot(np.dot(O_sysnew.T, H_sysnew), O_sysnew)
        Su_sys[N_sys] = np.dot(np.dot(O_sysnew.T, np.kron(I_sys, Su)), O_sysnew)
        Sd_sys[N_sys] = np.dot(np.dot(O_sysnew.T, np.kron(I_sys, Sd)), O_sysnew)
        Sz_sys[N_sys] = np.dot(np.dot(O_sysnew.T, np.kron(I_sys, Sz)), O_sysnew)

        Su_sys1[N_sys] = np.dot(np.dot(O_sysnew.T, np.kron(Su_sys1[N_sys-1], Is)), O_sysnew)
        Sd_sys1[N_sys] = np.dot(np.dot(O_sysnew.T, np.kron(Sd_sys1[N_sys-1], Is)), O_sysnew)
        Sz_sys1[N_sys] = np.dot(np.dot(O_sysnew.T, np.kron(Sz_sys1[N_sys-1], Is)), O_sysnew)
    
        Su2_sys[N_sys] = np.dot(np.dot(O_sysnew.T, np.kron(Su_sys[N_sys - 1], Is)), O_sysnew)
        Sd2_sys[N_sys] = np.dot(np.dot(O_sysnew.T, np.kron(Sd_sys[N_sys - 1], Is)), O_sysnew)
        Sz2_sys[N_sys] = np.dot(np.dot(O_sysnew.T, np.kron(Sz_sys[N_sys - 1], Is)), O_sysnew)

        if N_sys==1:
            Su_sys2[N_sys] = np.dot(np.dot(O_sysnew.T, np.kron(Is, Su)), O_sysnew)
            Sd_sys2[N_sys] = np.dot(np.dot(O_sysnew.T, np.kron(Is, Sd)), O_sysnew)
            Sz_sys2[N_sys] = np.dot(np.dot(O_sysnew.T, np.kron(Is, Sz)), O_sysnew)
        else :
            Su_sys2[N_sys] = np.dot(np.dot(O_sysnew.T, np.kron(Su_sys2[N_sys-1], Is)), O_sysnew)
            Sd_sys2[N_sys] = np.dot(np.dot(O_sysnew.T, np.kron(Sd_sys2[N_sys-1], Is)), O_sysnew)
            Sz_sys2[N_sys] = np.dot(np.dot(O_sysnew.T, np.kron(Sz_sys2[N_sys-1], Is)), O_sysnew)
            
        # Env
        # Diagonalize the reduced density matrix and find the transformation matrix
        eigenvalues, eigenvectors = np.linalg.eigh(Rho_envnew)
        idx_descending = np.argsort(-eigenvalues)
        Dim_env[N_env] = min(keptnow, Dim_envnew)
        O_envnew = eigenvectors[:, idx_descending[:Dim_env[N_env]]]
        
        # Check if O_envnew is orthogonal
        is_orthogonal = np.allclose(np.dot(O_envnew.T , O_envnew), np.eye(O_envnew.shape[1]), atol=1e-14)
        if not is_orthogonal:
            #print(f"N_env = {N_env}：O_envnew不是正交矩阵")
            O_envnew, _ = np.linalg.qr(O_envnew)
        
        # Update quantities
        H__env[N_env] = np.dot(np.dot(O_envnew.T , H_envnew) , O_envnew)
        Su_env[N_env] = np.dot(np.dot(O_envnew.T , np.kron(Su, I_env)) , O_envnew)
        Sd_env[N_env] = np.dot(np.dot(O_envnew.T , np.kron(Sd, I_env)) , O_envnew)
        Sz_env[N_env] = np.dot(np.dot(O_envnew.T , np.kron(Sz, I_env)) , O_envnew)

        Su_env1[N_env] = np.dot(np.dot(O_envnew.T , np.kron(Is, Su_env1[N_env-1])) , O_envnew)
        Sd_env1[N_env] = np.dot(np.dot(O_envnew.T , np.kron(Is, Sd_env1[N_env-1])) , O_envnew)
        Sz_env1[N_env] = np.dot(np.dot(O_envnew.T , np.kron(Is, Sz_env1[N_env-1])) , O_envnew)

        Su2_env[N_env] = np.dot(np.dot(O_envnew.T , np.kron(Is, Su_env[N_env - 1])) , O_envnew)
        Sd2_env[N_env] = np.dot(np.dot(O_envnew.T , np.kron(Is, Sd_env[N_env - 1])) , O_envnew)
        Sz2_env[N_env] = np.dot(np.dot(O_envnew.T , np.kron(Is, Sz_env[N_env - 1])) , O_envnew)

        if N_env==1:
            Su_env2[N_env] = np.dot(np.dot(O_envnew.T , np.kron(Su, Is)) , O_envnew)
            Sd_env2[N_env] = np.dot(np.dot(O_envnew.T , np.kron(Sd, Is)) , O_envnew)
            Sz_env2[N_env] = np.dot(np.dot(O_envnew.T , np.kron(Sz, Is)) , O_envnew)
        else :
            Su_env2[N_env] = np.dot(np.dot(O_envnew.T , np.kron(Is, Su_env2[N_env-1])) , O_envnew)
            Sd_env2[N_env] = np.dot(np.dot(O_envnew.T , np.kron(Is, Sd_env2[N_env-1])) , O_envnew)
            Sz_env2[N_env] = np.dot(np.dot(O_envnew.T , np.kron(Is, Sz_env2[N_env-1])) , O_envnew)

    print(f'涨点完成：DMRG ground state energy = {Ground_energy:.8f}')
    return N_sys,N_env,Is,Sz,Su,Sd,H__sys,H__env,Sz_sys,Sz_env,Su_sys,Su_env,Sd_sys,Sd_env,\
        Sz_sys1,Su_sys1,Sd_sys1,Sz_env1,Su_env1,Sd_env1,Sz_sys2,Sz_env2,Su_sys2,Su_env2,Sd_sys2,Sd_env2,\
        Sz2_sys,Sz2_env,Su2_sys,Su2_env,Sd2_sys,Sd2_env,Dim_spin,Dim_sys,Dim_env
