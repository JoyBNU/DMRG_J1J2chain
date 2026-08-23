# -*- coding: utf-8 -*-
# 2025.8.19 H = -J1 * ΣSi·Sj(NN)- J2 * ΣSi·Sj(NNN) PBC
import numpy as np
import sys

# Sweep
def sweep(N,J1,J2,kept,kept_plus,iteration,N_sys,N_env,Is,Sz,Su,Sd,H__sys,H__env,Sz_sys,Sz_env,Su_sys,Su_env,Sd_sys,Sd_env,\
          Sz_sys1,Su_sys1,Sd_sys1,Sz_env1,Su_env1,Sd_env1,Sz_sys2,Sz_env2,Su_sys2,Su_env2,Sd_sys2,Sd_env2,\
            Sz2_sys,Sz2_env,Su2_sys,Su2_env,Sd2_sys,Sd2_env,Dim_spin,Dim_sys,Dim_env):
    
    sweep_direction = 1   # Sweep_direction = 1 means → ; -1 means ←
    time_halfsweep = 0    # "Nsys equals N/2-1" means we finish a half-sweep
    keptnow = kept  
    O_sysnew = [None] * (N - 3)
    O_envnew = [None] * (N - 3)
    
    while time_halfsweep < iteration * 2:
        
        # One-step in a sweep
        N_sys = N_sys + sweep_direction
        N_env = N_env - sweep_direction
    
        if N_sys == 1 or N_env == 1:  
            sweep_direction = -1 * sweep_direction
    
        # Matrix representations of Hamiltonian of sys+1, 1+1, 1+env
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
            
        # Matrix representations of Hamiltonian of sys+1+1+env
        I_sys = np.eye(Dim_sys[N_sys - 1])                  # Identity of sys
        I_env = np.eye(Dim_env[N_env - 1])                  # Identity of env
        Dim_sysnew = Dim_sys[N_sys - 1] * Dim_spin          # Dimonsion of new sys
        Dim_envnew = Dim_env[N_env - 1] * Dim_spin          # Dimonsion of new env
        I_sysnew = np.eye(Dim_sysnew)                       # Identity of new sys
        I_envnew = np.eye(Dim_envnew)                       # Identity of new env
        
        H = np.kron(H_sysnew, I_envnew) + np.kron(I_sysnew, H_envnew) + np.kron(np.kron(I_sys, H_mid), I_env) + (
                np.kron(I_sys, H_midenv) + np.kron(H_midsys, I_env) + H_sn + H_sn1 + H_sn2
            )
        
        # print("==================sweep========================",N_sys,N_env)
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
            #print(f"time_halfsweep = {time_halfsweep}, sweep_direction = {sweep_direction}, N_sys = {N_sys}, N_env = {N_env}：H不是对称矩阵")
            sys.exit()  # Exit code
        
        # Diagonalize the Hamiltonian matrix and find the ground state
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        sorted_indices = np.argsort(eigenvalues)
        Ground_energy = min(eigenvalues)
        Ground_state = eigenvectors[:, np.argmin(eigenvalues)]
        # print(f'Ground_state= {Ground_state}')
        num=20
        A =eigenvalues
        B = eigenvectors[:, sorted_indices[:num]] 
        C = eigenvalues[sorted_indices[:num]]
        Rsp = Ground_state.reshape(Dim_sysnew, Dim_envnew)

        if sweep_direction == 1:
            
            # Rsp * Rsp' = \rho_{sys}
            Rho_sysnew = np.dot(Rsp,Rsp.T)
     
            # Diagonalize adnd truncate the reduced density matrix, and then the other quantities
            eigenvalues, eigenvectors = np.linalg.eigh(Rho_sysnew)
            idx_descending = np.argsort(-eigenvalues)
            Dim_sys[N_sys] = min(keptnow, Dim_sysnew)
            O_sysnew[N_sys] = eigenvectors[:, idx_descending[:Dim_sys[N_sys]]]
            
            # Check if O_sysnew is orthogonal
            is_orthogonal = np.allclose(np.dot(O_sysnew[N_sys].T,O_sysnew[N_sys]), np.eye(O_sysnew[N_sys].shape[1]), atol=1e-14)
            if not is_orthogonal:
                # print(f"time_halfsweep = {time_halfsweep}, sweep_direction = {sweep_direction}, N_sys = {N_sys}: O_sysnew Not an orthogonal matrix")
                O_sysnew[N_sys], _ = np.linalg.qr(O_sysnew[N_sys])
    
            # Update quantities
            
            H__sys[N_sys] = np.dot(np.dot(O_sysnew[N_sys].T, H_sysnew), O_sysnew[N_sys])
            Su_sys[N_sys] = np.dot(np.dot(O_sysnew[N_sys].T, np.kron(I_sys, Su)), O_sysnew[N_sys])
            Sd_sys[N_sys] = np.dot(np.dot(O_sysnew[N_sys].T, np.kron(I_sys, Sd)), O_sysnew[N_sys])
            Sz_sys[N_sys] = np.dot(np.dot(O_sysnew[N_sys].T, np.kron(I_sys, Sz)), O_sysnew[N_sys])
       
            Su_sys1[N_sys] = np.dot(np.dot(O_sysnew[N_sys].T, np.kron(Su_sys1[N_sys-1], Is)), O_sysnew[N_sys])
            Sd_sys1[N_sys] = np.dot(np.dot(O_sysnew[N_sys].T, np.kron(Sd_sys1[N_sys-1], Is)), O_sysnew[N_sys])
            Sz_sys1[N_sys] = np.dot(np.dot(O_sysnew[N_sys].T, np.kron(Sz_sys1[N_sys-1], Is)), O_sysnew[N_sys])
    
            Su2_sys[N_sys] = np.dot(np.dot(O_sysnew[N_sys].T, np.kron(Su_sys[N_sys - 1], Is)), O_sysnew[N_sys])
            Sd2_sys[N_sys] = np.dot(np.dot(O_sysnew[N_sys].T, np.kron(Sd_sys[N_sys - 1], Is)), O_sysnew[N_sys])
            Sz2_sys[N_sys] = np.dot(np.dot(O_sysnew[N_sys].T, np.kron(Sz_sys[N_sys - 1], Is)), O_sysnew[N_sys])

            if N_sys==1:
                Su_sys2[N_sys] = np.dot(np.dot(O_sysnew[N_sys].T, np.kron(Is, Su)), O_sysnew[N_sys])
                Sd_sys2[N_sys] = np.dot(np.dot(O_sysnew[N_sys].T, np.kron(Is, Sd)), O_sysnew[N_sys])
                Sz_sys2[N_sys] = np.dot(np.dot(O_sysnew[N_sys].T, np.kron(Is, Sz)), O_sysnew[N_sys])
            else :
                Su_sys2[N_sys] = np.dot(np.dot(O_sysnew[N_sys].T, np.kron(Su_sys2[N_sys-1], Is)), O_sysnew[N_sys])
                Sd_sys2[N_sys] = np.dot(np.dot(O_sysnew[N_sys].T, np.kron(Sd_sys2[N_sys-1], Is)), O_sysnew[N_sys])
                Sz_sys2[N_sys] = np.dot(np.dot(O_sysnew[N_sys].T, np.kron(Sz_sys2[N_sys-1], Is)), O_sysnew[N_sys])

        else:
            # Rsp' * Rsp = \rho_{env}^T
            Rho_envnew_transpose = np.dot(Rsp.T,Rsp)
            Rho_envnew = Rho_envnew_transpose.T
            
            # Diagonalize and truncate the reduced density matrix, and then the other quantities
            eigenvalues, eigenvectors = np.linalg.eigh(Rho_envnew)
            idx_descending = np.argsort(-eigenvalues)
            Dim_env[N_env] = min(keptnow, Dim_envnew)
            O_envnew[N_env] = eigenvectors[:, idx_descending[:Dim_env[N_env]]]

            # Check if O_envnew is orthogonal
            is_orthogonal = np.allclose(np.dot(O_envnew[N_env].T , O_envnew[N_env]), np.eye(O_envnew[N_env].shape[1]), atol=1e-14)
            
            if not is_orthogonal:
                # print(f"time_halfsweep = {time_halfsweep}, sweep_direction = {sweep_direction}, N_env = {N_env}: O_envnew Not an orthogonal matrix")
                O_envnew[N_env], _ = np.linalg.qr(O_envnew[N_env])
          
            # Update quantities
            H__env[N_env] = np.dot(np.dot(O_envnew[N_env].T , H_envnew) , O_envnew[N_env])
            Su_env[N_env] = np.dot(np.dot(O_envnew[N_env].T , np.kron(Su, I_env)) , O_envnew[N_env])
            Sd_env[N_env] = np.dot(np.dot(O_envnew[N_env].T , np.kron(Sd, I_env)) , O_envnew[N_env])
            Sz_env[N_env] = np.dot(np.dot(O_envnew[N_env].T , np.kron(Sz, I_env)) , O_envnew[N_env])

            Su_env1[N_env] = np.dot(np.dot(O_envnew[N_env].T , np.kron(Is, Su_env1[N_env-1])) , O_envnew[N_env])
            Sd_env1[N_env] = np.dot(np.dot(O_envnew[N_env].T , np.kron(Is, Sd_env1[N_env-1])) , O_envnew[N_env])
            Sz_env1[N_env] = np.dot(np.dot(O_envnew[N_env].T , np.kron(Is, Sz_env1[N_env-1])) , O_envnew[N_env])

            Su2_env[N_env] = np.dot(np.dot(O_envnew[N_env].T , np.kron(Is, Su_env[N_env - 1])) , O_envnew[N_env])
            Sd2_env[N_env] = np.dot(np.dot(O_envnew[N_env].T , np.kron(Is, Sd_env[N_env - 1])) , O_envnew[N_env])
            Sz2_env[N_env] = np.dot(np.dot(O_envnew[N_env].T , np.kron(Is, Sz_env[N_env - 1])) , O_envnew[N_env])

            if N_env==1:
                Su_env2[N_env] = np.dot(np.dot(O_envnew[N_env].T , np.kron(Su, Is)) , O_envnew[N_env])
                Sd_env2[N_env] = np.dot(np.dot(O_envnew[N_env].T , np.kron(Sd, Is)) , O_envnew[N_env])
                Sz_env2[N_env] = np.dot(np.dot(O_envnew[N_env].T , np.kron(Sz, Is)) , O_envnew[N_env])
            else :
                Su_env2[N_env] = np.dot(np.dot(O_envnew[N_env].T , np.kron(Is, Su_env2[N_env-1])) , O_envnew[N_env])
                Sd_env2[N_env] = np.dot(np.dot(O_envnew[N_env].T , np.kron(Is, Sd_env2[N_env-1])) , O_envnew[N_env])
                Sz_env2[N_env] = np.dot(np.dot(O_envnew[N_env].T , np.kron(Is, Sz_env2[N_env-1])) , O_envnew[N_env])
                
        if N_sys == N/2 - 1:  
            time_halfsweep += 1  
            keptnow += kept_plus


    print(f'Sweep结束：DMRG ground state energy = {Ground_energy:.8f}')
    # data = np.column_stack((
    # [N] * num,      
    # [J1] * num,     
    # [J2] * num,     
    # np.arange(1, num+1),  
    # A[0:num]    
    # ))
    # np.savetxt('eigenvalues.dat', data, fmt='%d %f %f %d %.8f')

    # return Sz_sys, Sz_env, O_sysnew, O_envnew, Ground_state
    return Sz_sys, Sz_env, O_sysnew, O_envnew, B, C, num, J1, J2
