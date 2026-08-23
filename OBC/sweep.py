# 2025.8.15 H = -J1 * ΣSi·Sj(NN)- J2 * ΣSi·Sj(NNN) OBC
import numpy as np
import sys

# Sweep
def sweep(N,J1,J2,kept,kept_plus,iteration,N_sys,N_env,Is,Sz,Su,Sd,H__sys,H__env,Sz_sys,Sz_env,Su_sys,Su_env,Sd_sys,Sd_env,Sz2_sys,Sz2_env,Su2_sys,Su2_env,Sd2_sys,Sd2_env,Dim_spin,Dim_sys,Dim_env):
    
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
    
        # Matrix representations of Hamiltonian of sys+1+1+env
        I_sys = np.eye(Dim_sys[N_sys - 1])                  # Identity of sys
        I_env = np.eye(Dim_env[N_env - 1])                  # Identity of env
        Dim_sysnew = Dim_sys[N_sys - 1] * Dim_spin          # Dimonsion of new sys
        Dim_envnew = Dim_env[N_env - 1] * Dim_spin          # Dimonsion of new env
        I_sysnew = np.eye(Dim_sysnew)                       # Identity of new sys
        I_envnew = np.eye(Dim_envnew)                       # Identity of new env
        H = np.kron(H_sysnew, I_envnew) + np.kron(I_sysnew, H_envnew) + np.kron(np.kron(I_sys, H_mid), I_env) + (
            np.kron(I_sys, H_midenv) + np.kron(H_midsys, I_env)
        )

        # Check if H is symmetrical
        is_symmetrical = np.allclose(H.T, H, atol=1e-14)
        if not is_symmetrical:
            print(f"time_halfsweep = {time_halfsweep}, sweep_direction = {sweep_direction}, N_sys = {N_sys}, N_env = {N_env}：H不是对称矩阵")
            sys.exit()  # Exit code
        
        # Diagonalize the Hamiltonian matrix and find the ground state
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        Ground_energy = min(eigenvalues)
        Ground_state = eigenvectors[:, np.argmin(eigenvalues)]
        A =eigenvalues
        Rsp = Ground_state.reshape(Dim_sysnew, Dim_envnew)

        if sweep_direction == 1:
            
            # Rsp * Rsp' = \rho_{sys}
            Rho_sysnew = Rsp @ Rsp.T    
     
            # Diagonalize adnd truncate the reduced density matrix, and then the other quantities
            eigenvalues, eigenvectors = np.linalg.eigh(Rho_sysnew)
            idx_descending = np.argsort(-eigenvalues)
            Dim_sys[N_sys] = min(keptnow, Dim_sysnew)
            O_sysnew[N_sys] = eigenvectors[:, idx_descending[:Dim_sys[N_sys]]]
            
            # Check if O_sysnew is orthogonal
            is_orthogonal = np.allclose(O_sysnew[N_sys].T @ O_sysnew[N_sys], np.eye(O_sysnew[N_sys].shape[1]), atol=1e-14)
            if not is_orthogonal:
                print(f"time_halfsweep = {time_halfsweep}, sweep_direction = {sweep_direction}, N_sys = {N_sys}：O_sysnew不是正交矩阵")
                O_sysnew[N_sys], _ = np.linalg.qr(O_sysnew[N_sys])
    
            # Update quantities
            H__sys[N_sys] = O_sysnew[N_sys].T @ H_sysnew @ O_sysnew[N_sys]
            Su_sys[N_sys] = O_sysnew[N_sys].T @ np.kron(I_sys, Su) @ O_sysnew[N_sys]
            Sd_sys[N_sys] = O_sysnew[N_sys].T @ np.kron(I_sys, Sd) @ O_sysnew[N_sys]
            Sz_sys[N_sys] = O_sysnew[N_sys].T @ np.kron(I_sys, Sz) @ O_sysnew[N_sys]
            Su2_sys[N_sys] = O_sysnew[N_sys].T @ np.kron(Su_sys[N_sys - 1], np.eye(Dim_spin)) @ O_sysnew[N_sys]
            Sd2_sys[N_sys] = O_sysnew[N_sys].T @ np.kron(Sd_sys[N_sys - 1], np.eye(Dim_spin)) @ O_sysnew[N_sys]
            Sz2_sys[N_sys] = O_sysnew[N_sys].T @ np.kron(Sz_sys[N_sys - 1], np.eye(Dim_spin)) @ O_sysnew[N_sys]
        
        else:
            # Rsp' * Rsp = \rho_{env}^T
            Rho_envnew_transpose = Rsp.T @ Rsp
            Rho_envnew = Rho_envnew_transpose.T
            
            # Diagonalize and truncate the reduced density matrix, and then the other quantities
            eigenvalues, eigenvectors = np.linalg.eigh(Rho_envnew)
            idx_descending = np.argsort(-eigenvalues)
            Dim_env[N_env] = min(keptnow, Dim_envnew)
            O_envnew[N_env] = eigenvectors[:, idx_descending[:Dim_env[N_env]]]
            
            # Check if O_envnew is orthogonal
            is_orthogonal = np.allclose(O_envnew[N_env].T @ O_envnew[N_env], np.eye(O_envnew[N_env].shape[1]), atol=1e-14)
            if not is_orthogonal:
                print(f"time_halfsweep = {time_halfsweep}, sweep_direction = {sweep_direction}, N_env = {N_env}：O_envnew不是正交矩阵")
                O_envnew[N_env], _ = np.linalg.qr(O_envnew[N_env])
          
            # Update quantities
            H__env[N_env] = O_envnew[N_env].T @ H_envnew @ O_envnew[N_env]
            Su_env[N_env] = O_envnew[N_env].T @ np.kron(Su, I_env) @ O_envnew[N_env]
            Sd_env[N_env] = O_envnew[N_env].T @ np.kron(Sd, I_env) @ O_envnew[N_env]
            Sz_env[N_env] = O_envnew[N_env].T @ np.kron(Sz, I_env) @ O_envnew[N_env]
            Su2_env[N_env] = O_envnew[N_env].T @ np.kron(np.eye(Dim_spin), Su_env[N_env - 1]) @ O_envnew[N_env]
            Sd2_env[N_env] = O_envnew[N_env].T @ np.kron(np.eye(Dim_spin), Sd_env[N_env - 1]) @ O_envnew[N_env]
            Sz2_env[N_env] = O_envnew[N_env].T @ np.kron(np.eye(Dim_spin), Sz_env[N_env - 1]) @ O_envnew[N_env]

        if N_sys == N/2 - 1:  
            time_halfsweep += 1  
            keptnow += kept_plus
    
    print(f'Sweep结束：DMRG ground state energy = {Ground_energy:.8f}')
    indices = np.arange(1, 21)
    data_to_save = np.column_stack((indices, A[:20]))
    np.savetxt('eigenvalues.dat',data_to_save, fmt='%d %.8f')
    return Sz_sys, Sz_env, O_sysnew, O_envnew, Ground_state

