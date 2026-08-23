import numpy as np

def measure_Sz(N,Sz_sys,Sz_env,O_sysnew,O_envnew,Ground_state):
    
    # Prepare some matrices for later computation
    Dim_spin = 2
    Dim_sys = O_sysnew[int(N/2 - 2)].shape[1]
    Dim_env = O_envnew[int(N/2 - 2)].shape[1]
    Is = np.eye(Dim_spin)
    I_sys = np.eye(Dim_sys)             
    I_env = np.eye(Dim_env)           
    Sz_spin = np.array([[0.5, 0], [0, -0.5]])
    
    # We divide the whole system into four parts to calculate: sys · · env
    Sz = np.zeros(N)                # Stores values of Sz for different sites 
    for i in range(N):              # i means measuring the i+1 site，
        
        # Sites in sys
        if i < N/2 - 1:             
            Sz_matrix = Sz_sys[i]   # Take the matrix that was computed before
            k = i + 2               # k means the length of sysnew for a O_sysnew
            while k < N/2:  
                Sz_matrix = O_sysnew[k - 1].T @ np.kron(Sz_matrix, Is) @ O_sysnew[k - 1]
                k += 1
            Sz[i] = Ground_state.T @ np.kron(np.kron(np.kron(Sz_matrix, Is), Is), I_env) @ Ground_state
        
        # Sites in env
        elif i > N/2:               
            Sz_matrix = Sz_env[N - 1 - i] 
            k = N - i + 1           # k means the length of envnew for a O_envnew
            while k < N/2:  
                Sz_matrix = O_envnew[k - 1].T @ np.kron(Is, Sz_matrix) @ O_envnew[k - 1]
                k += 1
            Sz[i] = Ground_state.T @ np.kron(I_sys, np.kron(Is, np.kron(Is, Sz_matrix))) @ Ground_state

        # The left site in the middle two points
        elif i == N/2 - 1:
            Sz_matrix = np.kron(I_sys, np.kron(np.kron(Sz_spin, Is), I_env))
            Sz[i] = Ground_state.T @ Sz_matrix @ Ground_state
        
        # The right site in the middle two points
        else:
            Sz_matrix = np.kron(I_sys, np.kron(np.kron(Is, Sz_spin), I_env))
            Sz[i] = Ground_state.T @ Sz_matrix @ Ground_state

    np.savetxt('measurement_Sz.txt', Sz)

