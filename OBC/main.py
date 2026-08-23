# 2025.8.15 H = -J1 * ΣSi·Sj(NN)- J2 * ΣSi·Sj(NNN) OBC
import time
from parameters import N, J1, J2, kept, kept_plus, iteration, m_Sz, check_parameters
from initialization_increasing import init_increasing
from sweep import sweep
import measurement

begin_Time = time.time()  

# Check whether the entered parameters are reasonable
check_parameters(N)

# Initializes and expands to the set length
ii_result = init_increasing(N, kept, J1, J2)

# Search for the optimal solution through sweep
sweep_result = sweep(N, J1, J2, kept, kept_plus, iteration, *ii_result)

# Measure the Sz
if m_Sz == 1:
    measurement.measure_Sz(N, *sweep_result)

end_Time = time.time()  
print(f"Elapsed time: {end_Time - begin_Time} seconds")

