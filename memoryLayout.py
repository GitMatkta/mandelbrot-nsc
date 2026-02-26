import numpy as np
import time

N = 10000
A = np.random.rand(N, N) #normal A = 3.80x diff
A = np.asfortranarray(A)  # Ensure column-major order (Fortran-style) #now 0.30x diff for rows

# Time row sum (accessing rows - should be faster)
start = time.time()
for i in range(N):
    s = np.sum(A[i, :])
elapsed_row = time.time() - start
print(f"Row sum time: {elapsed_row:.2f} seconds")

# Time column sum (accessing columns - should be slower)
start = time.time()
for i in range(N):
    s = np.sum(A[:, i])
elapsed_col = time.time() - start
print(f"Column sum time: {elapsed_col:.2f} seconds")

print(f"Column/Row ratio: {elapsed_col/elapsed_row:.2f}x")

