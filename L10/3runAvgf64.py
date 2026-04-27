import pyopencl as cl
import numpy as np
import time, statistics, matplotlib.pyplot as plt

KERNEL_SRC = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void mandelbrot(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;

    double c_real = x_min + col * (x_max - x_min) / (double)N;
    double c_imag = y_min + row * (y_max - y_min) / (double)N;

    double zr = 0.0, zi = 0.0;
    int count = 0;
    while (count < max_iter && zr*zr + zi*zi <= 4.0) {
        double tmp = zr*zr - zi*zi + c_real;
        zi = 2.0 * zr * zi + c_imag;
        zr = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""

ctx   = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
prog  = cl.Program(ctx, KERNEL_SRC).build()

N, MAX_ITER = 1024, 100
X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.25, 1.25

image = np.zeros((N, N), dtype=np.int32)
image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)

# --- Warm up ---
prog.mandelbrot(queue, (64, 64), None, image_dev,
                np.float64(X_MIN), np.float64(X_MAX),
                np.float64(Y_MIN), np.float64(Y_MAX),
                np.int32(64), np.int32(MAX_ITER))
queue.finish()

# --- Timed runs ---
times = []
for _ in range(3):
    t0 = time.perf_counter()
    prog.mandelbrot(queue, (N, N), None, image_dev,
                    np.float64(X_MIN), np.float64(X_MAX),
                    np.float64(Y_MIN), np.float64(Y_MAX),
                    np.int32(N), np.int32(MAX_ITER))
    queue.finish()
    times.append(time.perf_counter() - t0)

print(f"GPU f64 {N}x{N}: {statistics.median(times)*1e3:.1f} ms")

cl.enqueue_copy(queue, image, image_dev)
queue.finish()

plt.imshow(image, cmap='hot', origin='lower'); plt.axis('off')
plt.savefig("mandelbrot_gpu_f64.png", dpi=150, bbox_inches='tight')