import numpy as np
import pytest

from test_parallelMandelbrot import (
    mandelbrot_pixel,
    mandelbrot_chunk,
    mandelbrot_serial,
    mandelbrot_parallel,
)

# --- Helper small config for fast tests ---
TEST_N = 32
X_MIN, X_MAX = -2, 1
Y_MIN, Y_MAX = -1.5, 1.5
MAX_ITER = 50


def test_pixel_inside_set():
    """Point inside Mandelbrot set should not escape."""
    result = mandelbrot_pixel(0.0, 0.0, MAX_ITER)
    assert result is None or result == MAX_ITER


def test_pixel_outside_set():
    """Point clearly outside should escape quickly."""
    result = mandelbrot_pixel(2.0, 2.0, MAX_ITER)
    assert result is not None
    assert result < MAX_ITER


def test_chunk_shape():
    """Chunk output should have correct shape."""
    out = mandelbrot_chunk(
        0, 10, TEST_N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER
    )
    assert out.shape == (10, TEST_N)


def test_chunk_values_type():
    """Chunk output should be integer array."""
    out = mandelbrot_chunk(
        0, 5, TEST_N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER
    )
    assert np.issubdtype(out.dtype, np.integer)


def test_serial_vs_chunk_equivalence():
    """Serial wrapper should match full chunk computation."""
    serial = mandelbrot_serial(
        TEST_N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER
    )

    chunk = mandelbrot_chunk(
        0, TEST_N, TEST_N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER
    )

    np.testing.assert_array_equal(serial, chunk)


def test_parallel_matches_serial():
    """Parallel result should match serial result."""
    serial = mandelbrot_serial(
        TEST_N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER
    )

    parallel = mandelbrot_parallel(
        TEST_N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER, n_workers=2
    )

    np.testing.assert_array_equal(serial, parallel)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_parallel_different_workers(workers):
    """Parallel should work with different worker counts."""
    result = mandelbrot_parallel(
        TEST_N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER, n_workers=workers
    )

    assert result.shape == (TEST_N, TEST_N)