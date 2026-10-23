"""Label-free affine-invariant covariance alignment for offline participant batches."""
import numpy as np


def symmetric_function(matrix, function):
    matrix = (matrix + matrix.swapaxes(-1, -2)) / 2
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * function(values)[..., None, :]) @ vectors.swapaxes(-1, -2)


def spd_power(matrix, power):
    return symmetric_function(matrix, lambda values: np.maximum(values, 1e-12) ** power)


def riemannian_mean(covariances, tolerance=1e-5, max_iterations=50):
    reference = covariances.mean(axis=0)
    step, previous = .5, float("inf")
    for iteration in range(1, max_iterations + 1):
        inverse = spd_power(reference, -.5)
        relative = inverse @ covariances @ inverse
        tangent = symmetric_function(relative, lambda values: np.log(np.maximum(values, 1e-12))).mean(axis=0)
        residual = float(np.linalg.norm(tangent, ord="fro"))
        if residual < tolerance:
            return reference, {"iterations": iteration, "residual": residual, "converged": True}
        if residual > previous:
            step *= .5
        root = spd_power(reference, .5)
        reference = root @ symmetric_function(step * tangent, np.exp) @ root
        previous = residual
    return reference, {"iterations": max_iterations, "residual": residual, "converged": False}


def align_participant(x, shrinkage=.001):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Require finite nonempty trial/channel/sample signals")
    centered = x - x.mean(axis=-1, keepdims=True)
    covariance = centered @ centered.swapaxes(-1, -2) / centered.shape[-1]
    dim = x.shape[-2]
    scale = np.maximum(np.trace(covariance, axis1=-2, axis2=-1) / dim, 1e-12)
    covariance = (1-shrinkage) * covariance + shrinkage * scale[:, None, None] * np.eye(dim)
    reference, diagnostics = riemannian_mean(covariance)
    whitening = spd_power(reference, -.5)
    aligned = whitening @ centered
    if not np.isfinite(aligned).all():
        raise ValueError("Nonfinite aligned signals")
    return aligned.astype(np.float32), reference, diagnostics
