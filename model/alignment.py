"""OAS covariance alignment of one unlabeled recording batch."""
import numpy as np
from sklearn.covariance import oas
from model.riemannian import riemannian_mean,spd_power


def align_recording(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=3 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Require a finite nonempty recording batch")
    x=x-x.mean(axis=-1,keepdims=True)
    estimates=[oas(trial.T,assume_centered=False) for trial in x]
    covariance=np.stack([value[0] for value in estimates])
    covariance += np.eye(x.shape[1])*1e-12
    reference,diagnostic=riemannian_mean(covariance,tolerance=1e-8,max_iterations=50)
    aligned=spd_power(reference,-.5) @ x
    diagnostic["oas_shrinkage_median"]=float(np.median([value[1] for value in estimates]))
    if not np.isfinite(aligned).all():
        raise ValueError("Nonfinite recording alignment")
    return aligned.astype(np.float32),reference,diagnostic
