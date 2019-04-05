from scipy.io import loadmat
import numpy as np


def load_matlab_file(filename: str, var_names: list) -> list:
    data = loadmat(filename, matlab_compatible=True)
    return [
        data[name] for name in data.keys() if name in var_names
    ]


def transpose_matlab_array(A, shape: tuple):
    return np.vstack([
        A[i].reshape(*shape).transpose().reshape(-1) for i in range(len(A))
    ])
