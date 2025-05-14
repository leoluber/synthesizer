""" Asymmetric Voigt linshape for fitting perovskite nanocrystal PL spectra """


import numpy as np
from scipy.special import voigt_profile


# bound recommendations
PARAMETERS_SPLIT_VOIGT = ["x_0", "sigma", "gamma_1", "gamma_2", "y"]
BOUNDS_SPLIT_VOIGT = ([2000, 10, 10, 3, -0.01], [3000, 100, 100, 100, 0.01])



def Voigt(x, x_0, sigma, gamma, A = 1, y = 0):
    """
        Voigt function
    """
    voigt = voigt_profile(x - x_0, sigma, gamma)

    # normalize
    voigt = voigt / max(voigt)
    return A* voigt + y



def SplitVoigt(x, x_0, sigma, gamma_1, gamma_2, y = 0):
    """
        Return Voigt 1 if x < x_0, else Voigt 2
    """
    return np.where(x < x_0, Voigt(x, x_0, sigma, gamma_1), Voigt(x, x_0, sigma, gamma_2)) + y
