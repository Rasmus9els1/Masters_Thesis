"""
Module to simulate the brownian semistationary process using the Fourier approach.
Generalizes easily to semistationary Lévy processes.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

from numpy.exceptions import ComplexWarning
from scipy.integrate import quad

warnings.filterwarnings("ignore", category=ComplexWarning)


class FourierScheme:
    """
    Class for simulating the BSS process using the Fourier approach.
    """
    def __init__(self, n_time_steps: int, total_time: float, tau_0: float, n_fourier_terms: int, lambd: float, alpha: float,
                 seed: int = None):
        """
        Initializes the simulation parameters.
        """
        self.n_time_steps = n_time_steps
        self.total_time = total_time
        self.tau_0 = tau_0
        self.n_fourier_terms = n_fourier_terms
        self.lambd = lambd
        self.alpha = alpha

        if seed is not None:
            np.random.seed(seed)

        self.delta_t = total_time / (n_time_steps + 1)
        self.time_steps = np.linspace(0.00001, total_time, n_time_steps + 1)

        self.y = np.arange(0.00001, n_fourier_terms) * np.pi / tau_0

        h_hat_vals = self.h_hat_lambda()
        self.a_n = h_hat_vals / tau_0

        self.random_variables = self.sim_brownian_increments()

    def g(self, x: float) -> float:
        """Kernel function g(x) = x^alpha."""
        return x ** self.alpha

    def h(self, x: np.ndarray) -> np.ndarray:
        """Piecewise function from eq. (3.3)."""
        return np.where(np.abs(x) <= self.tau_0, self.g(np.abs(x)), 0)

    def h_lamb(self, x: np.ndarray) -> np.ndarray:
        """Piecewise function from eq. (3.4) with exponential factor."""
        return self.h(x) * np.exp(self.lambd * np.abs(x))

    def h_hat_lambda(self) -> np.ndarray:
        """Fourier transform of the piecewise function from eq. (3.4)."""
        delta_y = self.y[1] - self.y[0]

        h_lamb_vals = self.h_lamb(self.y)
        fft_vals = np.fft.fft(h_lamb_vals)

        return fft_vals


    def calc_X_hat_new(self, X_hat_prev: np.ndarray, levy_increment: float, y: np.ndarray) -> np.ndarray:
        """Simulates the next Fourier coefficient values."""
        return np.exp((-self.lambd + 1j * y) * self.delta_t) * (X_hat_prev + levy_increment)

    def chg_alpha(self, new_alpha: float):
        """Changes the value of the roughness parameter."""
        self.alpha = new_alpha
        h_hat_vals = self.h_hat_lambda()
        self.a_n = h_hat_vals / self.tau_0

    def sim_brownian_increments(self):
        """Simulates the Brownian increments."""
        return np.random.normal(size=self.n_time_steps + 1) * np.sqrt(self.delta_t)

    def chg_random_vars(self):
        """Changes the random variables used in the simulation."""
        self.random_variables = self.sim_brownian_increments()

    def sim(self) -> np.ndarray:
        """Simulates the process over the time grid."""
        X_hat_prev = np.zeros(self.n_fourier_terms, dtype=complex)
        process = np.zeros(self.n_time_steps + 1)

        for t in range(1, self.n_time_steps + 1):
            X_hat_new = self.calc_X_hat_new(X_hat_prev, self.random_variables[t], self.y)
            X_hat_prev = X_hat_new
            process[t] = 0.5 * self.a_n[0] * X_hat_new[0] +  np.real(np.sum(self.a_n[1:] * X_hat_new[1:]))

        return process





