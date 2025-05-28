"""Class for vectorized simulation of BSS process and computation of
option prices in the Rough Bergomi model"""

import numpy as np
import tqdm
import matplotlib.pyplot as plt

from time import time

from scipy.special import hyp2f1
from scipy.signal import fftconvolve

from src.parameters import ModelParameters
from src.parameters import SchemeParameters
from utils.black_scholes import bs_implied_vol_vectorized
from utils.black_scholes import bs

plt.style.use('ggplot')
rng = np.random.default_rng()
np.set_printoptions(precision=5, suppress=True)


class HybridSchemeVec:
    def __init__(self, scheme_parameters, model_parameters, kappa: int):
        """
        Parameters
        ----------
        scheme_parameters: dataclass to hold hyper parameters for the scheme
        model_parameters: dataclass as createed above with parameter for the Rough Bergomi model
        kappa: integer to control number of extra terms in the Volterra expansion
        """
        ### Hyperparameters for the hybrid scheme
        self.n_paths = scheme_parameters.n_paths
        self.n_time_steps = scheme_parameters.n_time_steps
        self.terminal_time = scheme_parameters.terminal_time
        self.kappa = kappa

        self.time_step_len = self.terminal_time / self.n_time_steps
        self.N = self.n_time_steps + 1
        self.time_steps = np.linspace(0, self.terminal_time, self.n_time_steps)

        self.model_parameters = model_parameters
        self.alpha = model_parameters.alpha

        self.cholesky_matrix = self._compute_cholesky_matrix()
        self.gamma = self._compute_gamma()
        self.rv = self.generate_random_variables()
        self.dW_1, self.dW_kappas = self.compute_processes()

    def _compute_cholesky_matrix(self) -> np.ndarray:
        """Computes the cholesky decomposition of the covariance matrix"""
        cov_matrix = np.zeros((self.kappa + 1, self.kappa + 1))
        cov_matrix[0, 0] = self.time_step_len

        for j in range(2, self.kappa + 2):
            first_row_column_value = ((j - 1) ** (self.alpha + 1) - (j - 2) ** (self.alpha + 1)) * (self.time_step_len ** (self.alpha + 1)) / (
                        self.alpha + 1)
            cov_matrix[0, j - 1] = first_row_column_value
            cov_matrix[j - 1, 0] = first_row_column_value

            diagonal = ((j - 1) ** (2 * self.alpha + 1) - (j - 2) ** (2 * self.alpha + 1)) * (self.time_step_len ** (2 * self.alpha + 1)) / (
                        2 * self.alpha + 1)
            cov_matrix[j - 1, j - 1] = diagonal

        for k in range(2, self.kappa + 2):
            for j in range(2, k):
                outside_factor = (self.time_step_len ** (2 * self.alpha + 1)) / (self.alpha + 1)

                first_term = ((j - 1) ** (self.alpha + 1)) * ((k - 1) ** self.alpha) * hyp2f1(-self.alpha, 1, self.alpha + 2,
                                                                                    (j - 1) / (k - 1))

                second_term = ((j - 2) ** (self.alpha + 1)) * ((k - 2) ** self.alpha) * hyp2f1(-self.alpha, 1, self.alpha + 2,
                                                                                     (j - 2) / (k - 2))

                cov_matrix[j - 1, k - 1] = outside_factor * (first_term - second_term)
                cov_matrix[k - 1, j - 1] = outside_factor * (first_term - second_term)

        return np.linalg.cholesky(cov_matrix)

    def _compute_gamma(self) -> np.ndarray:
        """Computes gammas used for convolution in the Hybrid Scheme"""
        ks = np.arange(2, self.N)
        g_of_b_star = ((((ks ** (self.alpha + 1) - (ks - 1) ** (self.alpha + 1)) / (self.alpha + 1)) ** (
                    1 / self.alpha)) / self.n_time_steps) ** self.alpha

        return g_of_b_star

    def generate_random_variables(self) -> np.ndarray:
        """Generates the correlated random variables using the Cholesky matrix"""
        return np.random.randn(self.n_paths, self.n_time_steps, self.kappa + 1)

    def change_random_variables(self):
        """Changes the random variables used in the sche,e"""
        self.rv = self.generate_random_variables()

    def compute_processes(self) -> tuple:
        """Generates the correlated random variables using the Cholesky matrix"""
        dW = self.rv @ self.cholesky_matrix.T
        return dW[:, :, 0], dW[:, :, 1:]

    def change_processes(self):
        """Changes the random variables"""
        self.dW_1, self.dW_kappas = self.compute_processes()

    def change_alpha(self, new_alpha: float):
        """Changes alpha"""
        self.alpha = new_alpha
        self.model_parameters.alpha = new_alpha

        self.cholesky_matrix = self._compute_cholesky_matrix()
        self.gamma = self._compute_gamma()

    def compute_bss(self) -> np.ndarray:
        """Using the random variables and gamma variable the Riemann Liouville fractional Brownian motions are generated"""

        X = self.dW_1
        gamma = np.zeros(1 + self.n_time_steps)
        gamma[2:] = self.gamma

        X_2 = np.zeros((self.n_paths, len(X[0,:]) + len(gamma) - 1))
        for i in range(self.n_paths):
            X_2[i,:] = np.convolve(gamma, X[i,:])
        X_2 = X_2[:,:1 + self.n_time_steps]

        for i in range(self.kappa):
            if i == 0:
                X_2 += np.concatenate([np.zeros((self.n_paths, 1)), self.dW_kappas[:, :, i]], axis=1)
            else:
                X_2 += np.concatenate([np.zeros((self.n_paths, 1)), self.dW_kappas[:, :, i]], axis=1) * self.time_step_len

        return X_2

    def compute_transformed_bss(self, bss: np.ndarray = None) -> np.ndarray:
        """Computes the Volterra process by """
        if bss is None:
            bss = self.compute_bss()

        V = np.sqrt(2 * self.alpha + 1) * bss
        return V

    def compute_variance_process(self, transformed_bss: np.ndarray = None) -> np.ndarray:
        """Computes the variance process of the Rough Bergomi model"""
        start_variance, eta = self.model_parameters.start_variance, self.model_parameters.eta
        if transformed_bss is None:
            transformed_bss = self.compute_transformed_bss()

        return start_variance * np.exp(eta * transformed_bss
                                       - 0.5 * (eta ** 2) * self.time_steps ** (2 * self.alpha + 1))

    def compute_terminal_price(self, variance_process: np.ndarray = None) -> np.ndarray:
        """Computes the price process of the Rough Bergomi model"""
        if variance_process is None:
            variance_process = self.compute_variance_process()

        start_price, rho = self.model_parameters.start_price, self.model_parameters.rho
        independent_increments = rng.standard_normal(size=(self.n_paths, self.n_time_steps)) * np.sqrt(self.time_step_len)

        increments_pricing_rvs = (rho * self.dW_1[:, self.N: self.n_time_steps + self.N] +
                                  np.sqrt(1 - rho ** 2) * independent_increments)

        return start_price * np.exp(np.sum(np.sqrt(variance_process) * increments_pricing_rvs -
                                           0.5 * variance_process * self.time_step_len, axis=1))

    def compute_option_payoff(self, strikes: np.ndarray, terminal_prices: np.ndarray = None):
        """Computes option payoffs terminal"""
        if terminal_prices is None:
            terminal_prices = self.compute_terminal_price()

        return np.maximum(terminal_prices - strikes.reshape(-1, 1), 0)

    def compute_cond_payoff(self, strikes):
        """Computes the option payoffs using the conditional estimator"""
        start_price, rho = self.model_parameters.start_price, self.model_parameters.rho
        V = self.compute_variance_process()
        integral = np.sum(rho * np.sqrt(V) * self.dW_1[:, self.N: self.n_time_steps + self.N]
                          - 0.5 * rho ** 2 * V * self.time_step_len, axis=1)
        S1_T = start_price * np.exp(integral)
        QV = np.sum(V, axis=1) * self.time_step_len
        X = bs(S1_T, strikes.reshape(-1, 1), (1 - rho ** 2) * QV)
        return X

    def compute_control_payoff(self, strikes):
        """Computes the option payoffs using the control estimator"""
        start_price, rho = self.model_parameters.start_price, self.model_parameters.rho
        V = self.compute_variance_process()
        terminal_prices = self.compute_terminal_price(V)

        X = self.compute_cond_payoff(strikes, terminal_prices)
        QV = np.sum(V, axis=1) * self.time_step_len
        Q = np.max(QV) + 1e-9

        Y = bs(terminal_prices, strikes.reshape(-1, 1), (Q - QV))
        eY = bs(start_price, strikes.reshape(-1, 1), Q)

        c = np.zeros_like(strikes)
        for i in range(len(strikes)):
            cov_mat = np.cov(X[i, :], Y[i, :])
            if cov_mat[1, 1] == 0:
                c[i] = c[i - 1]
            else:
                c[i] = - cov_mat[0, 1] / cov_mat[1, 1]

        # Payoffs, prices and volatilities
        payoffs = terminal_prices + c.reshape(-1, 1) * (Y - eY)
        return payoffs



    def turbo_charging(self, strikes: np.ndarray):
        """Computes the option payoff using the mixed estimator"""
        start_price, rho = self.model_parameters.start_price, self.model_parameters.rho

        V = self.compute_variance_process()

        integral = np.sum(rho * np.sqrt(V) * self.dW_1[:, self.N: self.n_time_steps + self.N]
                          - 0.5 * rho ** 2 * V * self.time_step_len, axis=1)
        S1_T = start_price * np.exp(integral)

        QV = np.sum(V, axis=1) * self.time_step_len
        Q = np.max(QV) + 1e-9

        X = bs(S1_T, strikes.reshape(-1, 1), (1 - rho ** 2) * QV)
        Y = bs(S1_T, strikes.reshape(-1, 1), rho ** 2 * (Q - QV))
        eY = bs(start_price, strikes.reshape(-1, 1), rho ** 2 * Q)

        c = np.zeros_like(strikes)
        for i in range(len(strikes)):
            cov_mat = np.cov(X[i,:], Y[i,:])
            if cov_mat[1, 1] == 0:
                c[i] = c[i-1]
            else:
                c[i] = - cov_mat[0, 1] / cov_mat[1, 1]

        # Payoffs, prices and volatilities
        payoffs = X + c.reshape(-1, 1) * (Y - eY)

        return payoffs





















