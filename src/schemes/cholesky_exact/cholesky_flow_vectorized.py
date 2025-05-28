"""
Module to simulate the fractional Brownian motion via the exact Cholesky method.
"""
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from time import time
from scipy.special import hyp2f1
from scipy.integrate import quad


from utils.black_scholes import bs_implied_vol_vectorized
from utils.black_scholes import bs
from utils.timing import timed
from src.parameters import ModelParameters
from src.parameters import SchemeParameters

plt.style.use('ggplot')
np.set_printoptions(suppress=True, precision=3)


rng = np.random.default_rng()


class CholeskySchemeVec:
    def __init__(self, scheme_parameters, model_parameters):
        """
        Parameters
        ----------
        scheme_parameters : dataclass created in the parameters script with parameters for the scheme
        model_paramters: dataclass created in parameters with paramters for the Rough Bergomi model
        """
        ### Hyperparameters for the Cholesky scheme
        self.n_paths = scheme_parameters.n_paths
        self.n_time_steps = scheme_parameters.n_time_steps
        self.terminal_time = scheme_parameters.terminal_time

        self.time_step_len = self.terminal_time / self.n_time_steps
        self.time_steps = np.linspace(0, self.terminal_time, self.n_time_steps)

        self.model_parameters = model_parameters
        self.alpha = model_parameters.alpha

        self.cholesky_matrix = self._compute_cholesky_matrix()
        self.cholesky_matrix_turbo = self._compute_turbo_cholesky_matrix()
        self.rv = self.generate_random_variables()

        self.processes = self.compute_processes()



    def _compute_g(self, x):
        """g function for joint matrix"""
        gamma = - self.alpha
        return 2 * (self.alpha + 0.5) * (
                    1/(1-gamma) * x**(-gamma) + gamma/(1-gamma) * x ** (-1-gamma) * 1/(2-gamma) * hyp2f1(
                        1, 1+gamma, 3-gamma,1/x
                    )
                )


    def _compute_autocovariance(self, t1: float, t2: float) -> float:
        """Computes the autocovariance of two Riemann Liouville integrals. Assumes t2 > t1"""
        factor = (t1 ** (self.alpha + 1) * t2 ** self.alpha) / (self.alpha + 1)
        return factor * hyp2f1(-self.alpha, 1, self.alpha + 2, t1 / t2)

    def _compute_pricing_covariance(self) -> np.ndarray:
        """Computes the part of the covariance matrix related to the pricing variables"""
        time_steps = np.linspace(self.time_step_len, self.terminal_time, self.n_time_steps)
        i_idx, j_idx = np.meshgrid(np.arange(self.n_time_steps), np.arange(self.n_time_steps), indexing="ij")

        return np.where(j_idx <= i_idx, time_steps[j_idx], time_steps[i_idx])

    def _compute_fractional_brownian_motion_covariance(self):
        """Computes the part of the covariance matrix related solely to the fractional Brownian motion"""
        fbm_cov = np.zeros((self.n_time_steps, self.n_time_steps))
        fbm_cov[0, 0] = self.time_step_len ** (2 * self.alpha + 1)

        time_steps = np.linspace(self.time_step_len, self.terminal_time, self.n_time_steps)

        for j, u in enumerate(time_steps):
            fbm_cov[j,j] = u ** (2 * self.alpha + 1)

            for i, v in enumerate(time_steps):
                if i < j:
                    fbm_cov[j, i] = v ** (2 * self.alpha + 1) * self._compute_g(u / v)
                else:
                    fbm_cov[j, i] = u ** (2 * self.alpha + 1) * self._compute_g(v / u)

        return fbm_cov


    def _compute_joint_covariance(self):
        """Computes the joint covariance structure of the fractional Brownian motion and the pricing Brownian motion"""
        rho = self.model_parameters.rho
        Dh = np.sqrt(2 * self.alpha + 1) / (self.alpha + 1)

        minimum_matrix = self._compute_pricing_covariance()
        time_steps = np.linspace(self.time_step_len, self.terminal_time, self.n_time_steps)

        return rho * Dh * (time_steps ** (self.alpha + 1) - (time_steps - minimum_matrix) ** (self.alpha + 1))

    @timed
    def _compute_cholesky_matrix(self):
        """Computes the covariance matrix of the time increments of the fractional brownian motion"""
        cov_matrix = np.zeros((2 * self.n_time_steps, 2 * self.n_time_steps))

        cov_matrix[:self.n_time_steps, :self.n_time_steps] = self._compute_pricing_covariance()
        cov_matrix[self.n_time_steps:, self.n_time_steps: ] = self._compute_fractional_brownian_motion_covariance()

        joint_matrix = self._compute_joint_covariance()
        cov_matrix[self.n_time_steps:, :self.n_time_steps] = joint_matrix.T
        cov_matrix[:self.n_time_steps, self.n_time_steps:] = joint_matrix

        return np.linalg.cholesky(cov_matrix)

    @timed
    def _compute_turbo_cholesky_matrix(self):
        """Computes the cholesky matrix using integration"""
        cov_matrix = np.zeros((self.n_time_steps, self.n_time_steps))

        time_steps_extended = np.linspace(self.time_step_len, self.terminal_time, self.n_time_steps)

        for i, s in enumerate(time_steps_extended):
            cov_matrix[i, i] = 1 / (2 * self.alpha + 1) * s ** (2 * self.alpha + 1)

            for j, t in enumerate(time_steps_extended):

                if j > i:
                    coefficient = self._compute_auto_covariance(s, t)
                    cov_matrix[i, j] = coefficient
                    cov_matrix[j, i] = coefficient

        return np.linalg.cholesky(cov_matrix)

    def _compute_auto_covariance(self, s: float, t: float) -> float:
        """Computes the autocovariance of two Riemann Liouville integrals. Assumes t > s"""

        # compute the integral from 0 to s of (s-u) ** alpha * (t-u) ** alpha du using quad
        def integrand(u):
            return (s - u) ** self.alpha * (t - u) ** self.alpha

        return quad(integrand, 0, s)[0]

    def generate_random_variables(self):
        """Simulates standard normal random variables"""
        dimensions = (self.n_paths, 2 * self.n_time_steps)
        return rng.standard_normal(size=dimensions)

    def change_random_variables(self):
        """Changes the random variables used in the scheme using the method generate random variasbles"""
        self.rv = self.generate_random_variables()

    def change_alpha(self, new_alpha: float):
        """New alpha replaces the old one"""
        self.alpha = new_alpha
        self.model_parameters.alpha = new_alpha

        self.cholesky_matrix = self._compute_cholesky_matrix()

    def compute_processes(self, ):
        """Computes the processes relevant in this scheme, both the Brownian motion relevant for pricing and
        the fractional Brownian motion"""
        return np.dot(self.rv, self.cholesky_matrix.T)

    def change_processes(self):
        """Changes the processes used in the scheme"""
        self.processes = self.compute_processes()

    def compute_bss(self) -> np.ndarray:
        """Using the initialized variables we compute the fractional process"""
        bss = self.processes[:, self.n_time_steps:]
        bss[:, 1:] = bss[:, :-1]
        bss[:, 0] = 0
        return bss

    def compute_transformed_bss(self, bss_process: np.ndarray = None) -> np.ndarray:
        """Computes the Volterra process by """
        if bss_process is None:
            bss_process = self.compute_bss()
        V = bss_process
        return V

    def compute_variance_process(self, volterra_process: np.ndarray = None) -> np.ndarray:
        """Computes the variance process of the Rough Bergomi model"""
        start_variance, eta = self.model_parameters.start_variance, self.model_parameters.eta
        if volterra_process is None:
            volterra_process = self.compute_transformed_bss()

        return start_variance * np.exp(eta * volterra_process
                                       - 0.5 * (eta ** 2) * self.time_steps ** (2 * self.alpha + 1))

    def compute_terminal_price(self, variance_process: np.ndarray = None) -> np.ndarray:
        """Computes the price process of the Rough Bergomi model"""
        if variance_process is None:
            variance_process = self.compute_variance_process()

        start_price, rho = self.model_parameters.start_price, self.model_parameters.rho
        increments_pricing_rvs = np.diff(self.processes[:, :self.n_time_steps], prepend=0)

        return start_price * np.exp(np.sum(np.sqrt(variance_process) *  increments_pricing_rvs -
                                           0.5 * variance_process * self.time_step_len, axis=1))

    def compute_option_payoff(self, strikes: np.ndarray, terminal_prices: np.ndarray = None):
        """Computes option payoffs terminal"""
        if terminal_prices is None:
            terminal_prices = self.compute_terminal_price()

        return np.maximum(terminal_prices - strikes.reshape(-1, 1), 0)

    def compute_option_payoff_with_parity(self, strikes, terminal_prices=None):
        """Computes option payoffs with the put call parity estimator"""
        if terminal_prices is None:
            terminal_prices = self.compute_terminal_price()
        S_0 = self.model_parameters.start_price
        K = strikes.reshape(-1, 1)
        S = terminal_prices

        call_payoffs = np.maximum(S - K, 0)

        put_payoffs = np.maximum(K - S, 0)
        call_payoffs_parity = put_payoffs + S_0 - K

        variance_calls = np.var(call_payoffs, axis=1)
        variance_calls_parity = np.var(call_payoffs_parity, axis=1)

        indexes = variance_calls_parity < variance_calls

        call_payoffs_combined = np.zeros_like(call_payoffs)
        call_payoffs_combined[indexes] = call_payoffs_parity[indexes]
        call_payoffs_combined[~indexes] = call_payoffs[~indexes]

        return call_payoffs_combined

    def compute_antithetic_payoff(self, strikes):
        """Computes the option payoffs using the antithetic estimator"""
        rho, start_price, start_variance, eta = (
            self.model_parameters.rho, self.model_parameters.start_price, self.model_parameters.start_variance,
            self.model_parameters.eta)

        n_antithetic = self.n_paths // 2
        rv = rng.standard_normal(size=(n_antithetic, 2 * self.n_time_steps))
        processes = np.dot(rv, self.cholesky_matrix.T)
        bss = processes[:, self.n_time_steps:]
        increments_pricing_rvs = np.diff(processes[:, :self.n_time_steps], prepend=0)
        increments_pricing_rvs_stacked = np.concatenate([increments_pricing_rvs, -increments_pricing_rvs], axis=0)

        bss[:, 1:] = bss[:, :-1]
        bss[:, 0] = 0
        volterra_process = np.concatenate([bss, -bss], axis=0)

        variance_proceses = start_variance * np.exp(eta * volterra_process
                                       - 0.5 * (eta ** 2) * self.time_steps ** (2 * self.alpha + 1))

        ST = start_price * np.exp(np.sum(np.sqrt(variance_proceses) *  increments_pricing_rvs_stacked -
                                           0.5 * variance_proceses * self.time_step_len, axis=1))

        return np.maximum(ST - strikes.reshape(-1, 1), 0)


    def compute_control_payoff(self, strikes: np.ndarray):
        """Computes the control payoff payoffs"""
        rho, start_price, start_variance, eta = (
            self.model_parameters.rho, self.model_parameters.start_price, self.model_parameters.start_variance,
            self.model_parameters.eta)

        rv = rng.standard_normal(size=(self.n_paths, 2 * self.n_time_steps))
        processes = np.dot(rv, self.cholesky_matrix.T)
        bss = processes[:, self.n_time_steps:]
        increments_pricing_rvs = np.diff(processes[:, :self.n_time_steps], prepend=0)

        bss[:, 1:] = bss[:, :-1]
        bss[:, 0] = 0
        volterra_process = bss

        variance_proceses = start_variance * np.exp(eta * volterra_process
                                       - 0.5 * (eta ** 2) * self.time_steps ** (2 * self.alpha + 1))

        terminal_prices = start_price * np.exp(np.sum(np.sqrt(variance_proceses) *  increments_pricing_rvs -
                                           0.5 * variance_proceses * self.time_step_len, axis=1))

        QV = np.sum(variance_proceses, axis=1) * self.time_step_len
        Q = np.max(QV) + 1e-9

        X = self.compute_option_payoff(strikes, terminal_prices)
        Y = bs(terminal_prices, strikes.reshape(-1, 1), (Q - QV))
        eY = bs(start_price, strikes.reshape(-1, 1), Q)

        c = np.zeros_like(strikes)
        for i in range(len(strikes)):
            cov_mat = np.cov(X[i, :], Y[i, :])
            c[i] = - cov_mat[0, 1] / cov_mat[1, 1]

        # Payoffs, prices and volatilities
        control_payoffs = X + c.reshape(-1, 1) * (Y - eY)

        return control_payoffs


    def compute_cond_payoff(self, strikes: np.ndarray):
        """Uses the turbo charging method to compute the option prices"""
        start_price, rho, eta, start_variance = (
            self.model_parameters.start_price, self.model_parameters.rho, self.model_parameters.eta, self.model_parameters.start_variance)

        random_variables = rng.standard_normal(size=(self.n_paths, self.n_time_steps))
        bss = np.dot(random_variables, self.cholesky_matrix_turbo.T)
        bss[:, 1:] = bss[:, :-1]
        bss[:, 0] = 0

        transformed_bss = np.sqrt(2 * self.alpha + 1) * bss

        variance_processes = start_variance * np.exp(eta * transformed_bss
                                                     - 0.5 * (eta ** 2) * self.time_steps ** (2 * self.alpha + 1))

        integral = np.sum(rho * np.sqrt(variance_processes) * random_variables * np.sqrt(self.time_step_len)
                          - 0.5 * rho ** 2 * variance_processes * self.time_step_len, axis=1)

        S1_T = start_price * np.exp(integral)
        QV = np.sum(variance_processes, axis=1) * self.time_step_len

        X = bs(S1_T, strikes.reshape(-1, 1), (1 - rho ** 2) * QV)
        return X

    def compute_mixed_payoff(self, strikes: np.ndarray):
        """Uses the turbo charging method to compute the option prices"""
        start_price, rho, eta, start_variance = (
            self.model_parameters.start_price, self.model_parameters.rho, self.model_parameters.eta, self.model_parameters.start_variance)

        random_variables = rng.standard_normal(size=(self.n_paths, self.n_time_steps))
        bss = np.dot(random_variables, self.cholesky_matrix_turbo.T)
        bss[:, 1:] = bss[:, :-1]
        bss[:, 0] = 0

        transformed_bss = np.sqrt(2 * self.alpha + 1) * bss

        variance_processes = start_variance * np.exp(eta * transformed_bss
                                                     - 0.5 * (eta ** 2) * self.time_steps ** (2 * self.alpha + 1))

        integral = np.sum(rho * np.sqrt(variance_processes) * random_variables * np.sqrt(self.time_step_len)
                          - 0.5 * rho ** 2 * variance_processes * self.time_step_len, axis=1)

        S1_T = start_price * np.exp(integral)

        QV = np.sum(variance_processes, axis=1) * self.time_step_len
        Q = np.max(QV) + 1e-9

        X = bs(S1_T, strikes.reshape(-1, 1), (1 - rho ** 2) * QV)
        Y = bs(S1_T, strikes.reshape(-1, 1), rho ** 2 * (Q - QV))
        eY = bs(start_price, strikes.reshape(-1, 1), rho ** 2 * Q)

        c = np.zeros_like(strikes)
        for i in range(len(strikes)):
            cov_mat = np.cov(X[i, :], Y[i, :])
            c[i] = - cov_mat[0, 1] / cov_mat[1, 1]

        # Payoffs, prices and volatilities
        mixed_payoffs = X + c.reshape(-1, 1) * (Y - eY)

        return mixed_payoffs

    def compute_antithetic_mixed_payoff(self, strikes: np.ndarray):
        """Uses the turbo charging method to compute the option prices with antithetic sampling"""
        start_price, rho, eta, start_variance = (
            self.model_parameters.start_price, self.model_parameters.rho, self.model_parameters.eta, self.model_parameters.start_variance)

        n_antithetic = self.n_paths // 2

        random_variables = rng.standard_normal(size=(n_antithetic, self.n_time_steps))
        bss = np.dot(random_variables, self.cholesky_matrix_turbo.T)
        bss[:, 1:] = bss[:, :-1]
        bss[:, 0] = 0

        # Create the antithetic random variables
        bss_stacked = np.concatenate([bss, -bss], axis=0)
        volterra_process = np.sqrt(2 * self.alpha + 1) * bss_stacked

        rv_stacked = np.concatenate([random_variables, -random_variables], axis=0)

        variance_processes = start_variance * np.exp(eta * volterra_process
                                                     - 0.5 * (eta ** 2) * self.time_steps ** (2 * self.alpha + 1))

        integral = np.sum(rho * np.sqrt(variance_processes) * rv_stacked * np.sqrt(self.time_step_len)
                          - 0.5 * rho ** 2 * variance_processes * self.time_step_len, axis=1)

        S1_T = start_price * np.exp(integral)

        QV = np.sum(variance_processes, axis=1) * self.time_step_len
        Q = np.max(QV) + 1e-9

        X = bs(S1_T, strikes.reshape(-1, 1), (1 - rho ** 2) * QV)
        Y = bs(S1_T, strikes.reshape(-1, 1), rho ** 2 * (Q - QV))
        eY = bs(start_price, strikes.reshape(-1, 1), rho ** 2 * Q)

        c = np.zeros_like(strikes)
        for i in range(len(strikes)):
            cov_mat = np.cov(X[i, :], Y[i, :])
            c[i] = - cov_mat[0, 1] / cov_mat[1, 1]

        mixed_payoffs_antithetic = X + c.reshape(-1, 1) * (Y - eY)

        return mixed_payoffs_antithetic






