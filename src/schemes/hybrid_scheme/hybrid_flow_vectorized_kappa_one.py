"""Class for vectorized simulation of BSS process vectorized and computation of
option prices in the Rough Bergomi model using the hyperparameter kappa=1 simplifying substantially"""

import numpy as np
import tqdm
import matplotlib.pyplot as plt

from time import time

from scipy.stats import multivariate_normal as mvn

from src.parameters import ModelParameters
from src.parameters import SchemeParameters
from utils.black_scholes import bs_implied_vol_vectorized
from utils.black_scholes import bs
from utils.timing import timed

plt.style.use('ggplot')
rng = np.random.multivariate_normal
np.set_printoptions(precision=5, suppress=True)


class HybridSchemeVecKappaOne:
    def __init__(self, scheme_parameters, model_parameters):
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
        self.kappa = 1

        ### Derived hyperparameters
        self.time_step_len = self.terminal_time / self.n_time_steps
        self.time_steps = np.linspace(0, self.terminal_time, self.n_time_steps + 1)

        ### Parameters
        self.model_parameters = model_parameters
        self.alpha = model_parameters.alpha

        ### Precomputations
        self.cholesky_matrix = self._compute_cholesky_matrix()
        self.gamma = self._compute_gamma()
        self.rv = self.generate_random_variables()
        self.dW_1, self.dW_2 = self.compute_processes()

    @timed
    def _compute_cholesky_matrix(self) -> np.ndarray:
        """Computes the cholesky decomposition of the covariance matrix"""
        cov_matrix = np.array([[0., 0.], [0., 0.]])
        cov_matrix[0, 0] = 1. / self.n_time_steps
        cov_matrix[0, 1] = 1. / ((1. * self.alpha + 1) * self.n_time_steps ** (1. * self.alpha + 1))
        cov_matrix[1, 1] = 1. / ((2. * self.alpha + 1) * self.n_time_steps ** (2. * self.alpha + 1))
        cov_matrix[1, 0] = cov_matrix[0, 1]

        return np.linalg.cholesky(cov_matrix)

    def _compute_gamma(self) -> np.ndarray:
        """Computes gammas used for convolution in the Hybrid Scheme"""
        ks = np.arange(2, self.n_time_steps + 1)
        g_of_b_star = ((((ks ** (self.alpha + 1) - (ks - 1) ** (self.alpha + 1)) / (self.alpha + 1)) ** (
                    1 / self.alpha)) / self.n_time_steps) ** self.alpha

        return g_of_b_star

    def generate_random_variables(self):
        return np.random.randn(self.n_paths, self.n_time_steps, self.kappa + 1)

    def change_random_variables(self):
        self.rv = self.generate_random_variables()

    def compute_processes(self):
        dW = self.rv @ self.cholesky_matrix.T
        return dW[:, :, 0], dW[:, :, 1]

    def change_processes(self):
        """Changes the processes"""
        self.dW_1, self.dW_2 = self.compute_processes()

    def change_alpha(self, new_alpha: float):
        """Changes alpha"""
        self.alpha = new_alpha
        self.model_parameters.alpha = new_alpha

        self.cholesky_matrix = self._compute_cholesky_matrix()
        self.gamma = self._compute_gamma()

    def compute_bss(self) -> np.ndarray:
        """Using the random variables and gamma variable the Riemann Liouville fractional Brownian motions are generated"""
        X_1 = np.concatenate([np.zeros((self.n_paths, 1)), self.dW_2], axis=1)

        X = self.dW_1
        gamma = np.zeros(1 + self.n_time_steps)
        gamma[2:] = self.gamma

        X_2 = np.zeros((self.n_paths, len(X[0,:]) + len(gamma) - 1))
        for i in range(self.n_paths):
            X_2[i,:] = np.convolve(gamma, X[i,:])
        X_2 = X_2[:,:1 + self.n_time_steps]

        return X_1 + X_2


    def compute_transformed_bss(self, bss_process: np.ndarray = None) -> np.ndarray:
        """Computes the Volterra process by """
        if bss_process is None:
            bss_process = self.compute_bss()

        V = np.sqrt(2 * self.alpha + 1) * bss_process
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
        independent_increments = np.random.randn(self.n_paths, self.n_time_steps) * np.sqrt(self.time_step_len)

        increments_pricing_rvs = (rho * self.dW_1 + np.sqrt(1 - rho ** 2) * independent_increments)

        return start_price * np.exp(np.sum(np.sqrt(variance_process[:, :-1]) * increments_pricing_rvs -
                                           0.5 * variance_process[:, :-1] * self.time_step_len, axis=1))

    def compute_option_payoff(self, strikes: np.ndarray, terminal_prices: np.ndarray = None):
        """Computes option payoffs terminal"""
        if terminal_prices is None:
            terminal_prices = self.compute_terminal_price()

        return np.maximum(terminal_prices - strikes.reshape(-1, 1), 0)

    def compute_option_payoff_with_parity(self, strikes, terminal_prices=None):
        """Computes the call price via the put call parity for itm options"""
        if terminal_prices is None:
            terminal_prices = self.compute_terminal_price()
        S_0 = self.model_parameters.start_price
        K = strikes.reshape(-1, 1)  # (m,1)
        S = terminal_prices  # (1,n_paths)

        call_payoffs = np.maximum(S - K, 0)

        # compute put payoffs for lower half then convert to call
        put_payoffs = np.maximum(K - S, 0)
        call_payoffs_parity = put_payoffs + S_0 - K

        # choose the one with the lowest variance
        variance_calls = np.var(call_payoffs, axis=1)
        variance_calls_parity = np.var(call_payoffs_parity, axis=1)
        indexes = variance_calls_parity < variance_calls

        call_payoffs_combined = np.zeros_like(call_payoffs)
        call_payoffs_combined[indexes] = call_payoffs_parity[indexes]
        call_payoffs_combined[~indexes] = call_payoffs[~indexes]

        return call_payoffs_combined

    def compute_antithetic_payoff(self, strikes):
        """Computes the option payoffs using the antithetic estimator"""
        start_price, rho, start_variance, eta = (self.model_parameters.start_price, self.model_parameters.rho,
                                                 self.model_parameters.start_variance, self.model_parameters.eta)

        n_antithetic_paths = self.n_paths // 2
        rvs = np.random.randn(n_antithetic_paths, self.n_time_steps, self.kappa + 1)
        dW = rvs @ self.cholesky_matrix.T
        dW_1, dW_2 = dW[:, :, 0], dW[:, :, 1]

        X_1 = np.concatenate([np.zeros((n_antithetic_paths, 1)), dW_2], axis=1)

        X = dW_1
        gamma = np.zeros(1 + self.n_time_steps)
        gamma[2:] = self.gamma

        X_2 = np.zeros((n_antithetic_paths, len(X[0,:]) + len(gamma) - 1))
        for i in range(n_antithetic_paths):
            X_2[i,:] = np.convolve(gamma, X[i,:])
        X_2 = X_2[:,:1 + self.n_time_steps]

        bss = X_1 + X_2
        transformed_bss = np.sqrt(2 * self.alpha + 1) * bss

        volterra_process_stacked = np.concatenate([transformed_bss, -transformed_bss], axis=0)
        dW_1_stacked = np.concatenate([dW_1, -dW_1], axis=0)

        V = start_variance * np.exp(eta * volterra_process_stacked
                                       - 0.5 * (eta ** 2) * self.time_steps ** (2 * self.alpha + 1))
        independent_increments = np.random.randn(2 * n_antithetic_paths, self.n_time_steps) * np.sqrt(self.time_step_len)
        increments_pricing_rvs = (rho * dW_1_stacked + np.sqrt(1 - rho ** 2) * independent_increments)

        ST = start_price * np.exp(np.sum(np.sqrt(V[:, :-1]) * increments_pricing_rvs -
                                           0.5 * V[:, :-1] * self.time_step_len, axis=1))
        return np.maximum(ST - strikes.reshape(-1, 1), 0)

    def compute_is_payoff(self, strikes):
        """Uses Importance sampling to compute the option payoffs"""
        start_price, rho, start_variance, eta = (self.model_parameters.start_price, self.model_parameters.rho,
                                                 self.model_parameters.start_variance, self.model_parameters.eta)

        n_antithetic_paths = self.n_paths // 2
        rvs = np.random.randn(n_antithetic_paths, self.n_time_steps, self.kappa + 1)
        dW = rvs @ self.cholesky_matrix.T
        dW_1, dW_2 = dW[:, :, 0], dW[:, :, 1]

        X_1 = np.concatenate([np.zeros((n_antithetic_paths, 1)), dW_2], axis=1)

        X = dW_1
        gamma = np.zeros(1 + self.n_time_steps)
        gamma[2:] = self.gamma

        X_2 = np.zeros((n_antithetic_paths, len(X[0,:]) + len(gamma) - 1))
        for i in range(n_antithetic_paths):
            X_2[i,:] = np.convolve(gamma, X[i,:])
        X_2 = X_2[:,:1 + self.n_time_steps]

        fractional_process = X_1 + X_2
        volterra_process = np.sqrt(2 * self.alpha + 1) * fractional_process

        volterra_process_stacked = np.concatenate([volterra_process, -volterra_process], axis=0)
        dW_1_stacked = np.concatenate([dW_1, -dW_1], axis=0)

        V = start_variance * np.exp(eta * volterra_process_stacked
                                       - 0.5 * (eta ** 2) * self.time_steps ** (2 * self.alpha + 1))

        # make an array of the format of ST - strikes.respae(-1, 1) to hold pyaoffs for alle strikes and paths
        payoffs = np.zeros((len(strikes), 2 * n_antithetic_paths))

        for i, strike in enumerate(strikes):
            target_mean = (np.log(strike / start_price) + 0.5 * self.time_step_len * np.sum(V[:, :-1], axis=1)) / (np.sqrt(1-rho ** 2) * np.sum(np.sqrt(V[:, :-1]), axis=1))

            independent_increments_with_target_mean = np.random.randn(2 * n_antithetic_paths, self.n_time_steps) * np.sqrt(self.time_step_len) + target_mean.reshape(-1, 1)
            increments_pricing_rvs = (rho * dW_1_stacked + np.sqrt(1 - rho ** 2) * independent_increments_with_target_mean)

            mean_r = np.sqrt(1 - rho ** 2) * target_mean
            cov_r = rho**2 * np.eye(len(target_mean)) + self.time_step_len * (1- rho**2) * np.eye(len(target_mean))
            likelihood = mvn.pdf(increments_pricing_rvs, mean=mean_r.reshape(-1, 1), cov=cov_r)

            weights = mvn.pdf(cov=cov_r) / likelihood

            ST = start_price * np.exp(np.sum(np.sqrt(V[:, :-1]) * increments_pricing_rvs -
                                             0.5 * V[:, :-1] * self.time_step_len, axis=1))

            # ratio of standard normal densitie and normal density with target mean for all ST use standard libraty
            #ratio = (norm.pdf(x=independent_increments_with_target_mean, scale=np.sqrt(self.time_step_len))
            #         / norm.pdf(x=independent_increments_with_target_mean, loc=target_mean.reshape(-1, 1), scale=np.sqrt(self.time_step_len)))
            #ratio_product = np.prod(ratio, axis=1)


            payoff = np.maximum(ST - strike, 0) * weights
            payoffs[i,:] = payoff

        return payoffs




    def compute_cond_payoff(self, strikes):
        """Computes the option payoffs using the conditional estimator"""
        start_price, rho = self.model_parameters.start_price, self.model_parameters.rho
        self.change_random_variables()
        self.change_processes()
        V = self.compute_variance_process()[:, :self.n_time_steps]
        integral = np.sum(rho * np.sqrt(V) * self.dW_1
                          - 0.5 * rho ** 2 * V * self.time_step_len, axis=1)
        S1_T = start_price * np.exp(integral)
        QV = np.sum(V, axis=1) * self.time_step_len
        X = bs(S1_T, strikes.reshape(-1, 1), (1 - rho ** 2) * QV)
        return X

    def compute_control_payoff(self, strikes):
        """Computes the option payoffs using the control estimator"""
        start_price, rho = self.model_parameters.start_price, self.model_parameters.rho
        self.change_random_variables()
        self.change_processes()

        V = self.compute_variance_process()
        terminal_prices = self.compute_terminal_price(V)

        X = self.compute_option_payoff(strikes, terminal_prices)
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
        payoffs = X + c.reshape(-1, 1) * (Y - eY)
        return payoffs


    def compute_mixed_payoff(self, strikes: np.ndarray):
        """Computes mixed payoffs"""
        start_price, rho = self.model_parameters.start_price, self.model_parameters.rho
        self.change_random_variables()
        self.change_processes()
        V = self.compute_variance_process()[:, :self.n_time_steps]

        integral = np.sum(rho * np.sqrt(V) * self.dW_1[:, :self.n_time_steps]
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
        mixed_payoffs = X + c.reshape(-1, 1) * (Y - eY)

        return mixed_payoffs

    def compute_antithetic_mixed_payoff(self, strikes: np.array):
        """Computes the mixed payofs using antithetic sampling"""
        start_price, rho, start_variance, eta = (self.model_parameters.start_price, self.model_parameters.rho,
                                                 self.model_parameters.start_variance, self.model_parameters.eta)

        n_antithetic_paths = self.n_paths // 2
        rvs = np.random.randn(n_antithetic_paths, self.n_time_steps, self.kappa + 1)
        dW = rvs @ self.cholesky_matrix.T
        dW_1, dW_2 = dW[:, :, 0], dW[:, :, 1]

        X_1 = np.concatenate([np.zeros((n_antithetic_paths, 1)), dW_2], axis=1)

        X = dW_1
        gamma = np.zeros(1 + self.n_time_steps)
        gamma[2:] = self.gamma

        X_2 = np.zeros((n_antithetic_paths, len(X[0,:]) + len(gamma) - 1))
        for i in range(n_antithetic_paths):
            X_2[i,:] = np.convolve(gamma, X[i,:])
        X_2 = X_2[:,:1 + self.n_time_steps]

        fractional_process = X_1 + X_2
        volterra_process = np.sqrt(2 * self.alpha + 1) * fractional_process

        volterra_process_stacked = np.concatenate([volterra_process, -volterra_process], axis=0)
        dW_1_stacked = np.concatenate([dW_1, -dW_1], axis=0)

        V = start_variance * np.exp(eta * volterra_process_stacked
                                       - 0.5 * (eta ** 2) * self.time_steps ** (2 * self.alpha + 1))
        V = V[:, :self.n_time_steps]

        integral = np.sum(rho * np.sqrt(V) * dW_1_stacked[:, :self.n_time_steps]
                          - 0.5 * rho ** 2 * V * self.time_step_len, axis=1)
        S1_T = start_price * np.exp(integral)

        QV = np.sum(V, axis=1) * self.time_step_len
        Q = np.max(QV) + 1e-9

        X = bs(S1_T, strikes.reshape(-1, 1), (1 - rho ** 2) * QV)
        Y = bs(S1_T, strikes.reshape(-1, 1), rho ** 2 * (Q - QV))
        eY = bs(start_price, strikes.reshape(-1, 1), rho ** 2 * Q)

        c = np.zeros_like(strikes)
        for i in range(len(strikes)):
            cov_mat = np.cov(X[i, :], Y[i, :])
            if cov_mat[1, 1] == 0:
                c[i] = c[i - 1]
            else:
                c[i] = - cov_mat[0, 1] / cov_mat[1, 1]

        # Payoffs, prices and volatilities
        mixed_payoffs = X + c.reshape(-1, 1) * (Y - eY)

        return mixed_payoffs
