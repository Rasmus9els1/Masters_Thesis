"""Black Scholes utility functions"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def bs(F, K, V):
    """
    Returns the Black call price for given forward, strike and integrated variance.
    """
    sv = np.sqrt(V)
    d1 = np.log(F / K) / sv + 0.5 * sv
    d2 = d1 - sv
    P = F * norm.cdf(d1) - K * norm.cdf(d2)
    return P


def bs_call_formula(S0: float, strikes: np.ndarray, maturities: np.ndarray, rates: np.ndarray, dividends: np.ndarray,
                    volatility: np.ndarray) -> np.ndarray:
    """Calculate the Black-Scholes call option pricing formula vectorized for multiple strikes and maturities."""
    d1 = (np.log(S0 / strikes) + (rates - dividends + 0.5 * volatility ** 2) * maturities) / (
            volatility * np.sqrt(maturities))
    d2 = d1 - volatility * np.sqrt(maturities)
    return S0 * np.exp(-dividends * maturities) * norm.cdf(d1) - strikes * np.exp(-rates * maturities) * norm.cdf(d2)


def bs_call_formula_vectorized(S0: float, strikes: np.ndarray, maturities: np.ndarray, rates: np.ndarray,
                               dividends: np.ndarray, volatilities: np.ndarray) -> np.ndarray:
    """Calculate the Black-Scholes call option pricing formula vectorized for multiple strikes and maturities."""
    #volatilities = volatilities[:, np.newaxis]
    d1 = (np.log(S0 / strikes) + (rates - dividends + 0.5 * volatilities ** 2) * maturities) / (
            volatilities * np.sqrt(maturities))
    d2 = d1 - volatilities * np.sqrt(maturities)
    call_prices = (S0 * np.exp(-dividends * maturities) * norm.cdf(d1) -
                   strikes * np.exp(-rates * maturities) * norm.cdf(d2))
    return call_prices

def bs_call_formula_vectorized_IV(S0: float, strikes: np.ndarray, maturities: np.ndarray, rates: np.ndarray,
                               dividends: np.ndarray, volatilities: np.ndarray) -> np.ndarray:
    """Calculate the Black-Scholes call option pricing formula vectorized for multiple strikes and maturities."""
    volatilities = volatilities[:, np.newaxis]
    d1 = (np.log(S0 / strikes) + (rates - dividends + 0.5 * volatilities ** 2) * maturities) / (
            volatilities * np.sqrt(maturities))
    d2 = d1 - volatilities * np.sqrt(maturities)
    call_prices = (S0 * np.exp(-dividends * maturities) * norm.cdf(d1) -
                   strikes * np.exp(-rates * maturities) * norm.cdf(d2))

    return call_prices


def bs_put_formula(S0: float, strikes: np.ndarray, maturities: np.ndarray, rates: np.ndarray, dividends: np.ndarray,
                   volatility: np.ndarray) -> np.ndarray:
    """Calculate the Black-Scholes put option pricing formula vectorized for multiple strikes and maturities."""
    d1 = (np.log(S0 / strikes) + (rates - dividends + 0.5 * volatility ** 2) * maturities) / (
            volatility * np.sqrt(maturities))
    d2 = d1 - volatility * np.sqrt(maturities)
    return strikes * np.exp(-rates * maturities) * norm.cdf(-d2) - S0 * np.exp(-dividends * maturities) * norm.cdf(-d1)


def bs_put_formula_vectorized(S0: float, strikes: np.ndarray, maturities: np.ndarray, rates: np.ndarray,
                              dividends: np.ndarray, volatilities: np.ndarray) -> np.ndarray:
    """Calculate the Black-Scholes put option pricing formula vectorized for multiple strikes and maturities."""
    volatilities = volatilities[:, np.newaxis]
    d1 = (np.log(S0 / strikes) + (rates - dividends + 0.5 * volatilities ** 2) * maturities) / (
            volatilities * np.sqrt(maturities))
    d2 = d1 - volatilities * np.sqrt(maturities)
    put_prices = (strikes * np.exp(-rates * maturities) * norm.cdf(-d2) - S0 * np.exp(-dividends * maturities)
                  * norm.cdf(-d1))
    return put_prices


def bs_implied_vol_vectorized(option_prices: np.ndarray, S0: float, strikes: np.ndarray, maturities: np.ndarray,
                              rates: np.ndarray, dividend_yield: np.ndarray, option_type: str, vol_range=(0.01, 0.9),
                              num_vol_steps=2500) -> np.ndarray:
    """Calculate implied volatilities using grid search."""
    vol_to_test = np.linspace(vol_range[0], vol_range[1], num_vol_steps)
    if option_type == 'C':
        black_scholes_option_prices = bs_call_formula_vectorized_IV(
            S0=S0,
            strikes=strikes,
            maturities=maturities,
            rates=rates,
            dividends=dividend_yield,
            volatilities=vol_to_test
        )
        abs_error = np.abs(black_scholes_option_prices - option_prices)
        implied_vols = vol_to_test[np.argmin(abs_error, axis=0)]

    else:
        black_scholes_option_prices = bs_put_formula_vectorized(
            S0=S0,
            strikes=strikes,
            maturities=maturities,
            rates=rates,
            dividends=dividend_yield,
            volatilities=vol_to_test
        )
        abs_error = np.abs(black_scholes_option_prices - option_prices)
        implied_vols = vol_to_test[np.argmin(abs_error, axis=0)]

    # if the bounds are achieved set np.nan
    implied_vols[implied_vols == vol_range[0]] = np.nan
    implied_vols[implied_vols == vol_range[1]] = np.nan

    return implied_vols