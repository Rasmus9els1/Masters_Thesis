import numpy as np
import tqdm
import matplotlib.pyplot as plt

from src.schemes.hybrid_scheme.hybrid_flow_vectorized_kappa_one import HybridSchemeVecKappaOne
from src.parameters import ModelParameters, SchemeParameters
from utils.black_scholes import bs_implied_vol_vectorized

plt.style.use('ggplot')
np.set_printoptions(precision=5, suppress=True)

if __name__ == "__main__":
    # Setup
    mp = ModelParameters(
        start_price=100.0,
        start_variance=round(0.235**2, 3),
        alpha=-0.43,
        eta=1.9,
        rho=-0.9
    )

    n_runs = 10

    sp = SchemeParameters(n_paths=1000, n_time_steps=500, terminal_time=1.0)
    hybrid = HybridSchemeVecKappaOne(sp, mp)

    strikes = np.exp(np.linspace(-0.4, 0.4, 50)) * mp.start_price
    log_m   = np.log(strikes / mp.start_price)

    methods = ['base', 'parity', 'mixed', 'control', 'conditional', 'antithetic', 'mixed antithetic']
    payoffs = {m: np.zeros((len(strikes), sp.n_paths * n_runs)) for m in methods}

    # Monte Carlo sampling
    for i in tqdm.tqdm(range(n_runs), desc="Hybrid MC"):
        hybrid.change_random_variables()
        hybrid.change_processes()
        off = i * sp.n_paths
        payoffs['base'][:,       off:off+sp.n_paths] = hybrid.compute_option_payoff(strikes)
        payoffs['mixed'][:,      off:off+sp.n_paths] = hybrid.compute_mixed_payoff(strikes)
        payoffs['control'][:,    off:off+sp.n_paths] = hybrid.compute_control_payoff(strikes)
        payoffs['conditional'][:,off:off+sp.n_paths] = hybrid.compute_cond_payoff(strikes)
        payoffs['antithetic'][:, off:off+sp.n_paths] = hybrid.compute_antithetic_payoff(strikes)
        payoffs['mixed antithetic'][:, off:off+sp.n_paths] = hybrid.compute_antithetic_mixed_payoff(strikes)

        hybrid.change_random_variables()
        hybrid.change_processes()
        payoffs['parity'][:,     off:off+sp.n_paths] = hybrid.compute_option_payoff_with_parity(strikes)

    # Aggregate
    var   = {m: np.var(payoffs[m], axis=1) for m in methods}
    price = {m: np.mean(payoffs[m], axis=1) for m in methods}
    iv    = {
        m: bs_implied_vol_vectorized(
            option_prices=price[m],
            S0=mp.start_price,
            strikes=strikes,
            maturities=np.ones_like(strikes)*sp.terminal_time,
            rates=np.zeros_like(strikes),
            dividend_yield=np.zeros_like(strikes),
            option_type='C'
        ) for m in methods
    }

    # Create side-by-side axes
    fig, (ax_var, ax_iv) = plt.subplots(1, 2, figsize=(10,4), sharex=True)

    # Plot
    for m in methods:
        linestyle = '-' if 'antithetic' not in m else '--'
        ax_var.plot(log_m, var[m], linestyle=linestyle, label=m)
        ax_iv .plot(log_m, iv[m],  linestyle=linestyle, label=m)


    # Labels
    ax_var.set_ylabel('Variance of payoff', fontsize=11)
    ax_iv .set_ylabel(r'$\sigma_{bs}$',       fontsize=11)
    ax_var.set_xlabel(r'$\log(K/S_0)$',        fontsize=9)
    ax_iv .set_xlabel(r'$\log(K/S_0)$',        fontsize=9)

    #set iv y axis to start at 0
    ax_iv.set_ylim(bottom=0)

    # Overall title
    fig.suptitle('Hybrid Scheme', fontsize=12)

    # Shared legend inside the figure, below axes
    handles, labels = ax_var.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.0),
        bbox_transform=fig.transFigure,
        ncol=len(methods),
        frameon=True,
        fontsize=9
    )

    # Make room for legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.2)
    plt.show()
