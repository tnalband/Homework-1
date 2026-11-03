# Binomial tree with 3 initial nodes, tree greeks, BS "exact" greeks and a control variate.
# Run cell by cell (# %% blocks) in Jupyter/VS Code, or just run the whole file.

# %% (1) Imports and Black-Scholes value / delta / gamma
import math
import numpy as np
import pandas as pd

Phi = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))        # normal cdf
phi = lambda x: math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)  # normal pdf


def bs_greeks(xi, S0, K, r, T, sigma, q):
    """Black-Scholes value, delta, gamma. xi = +1 call, -1 put. q = continuous dividend yield."""
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    value = xi * (S0 * math.exp(-q * T) * Phi(xi * d1) - K * math.exp(-r * T) * Phi(xi * d2))
    delta = xi * math.exp(-q * T) * Phi(xi * d1)
    gamma = math.exp(-q * T) * phi(d1) / (S0 * sigma * math.sqrt(T))
    return value, delta, gamma


# %% (2) Tree with 3 initial nodes (S0*u^2, S0, S0*d^2) -> value, delta, gamma
def tree_greeks(xi, S0, K, r, T, sigma, q, N, american):
    """CRR tree started from THREE nodes at time 0 instead of one.
    Level i has i+3 nodes with stock price S0*u^k, k = -(i+2), -(i+2)+2, ..., (i+2).
    Returns (value, delta, gamma) from the middle node and finite differences on the 3 start nodes."""
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)      # risk-neutral up probability
    disc = math.exp(-r * dt)

    def stock(i):                                   # stock prices at level i
        return S0 * u ** np.arange(-(i + 2), i + 3, 2)

    V = np.maximum(xi * (stock(N) - K), 0.0)        # payoff at maturity
    for i in range(N - 1, -1, -1):                  # roll back; ends with 3 nodes at level 0
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        if american:
            V = np.maximum(V, xi * (stock(i) - K))  # early exercise check at every node

    S_dn, S_0, S_up = stock(0)
    V_dn, V_0, V_up = V
    h = 0.5 * ((S_up - S_0) + (S_0 - S_dn))         # eq. (3): average spacing
    delta = (V_up - V_dn) / (2.0 * h)               # eq. (1): central difference
    gamma = (V_up - 2.0 * V_0 + V_dn) / h ** 2      # eq. (2)
    return V_0, delta, gamma


# %% (3) The 12-value function the assignment asks for
def tree_12(xi, S0, K, r, T, sigma, q, N):
    """Returns 12 values:
    BS (value, delta, gamma), European tree (v, d, g), American tree uncorrected (v, d, g),
    American control-variate corrected (v, d, g)."""
    bs = np.array(bs_greeks(xi, S0, K, r, T, sigma, q))
    eu = np.array(tree_greeks(xi, S0, K, r, T, sigma, q, N, american=False))
    am = np.array(tree_greeks(xi, S0, K, r, T, sigma, q, N, american=True))
    cv = am - (eu - bs)          # V_cv = V_A - (V_E,tree - V_E,BS), also for delta and gamma
    return (*bs, *eu, *am, *cv)


NAMES = ["BS_value", "BS_delta", "BS_gamma",
         "EU_value", "EU_delta", "EU_gamma",
         "AM_value", "AM_delta", "AM_gamma",
         "CV_value", "CV_delta", "CV_gamma"]

S0, T, r, q, sigma = 50.0, 0.1, 0.1, 0.02, 0.3
xi = -1                                   # puts
print(pd.Series(tree_12(xi, S0, 46, r, T, sigma, q, 100), index=NAMES).round(6))

# %% (4) Proxy "exact" values: 1000-step tree, CV-corrected
def proxy(K):
    out = tree_12(xi, S0, K, r, T, sigma, q, 1000)
    return np.array(out[9:12])            # X_V, X_Delta, X_Gamma


# %% (5) Big trees: average error of the UNCORRECTED American value / delta / gamma
BIG_STEPS = [100, 105, 110, 120]


def big_tree_errors(K, X):
    errs = []
    for N in BIG_STEPS:
        am = np.array(tree_greeks(xi, S0, K, r, T, sigma, q, N, american=True))
        errs.append(np.abs(am - X))       # absolute error vs proxy
    return np.mean(errs, axis=0)          # E_V^U, E_Delta^U, E_Gamma^U


# %% (6) Smallest CV tree that matches that error, and computational savings
def smallest_cv_steps(K, X, target, max_N=120):
    """For each of value/delta/gamma: first N whose CV-corrected error <= the big-tree average error."""
    found = [None, None, None]
    for N in range(2, max_N + 1):
        cv = np.array(tree_12(xi, S0, K, r, T, sigma, q, N)[9:12])
        err = np.abs(cv - X)
        for g in range(3):
            if found[g] is None and err[g] <= target[g]:
                found[g] = N
    return found


rows = []
ref_steps = np.mean(BIG_STEPS)
for K in (46, 53):
    X = proxy(K)
    E = big_tree_errors(K, X)
    Ns = smallest_cv_steps(K, X, E)
    print(f"\nStrike {K}: proxy X (V, Delta, Gamma) = {X}")
    for name, e, n in zip(["value", "delta", "gamma"], E, Ns):
        # tree cost ~ N^2 nodes; CV needs a European AND an American tree, so cost ~ 2 N^2
        saving_nodes = (ref_steps ** 2) / (2 * n ** 2) if n else float("nan")
        saving_steps = ref_steps / n if n else float("nan")
        rows.append(dict(strike=K, quantity=name, avg_uncorrected_error=e, cv_steps=n,
                         step_ratio=saving_steps, cost_ratio_N2=saving_nodes))
res = pd.DataFrame(rows)
pd.options.display.float_format = "{:.6g}".format
print(res.to_string(index=False))
