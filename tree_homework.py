# %% (0) Install ragtop if needed (run once, then restart the kernel)
# %pip install ragtop

# %% (1) Verify packages -- should show a price of about 113.34
import ragtop.blackscholes
ragtop.blackscholes.black_scholes(-1, 625, 612, 0.05, 0.75, 0.62, borrow_cost=0.011)

# %% (2) Imports and helpers
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from ragtop.blackscholes import black_scholes as _ragtop_bs
    HAVE_RAGTOP = True
except ModuleNotFoundError:
    HAVE_RAGTOP = False
    print("ragtop not installed - using the closed-form Black-Scholes formula instead")


def bs_formula(callput, spot, strike, rate, time, vol, borrow_cost=0.0):
    """Closed-form Black-Scholes. callput: +1 call, -1 put. borrow_cost acts as a dividend yield."""
    Phi = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    d1 = (math.log(spot / strike) + (rate - borrow_cost + 0.5 * vol ** 2) * time) / (vol * math.sqrt(time))
    d2 = d1 - vol * math.sqrt(time)
    return callput * (spot * math.exp(-borrow_cost * time) * Phi(callput * d1)
                      - strike * math.exp(-rate * time) * Phi(callput * d2))


def ragtop_price(res):
    """ragtop returns a dict; pull the price out of it."""
    if isinstance(res, dict):
        for key in res:
            if str(key).lower() in ("price", "value", "px", "prices"):
                return float(np.asarray(res[key]).squeeze())
        for key in res:
            if "price" in str(key).lower():
                return float(np.asarray(res[key]).squeeze())
        raise KeyError(f"No price-like key found. Keys are: {list(res.keys())}")
    return float(np.asarray(res).squeeze())


def bs_price_fn(callput, spot, strike, rate, time, vol, borrow_cost=0.0):
    """Black-Scholes price from ragtop if installed, otherwise from the closed form."""
    if HAVE_RAGTOP:
        return ragtop_price(_ragtop_bs(callput, spot, strike, rate, time, vol, borrow_cost=borrow_cost))
    return bs_formula(callput, spot, strike, rate, time, vol, borrow_cost)


# %% (3) CRR binomial tree. Same signature as ragtop.black_scholes, plus `steps` and `american`.
def crr_tree(callput, spot, strike, rate, time, vol, steps, american=False, borrow_cost=0.0):
    n = steps
    dt = time / n
    up = math.exp(vol * math.sqrt(dt))
    down = 1.0 / up
    growth = math.exp((rate - borrow_cost) * dt)     # risk-neutral growth of the stock per step
    p = (growth - down) / (up - down)                # risk-neutral up probability
    disc = math.exp(-rate * dt)

    j = np.arange(n + 1)
    S = spot * np.exp(vol * math.sqrt(dt) * (2 * j - n))     # terminal stock prices
    V = np.maximum(callput * (S - strike), 0.0)

    for i in range(n - 1, -1, -1):                           # roll back through the tree
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        if american:
            j = np.arange(i + 1)
            S = spot * np.exp(vol * math.sqrt(dt) * (2 * j - i))
            V = np.maximum(V, callput * (S - strike))        # early exercise
    return float(V[0])


# %% Check: American put >= European put on the tree
spot, strike, rate, time, vol, q = 625, 612, 0.05, 0.75, 0.62, 0.011
for n in (1, 5, 10, 50, 100, 500):
    eu = crr_tree(-1, spot, strike, rate, time, vol, n, american=False, borrow_cost=q)
    am = crr_tree(-1, spot, strike, rate, time, vol, n, american=True, borrow_cost=q)
    print(f"N={n:4d}  European={eu:9.5f}  American={am:9.5f}  American>=European: {am >= eu}")
print("Black-Scholes European put:", bs_price_fn(-1, spot, strike, rate, time, vol, borrow_cost=q))

# %% Build the data set: European put, near the money, N = 1..500
bs_price = bs_price_fn(-1, spot, strike, rate, time, vol, borrow_cost=q)
rows = []
for n in range(1, 501):
    tree = crr_tree(-1, spot, strike, rate, time, vol, n, american=False, borrow_cost=q)
    rows.append((n, tree))
df = pd.DataFrame(rows, columns=["N", "Tree"])
df["BS"] = bs_price
df["y"] = df["Tree"] - df["BS"]          # y_N = Tree_N - BS
df["e"] = df["y"].abs()                  # e_N = |y_N|
print(df.isna().sum())                   # every count should be 0
df.head()

# %% Plot 1: N vs y_N
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(df["N"], df["y"], lw=0.8)
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("N (tree steps)")
ax.set_ylabel(r"$y_N$ = Tree$_N$ - BS")
ax.set_title("European put: tree price minus Black-Scholes")
ax.grid(alpha=0.3)
plt.show()

# %% Plot 2: log-log of N vs e_N
fig, ax = plt.subplots(figsize=(9, 5))
ax.loglog(df["N"], df["e"], lw=0.8)
ax.set_xlabel("N (tree steps)")
ax.set_ylabel(r"$e_N$ = |Tree$_N$ - BS|")
ax.set_title("European put: absolute error vs N (log-log)")
ax.grid(alpha=0.3, which="both")
plt.show()

# %% (4) Fit: log e_N against log N and report the slope
df["logN"] = np.log(df["N"])
df["loge"] = np.log(df["e"])
slope, intercept = np.polyfit(df["logN"], df["loge"], 1)
print(f"slope = {slope:.4f}, intercept = {intercept:.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.loglog(df["N"], df["e"], lw=0.8, label=r"$e_N$")
ax.loglog(df["N"], np.exp(intercept) * df["N"] ** slope, "r--", label=f"fit, slope = {slope:.3f}")
ax.set_xlabel("N (tree steps)")
ax.set_ylabel(r"$e_N$")
ax.legend()
ax.grid(alpha=0.3, which="both")
plt.show()
