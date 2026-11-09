# %% [markdown]
# # Convertible Bond Analysis with ragtop
# Parts 1-4: pricing, option-market calibration, convertible pricing, and price-vs-stock plot.

# %%
import itertools, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ragtop.implicit import find_present_value
from ragtop.instruments import CALL, PUT, AmericanOption, ConvertibleBond
from ragtop.term_structures import spot_to_df_fcn

# %% [markdown]
# ## Market data and instruments

# %%
S0 = 241.8
T_OPT = 1.72

yield_curve = pd.DataFrame({
    "time": [0.04988584, 0.12659817, 0.37591324, 0.62522831, 0.72111872, 1.71837900, 5.0],
    "rate": [0.004208395, 0.004846041, 0.007777790, 0.009878801, 0.010491200, 0.010167270, 0.02],
})
df_fcn = spot_to_df_fcn(yield_curve)

# callput: -1 = put, +1 = call  (ragtop: PUT = -1, CALL = 1)
assert PUT == -1 and CALL == 1
mkt = pd.DataFrame({
    "callput": [-1, -1, 1],
    "K":       [230, 150, 370],
    "mid":     [52.40, 19.45, 13.95],
    "bid":     [51.35, 17.90, 13.00],
    "ask":     [53.45, 21.00, 14.90],
})
mkt["name"] = [("Put" if cp == -1 else "Call") + str(K) for cp, K in zip(mkt.callput, mkt.K)]

# Listed options are treated as American (ragtop's AmericanOption)
options = [AmericanOption(maturity=T_OPT, strike=float(r.K), callput=int(r.callput), name=r.name)
           for r in mkt.itertuples()]

# Convertible: 3.84615 shares per $1,000 bond (conversion price 260), 2 years, no coupons given
conv = ConvertibleBond(maturity=2.0, notional=1000.0, conversion_ratio=3.84615, name="Convertible")
instruments = options + [conv]

# Grid settings
N_STEPS = 250
STD_WIDTH = 4.5

# %% [markdown]
# ## Part 1: Pricing with constant hazard 7.5% and volatility 50%

# %%
def price(S0_=S0, vol=0.5, hazard_fcn=None, const_hazard=0.0, instr=instruments):
    return find_present_value(
        S0=S0_, instruments=instr, num_time_steps=N_STEPS,
        const_volatility=vol, discount_factor_fcn=df_fcn,
        default_intensity_fcn=hazard_fcn, const_default_intensity=const_hazard,
        std_devs_width=STD_WIDTH)

t0 = time.time()
part1 = price(vol=0.5, const_hazard=0.075)
print(f"one run: {time.time()-t0:.3f}s")
part1_df = pd.DataFrame({"model_price": part1})
part1_df["market_mid"] = list(mkt.mid) + [np.nan]
part1_df

# %% [markdown]
# ## Part 2: Calibration grid over (v, b, p)
# Price all four instruments on the same grid for each parameter triple. Error metric: RMSE of option *prices* (3 options).

# %%
vols = np.round(np.arange(0.20, 0.901, 0.05), 2)
ps   = np.arange(0, 9)
bs   = np.round(np.arange(0.01, 0.1001, 0.01), 2)

def make_hazard(b, p):
    return lambda t, S, *args, **kwargs: 0.075 * (b + (1 - b) * (241.80 / S) ** p)

rows = []
t0 = time.time()
for v, p, b in itertools.product(vols, ps, bs):
    res = price(vol=v, hazard_fcn=make_hazard(b, p))
    errs = np.array([res[o.name] - m for o, m in zip(options, mkt.mid)])
    rows.append(dict(v=v, p=p, b=b,
                     **{o.name: res[o.name] for o in options},
                     Convertible=res["Convertible"],
                     mse=np.mean(errs**2), rmse=np.sqrt(np.mean(errs**2))))
grid = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
print(f"{len(grid)} runs in {time.time()-t0:.0f}s")
grid.head(10)

# %%
best = grid.iloc[0]
print("Best parameters:")
print(best[["v", "b", "p", "rmse"]])
pd.DataFrame({"model": best[[o.name for o in options]].values, "mid": mkt.mid.values,
              "bid": mkt.bid.values, "ask": mkt.ask.values}, index=[o.name for o in options])

# %%
# Is each best-fit price inside the bid/ask?
for o, bid, ask in zip(options, mkt.bid, mkt.ask):
    print(o.name, "inside bid/ask:", bid <= best[o.name] <= ask)

# Best RMSE for each vol (profile), and spread of results
print(grid.groupby("v").rmse.min().round(3))
print("RMSE range over grid:", grid.rmse.min(), grid.rmse.max())

# %% [markdown]
# ## Part 3: Convertible bond price at best and non-optimal parameters

# %%
def conv_price(v, b, p):
    return price(vol=v, hazard_fcn=make_hazard(b, p))["Convertible"]

check = conv_price(best.v, best.b, best.p)
print("Best-parameter convertible price:", check)

picks = pd.concat([
    grid.iloc[[0, 1, 2, 3, 4]],                 # next-best fits
    grid.iloc[[len(grid)//4, len(grid)//2, 3*len(grid)//4]],  # mid-ranked
    grid.iloc[[-1]],                            # worst fit
])
conv_table = picks[["v", "b", "p", "rmse", "Convertible"]].copy()
conv_table["diff_vs_best"] = conv_table.Convertible - best.Convertible
conv_table["pct_diff_vs_best"] = 100 * conv_table.diff_vs_best / best.Convertible
conv_table

# %%
# How much does the bond price move among near-optimal fits?
for tol in [1.0, 2.0, 5.0]:
    near = grid[grid.rmse <= best.rmse + tol]
    print(f"RMSE within {tol} of best: {len(near)} sets; bond price range "
          f"[{near.Convertible.min():.2f}, {near.Convertible.max():.2f}]")
print("Bond price over whole grid: min %.2f max %.2f" % (grid.Convertible.min(), grid.Convertible.max()))

# %% [markdown]
# ## Part 4: Convertible bond value vs. stock price (best parameters)
# The hazard function keeps its reference price at 241.80; only the current stock price varies.

# %%
S_values = np.arange(20, 501, 10)
haz = make_hazard(best.b, best.p)
bond_vals = [find_present_value(S0=float(s), instruments=[conv], num_time_steps=N_STEPS,
                                const_volatility=best.v, discount_factor_fcn=df_fcn,
                                default_intensity_fcn=haz, std_devs_width=STD_WIDTH)["Convertible"]
             for s in S_values]
bond_df = pd.DataFrame({"S": S_values, "value": bond_vals})
bond_df["parity"] = 3.84615 * bond_df.S

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(bond_df.S, bond_df.value, lw=2.5, color="tab:orange", label="Convertible value")
ax.plot(bond_df.S, bond_df.parity, "--", color="gray", label="Conversion value (ratio x S)")
ax.axhline(1000, ls=":", color="k", lw=1, label="Par ($1,000)")
ax.set_xlabel("Stock price S")
ax.set_ylabel("Convertible bond price ($)")
ax.set_title(f"Convertible value vs stock price (v={best.v}, b={best.b}, p={int(best.p)})")
ax.legend()
plt.tight_layout()
plt.show()
bond_df.iloc[::5]

# %%
# Optional: compare a few hazard shapes on the same plot (p varies, v and b at best)
fig, ax = plt.subplots(figsize=(8, 5))
for p_ in [0, 2, int(best.p), 8]:
    h = make_hazard(best.b, p_)
    vals = [find_present_value(S0=float(s), instruments=[conv], num_time_steps=N_STEPS,
                               const_volatility=best.v, discount_factor_fcn=df_fcn,
                               default_intensity_fcn=h, std_devs_width=STD_WIDTH)["Convertible"]
            for s in S_values]
    ax.plot(S_values, vals, label=f"p={p_}")
ax.set_xlabel("Stock price S"); ax.set_ylabel("Convertible bond price ($)")
ax.set_title("Sensitivity of convertible value to hazard exponent p")
ax.legend(); plt.tight_layout(); plt.show()
