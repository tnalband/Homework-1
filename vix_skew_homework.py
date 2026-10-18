# %% [markdown]
# # VIX option skews: UXZ5 (17 Dec 2025) and UXF6 (21 Jan 2026)
# Needs only numpy, pandas and matplotlib.

# %% Imports and settings
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CHAIN_FILE = "VIX_65d_Options_ChainData.csv"
UNDER_FILE = "VIX_65d_Options_UnderlyingData.csv"
N_BISECT = 100          # bisection steps for mid-price vol (plenty for 1e-12 accuracy)
VOL_LO, VOL_HI = 1e-4, 10.0

# %% Load data
chain = pd.read_csv(CHAIN_FILE)
under = pd.read_csv(UNDER_FILE)
chain.columns = chain.columns.str.strip()
under.columns = under.columns.str.strip()
print(chain.columns.tolist())
print(under.columns.tolist())
# Expected chain columns: Strike, T, Call Bid, Call Ask, Call IVM, Call Volm, CofC,
#                         Put Bid, Put Ask, Put IVM, Put Volm
# Expected underlying columns: Future, r, q, T

# %% Black-Scholes helpers (S is the spot-equivalent: S = F * exp(-(r-q)T))
_ncdf = np.vectorize(lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))))


def N(x):
    return _ncdf(x)


def d1(S, K, r, q, T, sig):
    return (np.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))


def d2(S, K, r, q, T, sig):
    return d1(S, K, r, q, T, sig) - sig * np.sqrt(T)


def bs_price(S, K, r, q, T, sig, xi):
    """xi = +1 call, -1 put."""
    a = d1(S, K, r, q, T, sig)
    b = a - sig * np.sqrt(T)
    return xi * (S * np.exp(-q * T) * N(xi * a) - K * np.exp(-r * T) * N(xi * b))


def bs_delta(S, K, r, q, T, sig, xi):
    """Delta = xi * exp(-qT) * N(xi * d1)."""
    return xi * np.exp(-q * T) * N(xi * d1(S, K, r, q, T, sig))


def implied_vol(price, S, K, r, q, T, xi, lo=VOL_LO, hi=VOL_HI, n=N_BISECT):
    """Bisection root finder. Returns NaN if the price cannot be matched in [lo, hi]."""
    if not np.isfinite(price) or price <= 0:
        return np.nan
    f_lo = bs_price(S, K, r, q, T, lo, xi) - price
    f_hi = bs_price(S, K, r, q, T, hi, xi) - price
    if f_lo > 0 or f_hi < 0:     # price below no-arbitrage floor, or above the bracket
        return np.nan
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, r, q, T, mid, xi) < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# %% Build one dataframe per series
def build_series(chain, under_row):
    T = float(under_row["T"])
    r, q = float(under_row["r"]), float(under_row["q"])
    df = chain[np.isclose(chain["T"], T, atol=1e-4)].copy().reset_index(drop=True)
    df = df.sort_values("Strike").reset_index(drop=True)

    # Valid quotes only: positive bid and ask, ask >= bid
    c_ok = (df["Call Bid"] > 0) & (df["Call Ask"] > 0) & (df["Call Ask"] >= df["Call Bid"])
    p_ok = (df["Put Bid"] > 0) & (df["Put Ask"] > 0) & (df["Put Ask"] >= df["Put Bid"])
    df["call_mid"] = np.where(c_ok, 0.5 * (df["Call Bid"] + df["Call Ask"]), np.nan)
    df["put_mid"] = np.where(p_ok, 0.5 * (df["Put Bid"] + df["Put Ask"]), np.nan)

    # Forward from put-call parity: C - P = exp(-rT) (F - K)  ->  F = K + exp(rT) (C - P)
    # Use the strike where |C - P| is smallest.
    diff = (df["call_mid"] - df["put_mid"]).abs()
    k0 = diff.idxmin()
    K0 = df.loc[k0, "Strike"]
    F = K0 + np.exp(r * T) * (df.loc[k0, "call_mid"] - df.loc[k0, "put_mid"])
    S = F * np.exp(-(r - q) * T)

    # ATMF strike = strike closest to F; sigma_ATM = average of the two Bloomberg vols there
    iatm = (df["Strike"] - F).abs().idxmin()
    K_atm = df.loc[iatm, "Strike"]
    sig_atm = 0.5 * (df.loc[iatm, "Call IVM"] + df.loc[iatm, "Put IVM"]) / 100.0

    # Bloomberg vols (percent -> decimal)
    df["call_ivm"] = df["Call IVM"] / 100.0
    df["put_ivm"] = df["Put IVM"] / 100.0

    # Mid-price vols
    df["call_mv"] = [implied_vol(m, S, K, r, q, T, +1) for m, K in zip(df["call_mid"], df["Strike"])]
    df["put_mv"] = [implied_vol(m, S, K, r, q, T, -1) for m, K in zip(df["put_mid"], df["Strike"])]

    # Deltas: Delta_ATM uses sigma_ATM; Delta_Loc uses the option's own mid vol
    K = df["Strike"].values
    for tag, xi in (("call", +1), ("put", -1)):
        df[f"{tag}_dATM"] = bs_delta(S, K, r, q, T, sig_atm, xi)
        df[f"{tag}_dLoc"] = bs_delta(S, K, r, q, T, df[f"{tag}_mv"].values, xi)

    df["moneyness"] = np.log(df["Strike"] / F)
    df["z"] = df["moneyness"] / (sig_atm * np.sqrt(T))

    info = dict(T=T, r=r, q=q, F=F, S=S, K_atm=K_atm, sig_atm=sig_atm, K_parity=K0)
    return df, info


series = {}
for i, name in enumerate(["UXZ5 (17 Dec 2025)", "UXF6 (21 Jan 2026)"]):
    series[name] = build_series(chain, under.iloc[i])

for name, (df, info) in series.items():
    print(name)
    print({k: round(v, 6) for k, v in info.items()})   # F for series 2 should be near 21.05
    print("rows:", len(df), "| call mid vols:", df["call_mv"].notna().sum(),
          "| put mid vols:", df["put_mv"].notna().sum())

# %% Plot helpers
COLORS = {
    "A: Bloomberg Call IVM": "tab:blue",
    "B: Bloomberg Put IVM": "tab:orange",
    "C: Call mid vol": "tab:green",
    "D: Put mid vol": "tab:red",
}


def skew_xy(df, kind):
    """Return {label: (x, y)} for the four skews for one plot type."""
    out = {}
    for tag, xi, bb_label, mid_label, bb_col, mv_col in (
        ("call", +1, "A: Bloomberg Call IVM", "C: Call mid vol", "call_ivm", "call_mv"),
        ("put", -1, "B: Bloomberg Put IVM", "D: Put mid vol", "put_ivm", "put_mv"),
    ):
        if kind == "strike":
            x = df["Strike"]
            xs = {bb_label: x, mid_label: x}
        elif kind == "moneyness":
            x = df["moneyness"]
            xs = {bb_label: x, mid_label: x}
        elif kind == "z":
            x = df["z"]
            xs = {bb_label: x, mid_label: x}
        elif kind == "center_delta":
            d = df[f"{tag}_dATM"]
            keep = (xi * d) < 0.5
            x = (1 + xi) / 2 - d
            xs = {bb_label: x.where(keep), mid_label: x.where(keep)}
        elif kind == "delta":
            d = df[f"{tag}_dLoc"]
            keep = (xi * d) < 0.5
            x = (1 + xi) / 2 - d
            xs = {bb_label: x.where(keep), mid_label: x.where(keep)}
        out[bb_label] = (xs[bb_label], df[bb_col])
        out[mid_label] = (xs[mid_label], df[mv_col])
    return out


PLOT_TYPES = {
    "strike": ("Strike K", "Strike"),
    "moneyness": ("Moneyness log(K/F)", "Moneyness"),
    "z": (r"Z-score  log(K/F) / ($\sigma_{ATM}\sqrt{T}$)", "Z-score"),
    "center_delta": (r"Center delta  $(1+\xi)/2 - \Delta_{ATM}$", "Center Delta"),
    "delta": (r"Delta  $(1+\xi)/2 - \Delta_{Loc}$", "Delta"),
}


def make_plot(name, df, info, kind):
    xlabel, title_kind = PLOT_TYPES[kind]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, (x, y) in skew_xy(df, kind).items():
        m = x.notna() & y.notna()
        ax.scatter(x[m], y[m], s=22, color=COLORS[label], label=label, alpha=0.85)
    if kind in ("center_delta", "delta"):
        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0, 1.01, 0.1))
        ax.axvline(0.25, color="k", ls="--", lw=1)
        ax.axvline(0.75, color="k", ls="--", lw=1)
        ax.axvline(0.5, color="grey", ls=":", lw=0.8)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="out", top=True, right=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Implied volatility (0.5 = 50%)")
    ax.set_title(f"{name}: implied volatility vs {title_kind}\n"
                 f"F = {info['F']:.3f}, ATMF strike = {info['K_atm']}, "
                 f"$\\sigma_{{ATM}}$ = {info['sig_atm']:.3f}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


# %% Plots (10 figures) and a combined PDF
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages("VIX_skew_plots.pdf") as pdf:
    for name, (df, info) in series.items():
        for kind in PLOT_TYPES:
            fig = make_plot(name, df, info, kind)
            pdf.savefig(fig)
            plt.show()
print("Saved VIX_skew_plots.pdf")
