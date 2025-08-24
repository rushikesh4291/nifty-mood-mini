import numpy as np, pandas as pd
from pathlib import Path
from .etl import merge_all
from .alerts import evaluate_rules_trend, load_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"

def metrics(returns):
    ann = 252
    mu, sd = returns.mean(), returns.std(ddof=0)
    sharpe = (mu * ann) / (sd * np.sqrt(ann)) if sd > 0 else 0.0
    eq = (1 + returns).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1).min()
    win_rate = (returns > 0).mean()
    return sharpe, dd, win_rate, eq

def main():
    df = merge_all()
    rules = load_rules()
    hist = evaluate_rules_trend(df, rules, window=len(df))  # full history
    hist = hist.dropna(subset=["Close"]).copy()
    hist["ret"] = hist["Close"].pct_change().fillna(0.0)

    # Simple strategy: long when Mood >= 55, flat otherwise
    hist["pos"] = (hist["mood_score"] >= 55).astype(float)
    strat_ret = hist["pos"] * hist["ret"]

    sharpe, maxdd, winr, eq = metrics(strat_ret)

    OUT.mkdir(exist_ok=True, parents=True)
    pd.DataFrame({
        "metric": ["Sharpe", "MaxDD", "WinRate", "CAGR"],
        "value": [
            sharpe,
            maxdd,
            (winr),
            (eq.iloc[-1] ** (252/len(eq)) - 1) if len(eq) > 0 else 0.0
        ],
    }).to_csv(OUT / "backtest_summary.csv", index=False)

    pd.DataFrame({"date": hist["date"], "equity": eq}).to_csv(OUT / "equity_curve.csv", index=False)
    print("Wrote:", OUT / "backtest_summary.csv", "and", OUT / "equity_curve.csv")

if __name__ == "__main__":
    main()
