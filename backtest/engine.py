"""
backtest/engine.py — Historical backtesting engine.

Strategy:
  BUY  signal  → enter long position at next close
  SELL signal  → exit long position at next close
  NEUTRAL      → hold current state
  No short selling, no leverage.

Transaction cost: 0.15% each way (brokerage + STT + exchange charges)
Risk-free rate:   7.0% annual (India FD rate approximation)
Initial capital:  ₹1,00,000
"""

import numpy as np
import pandas as pd
import yfinance as yf

TRANSACTION_COST = 0.0015   # 0.15% per side
RISK_FREE_RATE   = 0.07     # 7% annual
INITIAL_CAPITAL  = 100_000  # ₹1,00,000

# ── Indicator helpers (mirror train.py exactly) ──────────────────────────────────
def _rsi(s, p=14):
    d = s.diff(); g = d.where(d>0,0).rolling(p).mean(); l = (-d.where(d<0,0)).rolling(p).mean()
    return 100 - (100/(1+g/l))

def _macd(s, fast=12, slow=26, sig=9):
    ml = s.ewm(span=fast,adjust=False).mean()-s.ewm(span=slow,adjust=False).mean()
    return ml, ml.ewm(span=sig,adjust=False).mean()

def _bb_width(s, p=20):
    sma=s.rolling(p).mean(); std=s.rolling(p).std()
    return (sma+2*std-(sma-2*std))/sma

def _atr(high, low, close, p=14):
    tr=pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(p).mean()

def _earnings_mask(index):
    m,d=index.month,index.day
    return (((m==4)&(d>=15))|(m==5)|((m==7)&(d>=15))|(m==8)|
            ((m==10)&(d>=15))|(m==11)|((m==1)&(d>=15))|(m==2)).astype(int)

def _build_features(df, nifty_close, nifty_ma200, nifty_return, ticker_code, sector_code):
    """Vectorised feature matrix matching train.py exactly."""
    close=df["Close"].squeeze(); high=df["High"].squeeze()
    low=df["Low"].squeeze();     vol=df["Volume"].squeeze()

    f=pd.DataFrame(index=df.index)
    f["RSI"]           = _rsi(close)
    f["MA50"]          = close.rolling(50).mean()
    f["MA200"]         = close.rolling(200).mean()
    f["MA_Cross"]      = f["MA50"]-f["MA200"]
    f["Volatility"]    = close.pct_change().rolling(10).std()
    ml,sl              = _macd(close)
    f["MACD"]          = ml
    f["MACD_Signal"]   = sl
    f["MACD_Hist"]     = ml-sl
    f["BB_Width"]      = _bb_width(close)
    f["Volume_Log"]    = np.log1p(vol)
    f["Volume_Spike"]  = vol/vol.rolling(20).mean()
    f["ATR"]           = _atr(high,low,close)
    f["High52W_Pct"]   = close/close.rolling(252).max()
    f["Low52W_Pct"]    = close/close.rolling(252).min()
    f["Market_Return"] = nifty_return.reindex(df.index).fillna(0)
    f["Market_Regime"] = (nifty_close.reindex(df.index)>nifty_ma200.reindex(df.index)).astype(int).fillna(0)
    f["Earnings_Season"]= _earnings_mask(df.index)
    f["Ticker"]        = ticker_code
    f["Sector"]        = sector_code
    f.dropna(inplace=True)
    return f

# ── Main backtest ────────────────────────────────────────────────────────────────
def run_backtest(ticker, model, label_encoder, metadata, sector_map, period="3y"):
    """
    Run a full vectorised historical backtest for one ticker.

    Returns a dict with:
      metrics      → CAGR, Sharpe, Max Drawdown, Win Rate etc.
      equity_curve → {dates, portfolio (normalised), benchmark (normalised)}
      trades       → list of recent buy/sell events
    """
    PERIOD_DAYS = {"1y":252,"2y":504,"3y":756,"5y":1260}
    n_days_req  = PERIOD_DAYS.get(period, 756)

    # Download — need extra history for MA200 warmup
    raw   = yf.download(ticker, period="10y", interval="1d", auto_adjust=True, progress=False)
    nifty = yf.download("^NSEI", period="10y", interval="1d", auto_adjust=True, progress=False)
    if raw.empty or len(raw) < 260:
        raise ValueError(f"Insufficient historical data for {ticker}")

    nc  = nifty["Close"].squeeze()
    nm  = nc.rolling(200).mean()
    nr  = nc.pct_change()

    ticker_code = 0
    if label_encoder is not None and ticker in label_encoder.classes_:
        ticker_code = int(label_encoder.transform([ticker])[0])
    sector_code = sector_map.get(ticker, 7)

    feat_df = _build_features(raw, nc, nm, nr, ticker_code, sector_code)

    # Select correct feature list from metadata
    g_feats = metadata.get("global_features")
    s_feats = metadata.get("stock_features")
    try:
        probs = model.predict_proba(feat_df[g_feats])[:,1]
    except Exception:
        probs = model.predict_proba(feat_df[s_feats])[:,1]

    buy_thr  = metadata.get("buy_threshold",  0.65)
    sell_thr = metadata.get("sell_threshold", 0.35)
    signals  = np.where(probs>buy_thr,"BUY",np.where(probs<sell_thr,"SELL","NEUTRAL"))

    # Trim to requested period
    prices  = raw["Close"].squeeze().reindex(feat_df.index)
    if len(prices) > n_days_req:
        prices  = prices.iloc[-n_days_req:]
        signals = signals[-n_days_req:]

    # ── Simulate trading ──────────────────────────────────────────────────────────
    capital     = float(INITIAL_CAPITAL)
    shares      = 0.0
    in_pos      = False
    entry_price = 0.0
    trades      = []
    equity      = []   # portfolio value each day

    for i,(date,price) in enumerate(prices.items()):
        p=float(price)
        if np.isnan(p):
            equity.append({"date":str(date.date()),"value":round(capital if not in_pos else shares*p,2)})
            continue
        sig=signals[i]

        if sig=="BUY" and not in_pos:
            cost    = capital*TRANSACTION_COST
            shares  = (capital-cost)/p
            entry_price=p; in_pos=True; capital=0
            trades.append({"date":str(date.date()),"type":"BUY","price":round(p,2)})

        elif sig=="SELL" and in_pos:
            gross   = shares*p
            cost    = gross*TRANSACTION_COST
            capital = gross-cost
            ret_pct = (capital/(shares*entry_price)-1)*100
            shares=0; in_pos=False
            trades.append({"date":str(date.date()),"type":"SELL",
                           "price":round(p,2),"return_pct":round(ret_pct,2)})

        pv = shares*p if in_pos else capital
        equity.append({"date":str(date.date()),"value":round(pv,2)})

    # Close open position at end
    if in_pos:
        last_p  = float(prices.iloc[-1])
        gross   = shares*last_p
        capital = gross-gross*TRANSACTION_COST
        ret_pct = (capital/(shares*entry_price)-1)*100
        trades.append({"date":str(prices.index[-1].date()),"type":"SELL (auto-close)",
                       "price":round(last_p,2),"return_pct":round(ret_pct,2)})
        if equity: equity[-1]["value"] = round(capital,2)

    final_val = equity[-1]["value"] if equity else INITIAL_CAPITAL

    # ── Metrics ───────────────────────────────────────────────────────────────────
    vals   = pd.Series([e["value"] for e in equity])
    drets  = vals.pct_change().dropna()
    n      = len(vals)

    total_ret = (final_val/INITIAL_CAPITAL-1)*100
    cagr      = ((final_val/INITIAL_CAPITAL)**(252/n)-1)*100 if n>0 else 0
    rf_d      = RISK_FREE_RATE/252
    exc       = drets-rf_d
    sharpe    = float(exc.mean()/exc.std()*np.sqrt(252)) if exc.std()>0 else 0

    roll_max  = vals.cummax()
    drawdown  = (vals-roll_max)/roll_max*100
    max_dd    = float(drawdown.min())

    sells     = [t for t in trades if "SELL" in t["type"] and "return_pct" in t]
    wins      = [t for t in sells if t["return_pct"]>0]
    losses    = [t for t in sells if t["return_pct"]<=0]
    win_rate  = len(wins)/len(sells)*100 if sells else 0
    gp        = sum(t["return_pct"] for t in wins)
    gl        = abs(sum(t["return_pct"] for t in losses))
    pf        = gp/gl if gl>0 else (float("inf") if gp>0 else 0)
    avg_ret   = sum(t["return_pct"] for t in sells)/len(sells) if sells else 0

    # Nifty benchmark
    nifty_p   = nc.reindex(prices.index).dropna()
    bench_ret = float((nifty_p.iloc[-1]/nifty_p.iloc[0]-1)*100) if len(nifty_p)>1 else 0
    nifty_norm= [round(float(v)/float(nifty_p.iloc[0])*100,2) for v in nifty_p]
    port_norm = [round(e["value"]/INITIAL_CAPITAL*100,2) for e in equity]
    dates_out = [e["date"] for e in equity]

    # Drawdown curve (for chart)
    dd_curve  = [round(float(d),2) for d in drawdown]

    return {
        "metrics": {
            "total_return":     round(total_ret,2),
            "cagr":             round(cagr,2),
            "sharpe_ratio":     round(sharpe,2),
            "max_drawdown":     round(max_dd,2),
            "win_rate":         round(win_rate,2),
            "total_trades":     len(sells),
            "profit_factor":    round(pf,2) if pf!=float("inf") else 99.0,
            "avg_trade_return": round(avg_ret,2),
            "benchmark_return": round(bench_ret,2),
            "alpha":            round(total_ret-bench_ret,2),
            "initial_capital":  INITIAL_CAPITAL,
            "final_capital":    round(final_val,2),
        },
        "equity_curve": {
            "dates":      dates_out,
            "portfolio":  port_norm,
            "benchmark":  nifty_norm[:len(port_norm)],
            "drawdown":   dd_curve,
        },
        "trades":  trades[-30:],
        "period":  period,
        "ticker":  ticker,
        "transaction_cost_pct": TRANSACTION_COST*100,
    }
