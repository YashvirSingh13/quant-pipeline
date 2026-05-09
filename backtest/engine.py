# QP-38544345-675 2026-05-09 03:14:00
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


def _json_safe(obj):
    # Recursively sanitise for JSON — handles numpy types, NaN, inf, -inf
    import numpy as np
    import pandas as pd
    import math
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return [_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return None  # shouldn't appear but safety net
    return obj


def _safe_ret(s, idx, p=1):
    """Return a Series of pct_change(p) for s aligned to idx, zeros if unavailable."""
    if s is None or (hasattr(s,'empty') and s.empty):
        return pd.Series(0.0, index=idx)
    return s.reindex(idx).ffill().bfill().pct_change(p).fillna(0)

def _safe_lvl(s, idx, default=0.0):
    if s is None or (hasattr(s,'empty') and s.empty):
        return pd.Series(default, index=idx)
    return s.reindex(idx).ffill().bfill().fillna(default)


def _build_features(df, nifty_close, nifty_ma200, nifty_return, ticker_code,
                    sector_code, macro=None):
    """
    Vectorised feature matrix matching train.py exactly — all phases.
    macro: dict of {name: pd.Series} for USDINR, Crude, SP500, etc.
    """
    macro = macro or {}
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()
    idx   = df.index

    f = pd.DataFrame(index=idx)

    # ── Core technical (Phase 0) ─────────────────────────────────────────────
    f["RSI"]          = _rsi(close)
    f["MA50"]         = close.rolling(50).mean()
    f["MA200"]        = close.rolling(200).mean()
    f["MA_Cross"]     = f["MA50"] - f["MA200"]
    f["Volatility"]   = close.pct_change().rolling(10).std()
    ml, sl            = _macd(close)
    f["MACD"]         = ml
    f["MACD_Signal"]  = sl
    f["MACD_Hist"]    = ml - sl
    f["BB_Width"]     = _bb_width(close)
    f["Volume_Log"]   = np.log1p(vol)
    f["Volume_Spike"] = vol / vol.rolling(20).mean()
    f["ATR"]          = _atr(high, low, close)
    f["High52W_Pct"]  = close / close.rolling(252).max()
    f["Low52W_Pct"]   = close / close.rolling(252).min()
    nc_aligned        = nifty_close.reindex(idx).ffill().bfill()
    nm_aligned        = nifty_ma200.reindex(idx).ffill().bfill()
    nr_aligned        = nifty_return.reindex(idx).fillna(0)
    f["Market_Return"]  = nr_aligned
    f["Market_Regime"]  = (nc_aligned > nm_aligned).astype(int)
    f["Earnings_Season"]= _earnings_mask(idx)

    # ── Phase 1: Momentum + Beta ─────────────────────────────────────────────
    daily_ret         = close.pct_change()
    f["Return_1d"]    = daily_ret
    f["Return_5d_lag"]= close.pct_change(5)
    f["Return_20d"]   = close.pct_change(20)
    cov60             = daily_ret.rolling(60).cov(nr_aligned)
    var60             = nr_aligned.rolling(60).var()
    f["Beta_60d"]     = (cov60 / (var60 + 1e-9)).clip(-3, 3)
    nifty_20d         = nc_aligned.pct_change(20)
    f["Rel_Strength"] = close.pct_change(20) - nifty_20d

    # ── Phase 4: Market structure ─────────────────────────────────────────────
    ma20              = close.rolling(20).mean()
    ma50              = close.rolling(50).mean()
    std20             = close.rolling(20).std()
    f["Dist_MA20"]    = (close - ma20) / (ma20 + 1e-9)
    f["Dist_MA50"]    = (close - ma50) / (ma50 + 1e-9)
    f["MA20_Slope"]   = ma20.pct_change(5)
    f["MA50_Slope"]   = ma50.pct_change(10)
    bb_lower          = ma20 - 2 * std20
    bb_range          = 4 * std20
    f["BB_Position"]  = ((close - bb_lower) / (bb_range + 1e-9)).clip(0, 1)

    # ── Phase 5B: Macro — Dollar, Crude, Seasonal ───────────────────────────
    usd = macro.get("usdinr")
    crd = macro.get("crude")
    f["USDINR_Return"]   = _safe_ret(usd, idx)
    f["USDINR_20d_Mom"]  = _safe_ret(usd, idx, p=20)
    f["Crude_Return"]    = _safe_ret(crd, idx)
    f["Crude_20d_Mom"]   = _safe_ret(crd, idx, p=20)
    f["Month_Sin"]       = pd.Series(np.sin(2*np.pi*idx.month/12), index=idx)
    f["Month_Cos"]       = pd.Series(np.cos(2*np.pi*idx.month/12), index=idx)
    f["Is_Budget_Month"] = pd.Series((idx.month == 2).astype(int), index=idx)
    f["Is_Monsoon"]      = pd.Series(idx.month.isin([6,7,8,9]).astype(int), index=idx)

    # ── Phase 6: Global macro ────────────────────────────────────────────────
    sp5   = macro.get("sp500")
    nsdq  = macro.get("nasdaq")
    vixus = macro.get("vix_us")
    vixin = macro.get("vix_in")
    us10y = macro.get("us10y")
    copp  = macro.get("copper")
    shan  = macro.get("shanghai")

    sp5_r             = _safe_ret(sp5,  idx)
    f["SP500_Return"]  = sp5_r
    f["SP500_5d"]      = _safe_ret(sp5,  idx, p=5)
    f["FII_Proxy"]     = nr_aligned - sp5_r
    f["VIX_US_Level"]  = _safe_lvl(vixus, idx) / 100

    if vixin is not None and not vixin.empty:
        vi = vixin.reindex(idx).ffill().bfill()
        f["VIX_IN_ROC5"] = vi.pct_change(5).fillna(0)
        f["VIX_IN_Pct"]  = vi.rolling(252, min_periods=30).rank(pct=True).fillna(0.5)
    else:
        f["VIX_IN_ROC5"] = 0.0
        f["VIX_IN_Pct"]  = 0.5

    if us10y is not None and not us10y.empty:
        us = us10y.reindex(idx).ffill().bfill()
        f["US10Y_Level"] = us / 100
        f["US10Y_Chg"]   = us.diff().fillna(0)
    else:
        f["US10Y_Level"] = 0.04
        f["US10Y_Chg"]   = 0.0

    copp_r              = _safe_ret(copp, idx)
    f["Copper_Return"]  = copp_r
    f["Shanghai_Return"]= _safe_ret(shan, idx)
    nsdq_r              = _safe_ret(nsdq, idx)

    # ── Phase 6: Sector-conditional ──────────────────────────────────────────
    is_it      = int(sector_code == 2)
    is_export  = int(sector_code in [2, 7])
    is_energy  = int(sector_code == 4)
    is_metals  = int(sector_code == 8)
    is_infra   = int(sector_code == 6)
    is_banking = int(sector_code in [0, 1])
    is_fmcg    = int(sector_code == 5)

    f["NASDAQ_IT"]       = nsdq_r                    * is_it
    f["USD_Export"]      = f["USDINR_Return"]         * is_export
    f["Crude_Sector"]    = f["Crude_Return"]          * is_energy
    f["Copper_Sector"]   = copp_r                     * is_metals
    f["Shanghai_Sector"] = f["Shanghai_Return"]       * int(is_metals or is_infra)
    f["Yield_Banking"]   = f["US10Y_Chg"]             * is_banking
    f["Monsoon_FMCG"]    = f["Is_Monsoon"]            * is_fmcg

    # ── Identity ──────────────────────────────────────────────────────────────
    f["Ticker"] = ticker_code
    f["Sector"] = sector_code

    # ── Phase 7: NSE proxies (neutral — no historical option chain data) ──────
    # Match training defaults exactly so model sees consistent values
    f["PCR"]          = 1.0   # neutral options sentiment
    f["PCR_Signal"]   = 0.0   # z-score neutral
    f["FII_Net_Norm"] = (f["FII_Proxy"].clip(-1, 1) if "FII_Proxy" in f.columns
                         else 0.0)
    f["DII_Net_Norm"] = 0.0
    f["Breadth_Pct"]  = (f["Market_Regime"] * 40 + 50).clip(10, 90)
    f["AdvDec_Ratio"] = 50.0

    # ── Phase 8: Lag features ────────────────────────────────────────────────
    f["RSI_lag1"]       = f["RSI"].shift(1)
    f["RSI_lag3"]       = f["RSI"].shift(3)
    f["MACD_Hist_lag1"] = f["MACD_Hist"].shift(1)
    f["Return_lag2"]    = close.pct_change().shift(2)
    f["Vol_Spike_lag1"] = f["Volume_Spike"].shift(1)
    f["Max_Pain_Dist"]  = 0.0  # no historical max pain data

    # ── Phase 9: Fundamental proxies (neutral during backtest) ───────────────
    f["EPS_Surprise"]    = 0.0
    f["Promoter_Change"] = 0.0
    f["Sector_Momentum"] = 0.5
    f["Sector_Rel_Perf"] = f["Rel_Strength"].rolling(5).mean().fillna(0)

    f.dropna(subset=["RSI","MA50","MACD"], inplace=True)
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

    # Download — stock + nifty + all macro data in parallel
    from concurrent.futures import ThreadPoolExecutor
    MACRO_SYMS = {
        "nifty":    "^NSEI",
        "usdinr":   "USDINR=X",
        "crude":    "BZ=F",
        "sp500":    "^GSPC",
        "nasdaq":   "^IXIC",
        "vix_us":   "^VIX",
        "vix_in":   "^INDIAVIX",
        "us10y":    "^TNX",
        "copper":   "HG=F",
        "shanghai": "000001.SS",
    }

    def _dl(sym):
        try:
            df = yf.download(sym, period="10y", interval="1d",
                             auto_adjust=True, progress=False)
            return df["Close"].squeeze() if not df.empty else pd.Series(dtype=float)
        except Exception:
            return pd.Series(dtype=float)

    def _dl_stock(sym):
        try:
            return yf.download(sym, period="10y", interval="1d",
                               auto_adjust=True, progress=False)
        except Exception:
            return pd.DataFrame()

    with ThreadPoolExecutor(max_workers=11) as ex:
        raw_fut    = ex.submit(_dl_stock, ticker)
        macro_futs = {k: ex.submit(_dl, v) for k, v in MACRO_SYMS.items()}
        raw        = raw_fut.result(timeout=30)
        macro      = {k: f.result(timeout=30) for k, f in macro_futs.items()}

    if raw.empty or len(raw) < 260:
        raise ValueError(f"Insufficient historical data for {ticker}")

    nc  = macro["nifty"]
    nm  = nc.rolling(200).mean() if not nc.empty else pd.Series(dtype=float)
    nr  = nc.pct_change()

    ticker_code = 0
    if label_encoder is not None and ticker in label_encoder.classes_:
        ticker_code = int(label_encoder.transform([ticker])[0])
    sector_code = sector_map.get(ticker, 7)

    feat_df = _build_features(raw, nc, nm, nr, ticker_code, sector_code, macro=macro)

    # ── Feature alignment — definitive fix ──────────────────────────────────
    # Strip column name spaces (legacy pandas MultiIndex quirk)
    feat_df.columns = [str(col).strip() for col in feat_df.columns]

    # Get booster's expected feature list
    # Handles both raw XGBoost and CalibratedClassifierCV wrapper
    booster_feats = None
    try:
        # Direct XGBoost
        booster_feats = [f.strip() for f in model.get_booster().feature_names]
    except Exception:
        pass
    if not booster_feats:
        try:
            # Our CalibratedModel wrapper — delegates get_booster() to base
            booster_feats = [f.strip() for f in model.get_booster().feature_names]
        except Exception:
            pass
    if not booster_feats:
        # Final fallback: use metadata feature list
        g_feats = [f.strip() for f in (metadata.get("global_features") or [])]
        s_feats = [f.strip() for f in (metadata.get("stock_features")  or [])]
        booster_feats = g_feats or s_feats or list(feat_df.columns)

    if not booster_feats:
        raise ValueError("Could not determine model feature names")

    # Build numpy array in EXACT booster order.
    # Missing features → fill with 0 (neutral).
    # This guarantees shape always matches — no count mismatch possible.
    n_rows   = len(feat_df)
    n_feats  = len(booster_feats)
    feat_arr = np.zeros((n_rows, n_feats), dtype=np.float32)
    missing  = []
    for i, fname in enumerate(booster_feats):
        if fname in feat_df.columns:
            feat_arr[:, i] = feat_df[fname].values.astype(np.float32)
        else:
            missing.append(fname)

    if missing:
        print(f"⚠  Backtest: {len(missing)} features filled with 0: {missing[:5]}...")

    # Use numpy array for raw XGBoost; DataFrame for calibrated wrapper
    try:
        probs = model.predict_proba(feat_arr)[:,1]
    except Exception:
        # Calibrated model may need DataFrame with feature names
        feat_df_ordered = pd.DataFrame(feat_arr, columns=booster_feats)
        probs = model.predict_proba(feat_df_ordered)[:,1]

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

    # ── Phase 10: Trailing stop state ────────────────────────────────────────
    trail_stop      = 0.0    # current trailing stop price
    trail_target1   = 0.0    # first target (move stop to breakeven)
    trail_target2   = 0.0    # second target (lock in profit)
    partial_exited  = False  # did we take partial profit at T1?

    # Pre-compute ATR aligned to prices index
    price_series    = raw["Close"].squeeze().reindex(prices.index).ffill()
    high_series     = raw["High"].squeeze().reindex(prices.index).ffill()
    low_series      = raw["Low"].squeeze().reindex(prices.index).ffill()
    atr_series      = _atr(high_series, low_series, price_series)

    for i,(date,price) in enumerate(prices.items()):
        p=float(price)
        if np.isnan(p):
            equity.append({"date":str(date.date()),"value":round(capital if not in_pos else shares*p,2),"in_pos":in_pos})
            continue
        sig=signals[i]
        atr_val = float(atr_series.get(date, atr_series.iloc[-1] if len(atr_series) else 0) or 0)

        # ── Entry ────────────────────────────────────────────────────────────
        if sig=="BUY" and not in_pos:
            cost         = capital*TRANSACTION_COST
            shares       = (capital-cost)/p
            entry_price  = p; in_pos=True; capital=0
            partial_exited = False
            # Set initial trailing stop and targets
            trail_stop   = p - 1.5 * atr_val if atr_val > 0 else p * 0.97
            trail_target1= p + 1.5 * atr_val if atr_val > 0 else p * 1.03
            trail_target2= p + 3.0 * atr_val if atr_val > 0 else p * 1.06
            trades.append({"date":str(date.date()),"type":"BUY","price":round(p,2),
                           "stop":round(trail_stop,2),"t1":round(trail_target1,2)})

        elif in_pos:
            # ── Trail stop management (Phase 10) ─────────────────────────────
            if atr_val > 0:
                # After T1 hit: move stop to breakeven
                if p >= trail_target1 and not partial_exited:
                    trail_stop     = max(trail_stop, entry_price)
                    partial_exited = True
                # After T2 hit: lock in half ATR profit
                if p >= trail_target2:
                    trail_stop = max(trail_stop, entry_price + 0.5*atr_val)
                # Normal trail: move stop up if price moves up (ratchet)
                new_trail = p - 1.5*atr_val
                trail_stop = max(trail_stop, new_trail)

            # ── Exit conditions ───────────────────────────────────────────────
            exit_triggered = False
            exit_reason    = ""

            # 1. Trailing stop hit
            if p <= trail_stop and trail_stop > 0:
                exit_triggered = True; exit_reason = "Trail Stop"
            # 2. Model SELL signal
            elif sig == "SELL":
                exit_triggered = True; exit_reason = "SELL Signal"
            # 3. VIX spike — exit if regime turns VOLATILE (regime filter Phase 10)
            elif macro.get("vix_in") is not None:
                vix_now = float(macro["vix_in"].reindex([date]).ffill().iloc[0])                           if date in macro["vix_in"].index else 0
                if vix_now > 28:
                    exit_triggered = True; exit_reason = "VIX Crisis Exit"

            if exit_triggered:
                gross   = shares*p
                cost    = gross*TRANSACTION_COST
                capital = gross-cost
                ret_pct = (capital/(shares*entry_price)-1)*100
                shares=0; in_pos=False; trail_stop=0
                trades.append({"date":str(date.date()),"type":f"SELL ({exit_reason})",
                               "price":round(p,2),"return_pct":round(ret_pct,2)})

        pv = shares*p if in_pos else capital
        equity.append({"date":str(date.date()),"value":round(pv,2),"in_pos":in_pos})

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

    # ── Core Metrics ────────────────────────────────────────────────────────────
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
    max_dd_abs= float((roll_max-vals).max())

    sells     = [t for t in trades if "SELL" in t["type"] and "return_pct" in t]
    wins      = [t for t in sells if t["return_pct"]>0]
    losses    = [t for t in sells if t["return_pct"]<=0]
    win_rate  = len(wins)/len(sells)*100 if sells else 0
    gp        = sum(t["return_pct"] for t in wins)
    gl        = abs(sum(t["return_pct"] for t in losses))
    pf        = gp/gl if gl>0 else (float("inf") if gp>0 else 0)
    avg_ret   = sum(t["return_pct"] for t in sells)/len(sells) if sells else 0

    # ── Advanced Metrics ─────────────────────────────────────────────────────
    # Sortino ratio (uses downside deviation only)
    down_dev  = drets[drets<rf_d].std()*np.sqrt(252)
    sortino   = float(exc.mean()*252/down_dev) if down_dev>0 else 0

    # Calmar ratio (CAGR / |Max Drawdown|)
    calmar    = float(cagr/abs(max_dd)) if max_dd!=0 else 0

    # Recovery factor (net profit / max drawdown in ₹)
    recovery  = float((final_val-INITIAL_CAPITAL)/max_dd_abs) if max_dd_abs>0 else 0

    # Exposure % (time in market)
    in_market = sum(1 for e in equity if e.get("in_pos", False))
    exposure  = round(in_market/len(equity)*100, 1) if equity else 0

    # Rolling Sharpe (90-day)
    roll_sharpe_90 = (exc.rolling(90).mean()/exc.rolling(90).std()*np.sqrt(252)).fillna(0)
    roll_sharpe_30 = (exc.rolling(30).mean()/exc.rolling(30).std()*np.sqrt(252)).fillna(0)

    # ── Monte Carlo Simulation (500 runs, bootstrap resampling of trades) ─────
    mc_result = {"cagr_p5": 0, "cagr_p50": cagr, "cagr_p95": 0, "prob_positive": 0}
    trade_rets_arr = np.array([t["return_pct"]/100 for t in sells]) if len(sells) >= 5 else None
    if trade_rets_arr is not None and len(trade_rets_arr) >= 5:
        mc_cagrs = []
        for _ in range(500):
            sim = np.random.choice(trade_rets_arr, size=len(trade_rets_arr), replace=True)
            final_sim = np.prod(1+sim)*100   # normalised to 100 base
            trading_days = len(trade_rets_arr)*5   # ~5d avg hold
            mc_cagr = (final_sim/100)**(252/max(trading_days,1))-1
            mc_cagrs.append(mc_cagr*100)
        mc_result = {
            "cagr_p5":       round(float(np.percentile(mc_cagrs, 5)), 2),
            "cagr_p50":      round(float(np.percentile(mc_cagrs, 50)), 2),
            "cagr_p95":      round(float(np.percentile(mc_cagrs, 95)), 2),
            "prob_positive": round(float(np.mean([c>0 for c in mc_cagrs])*100), 1),
        }

    # ── Slippage Stress Test ──────────────────────────────────────────────────
    # Re-run P&L at 3 different cost levels to show sensitivity
    def _stress_pnl(cost_pct):
        c2, s2, ep2, in2 = INITIAL_CAPITAL, 0, 0, False
        for i,(date,pr) in enumerate(prices.items()):
            p2 = float(pr)
            if np.isnan(p2): continue
            sg = signals[i]
            if sg=="BUY" and not in2:
                s2 = (c2*(1-cost_pct))/p2; ep2=p2; in2=True; c2=0
            elif sg=="SELL" and in2:
                c2 = s2*p2*(1-cost_pct); in2=False; s2=0
        if in2:
            c2 = s2*float(prices.iloc[-1])*(1-cost_pct)
        return round((c2/INITIAL_CAPITAL-1)*100, 2)

    stress = {
        "cost_0.10pct": _stress_pnl(0.001),
        "cost_0.30pct": _stress_pnl(0.003),
        "cost_0.50pct": _stress_pnl(0.005),
    }

    # ── Regime Filter Stats ───────────────────────────────────────────────────
    # Stats on trades filtered by VIX level using macro data already downloaded
    vix_in = macro.get("vix_in")
    regime_stats = {}
    if vix_in is not None and not vix_in.empty and sells:
        vix_aligned = vix_in.reindex(prices.index).ffill().bfill()
        trade_dates = {t["date"] for t in sells}
        low_vix_trades  = [t for t in sells if vix_aligned.get(t["date"], 18) < 15]
        high_vix_trades = [t for t in sells if vix_aligned.get(t["date"], 18) > 22]
        mid_vix_trades  = [t for t in sells
                           if 15 <= vix_aligned.get(t["date"], 18) <= 22]
        def _regime_metrics(tlist):
            if not tlist: return {"count":0,"win_rate":0,"avg_ret":0}
            wr = sum(1 for t in tlist if t["return_pct"]>0)/len(tlist)*100
            ar = sum(t["return_pct"] for t in tlist)/len(tlist)
            return {"count":len(tlist),"win_rate":round(wr,1),"avg_ret":round(ar,2)}
        regime_stats = {
            "low_vix_lt15":   _regime_metrics(low_vix_trades),
            "mid_vix_15_22":  _regime_metrics(mid_vix_trades),
            "high_vix_gt22":  _regime_metrics(high_vix_trades),
        }

    # ── Walk-Forward Period Validation ────────────────────────────────────────
    # Split into 3 equal windows, compute metrics per window
    wf_results = []
    if len(prices) >= 200:
        chunk = len(prices)//3
        for wi in range(3):
            s_i = wi*chunk
            e_i = (wi+1)*chunk if wi<2 else len(prices)
            p_chunk = prices.iloc[s_i:e_i]
            s_chunk = signals[s_i:e_i]
            cap2, sh2, ep2, in2 = float(INITIAL_CAPITAL), 0.0, 0.0, False
            for k,(dt,pr) in enumerate(p_chunk.items()):
                p2 = float(pr)
                if np.isnan(p2): continue
                sg = s_chunk[k]
                if sg=="BUY" and not in2:
                    sh2=(cap2*(1-TRANSACTION_COST))/p2; ep2=p2; in2=True; cap2=0
                elif sg=="SELL" and in2:
                    cap2=sh2*p2*(1-TRANSACTION_COST); in2=False; sh2=0
            if in2: cap2=sh2*float(p_chunk.iloc[-1])*(1-TRANSACTION_COST)
            wf_ret = round((cap2/INITIAL_CAPITAL-1)*100, 2)
            wf_results.append({
                "period":f"Window {wi+1}",
                "start": str(p_chunk.index[0].date()),
                "end":   str(p_chunk.index[-1].date()),
                "return_pct": wf_ret,
            })

    # ── Nifty benchmark ───────────────────────────────────────────────────────
    nifty_p   = nc.reindex(prices.index).dropna()
    bench_ret = float((nifty_p.iloc[-1]/nifty_p.iloc[0]-1)*100) if len(nifty_p)>1 else 0
    nifty_norm= [round(float(v)/float(nifty_p.iloc[0])*100,2) for v in nifty_p]
    port_norm = [round(e["value"]/INITIAL_CAPITAL*100,2) for e in equity]
    dates_out = [e["date"] for e in equity]
    dd_curve  = [round(float(d),2) for d in drawdown]
    roll_s_out= [round(float(v),2) for v in roll_sharpe_90]

    return _json_safe({
        "metrics": {
            "total_return":     round(total_ret,2),
            "cagr":             round(cagr,2),
            "sharpe_ratio":     round(sharpe,2),
            "sortino_ratio":    round(sortino,2),
            "calmar_ratio":     round(calmar,2),
            "max_drawdown":     round(max_dd,2),
            "win_rate":         round(win_rate,2),
            "total_trades":     len(sells),
            "profit_factor":    round(pf,2) if pf!=float("inf") else 99.0,
            "avg_trade_return": round(avg_ret,2),
            "benchmark_return": round(bench_ret,2),
            "alpha":            round(total_ret-bench_ret,2),
            "recovery_factor":  round(recovery,2),
            "exposure_pct":     exposure,
            "initial_capital":  INITIAL_CAPITAL,
            "final_capital":    round(final_val,2),
        },
        "monte_carlo":    mc_result,
        "slippage_stress":stress,
        "regime_stats":   regime_stats,
        "walk_forward":   wf_results,
        "equity_curve": {
            "dates":          dates_out,
            "portfolio":      port_norm,
            "benchmark":      nifty_norm[:len(port_norm)],
            "drawdown":       dd_curve,
            "rolling_sharpe": roll_s_out,
        },
        "trades":  trades[-30:],
        "period":  period,
        "ticker":  ticker,
        "transaction_cost_pct": TRANSACTION_COST*100,
    })
