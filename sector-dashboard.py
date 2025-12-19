import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# =========================
# Configuration
# =========================

SECTOR_ETFS = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}
BENCH = "SPY"

# Breadth proxy: "sector leaders" list. Expand/replace with your universe later.
SECTOR_LEADERS = {
    "XLC": ["GOOGL", "GOOG", "META", "NFLX", "DIS", "TMUS", "T", "VZ", "CMCSA"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "BKNG", "SBUX", "TJX"],
    "XLP": ["PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "MDLZ", "CL"],
    "XLE": ["XOM", "CVX", "SLB", "COP", "EOG", "PSX", "MPC", "OXY", "VLO"],
    "XLF": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW"],
    "XLV": ["UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "PFE", "AMGN", "MDT"],
    "XLI": ["CAT", "BA", "HON", "UNP", "DE", "UPS", "GE", "RTX", "LMT"],
    "XLB": ["LIN", "SHW", "APD", "ECL", "NEM", "FCX", "DD", "DOW", "PPG"],
    "XLRE": ["PLD", "AMT", "EQIX", "PSA", "O", "WELL", "SPG", "DLR", "VICI"],
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "CRM", "ORCL", "AMD", "INTC", "ADBE"],
    "XLU": ["NEE", "DUK", "SO", "AEP", "EXC", "SRE", "XEL", "ED", "PEG"],
}

LOOKBACK_DAYS = 365  # enough for 90D and moving averages

# Score weights (tweakable)
W_RS20 = 0.35
W_RS50 = 0.20
W_RS90 = 0.10
W_TREND = 0.15
W_VOL = 0.10
W_BREADTH = 0.10

# =========================
# Helpers
# =========================

@st.cache_data(ttl=60 * 30)
def download_adj_close(symbols: list[str], start: str) -> pd.DataFrame:
    data = yf.download(
        tickers=" ".join(symbols),
        start=start,
        progress=False,
        auto_adjust=True,
        threads=True,
    )
    # yf returns MultiIndex columns when multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
        vol = data["Volume"].copy()
    else:
        close = data[["Close"]].rename(columns={"Close": symbols[0]})
        vol = data[["Volume"]].rename(columns={"Volume": symbols[0]})
    return close.dropna(how="all"), vol.dropna(how="all")

def safe_last(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else np.nan

def pct_change_n(s: pd.Series, n: int) -> float:
    s = s.dropna()
    if len(s) < n + 1:
        return np.nan
    return float((s.iloc[-1] / s.iloc[-(n+1)] - 1.0) * 100.0)

def rolling_zscore(s: pd.Series, window: int) -> float:
    s = s.dropna()
    if len(s) < window + 1:
        return np.nan
    recent = s.iloc[-window:]
    mu = recent.mean()
    sigma = recent.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    return float((recent.iloc[-1] - mu) / sigma)

def sma_last(s: pd.Series, window: int) -> float:
    if len(s.dropna()) < window:
        return np.nan
    return float(s.rolling(window).mean().iloc[-1])

def volume_ratio(vol: pd.Series, window: int = 30) -> float:
    vol = vol.dropna()
    if len(vol) < window + 1:
        return np.nan
    avg = vol.iloc[-window:].mean()
    if avg == 0 or np.isnan(avg):
        return np.nan
    return float(vol.iloc[-1] / avg)

def clamp01(x: float) -> float:
    if np.isnan(x):
        return np.nan
    return float(max(0.0, min(1.0, x)))

def normalize_rank(values: pd.Series) -> pd.Series:
    """
    Convert a series into 0..1 scores by rank (higher is better).
    NaNs remain NaN.
    """
    v = values.copy()
    mask = v.notna()
    if mask.sum() <= 1:
        return pd.Series(np.nan, index=v.index)
    ranks = v[mask].rank(ascending=True, method="average")
    return (ranks - 1) / (len(ranks) - 1)

# =========================
# Core Computation
# =========================

def compute_sector_features(close: pd.DataFrame, vol: pd.DataFrame, sector_symbols: dict) -> pd.DataFrame:
    """
    Returns a dataframe indexed by sector name with raw metrics:
    rs20, rs50, rs90 (relative returns vs SPY in %),
    rs20_z (zscore of ratio changes),
    trend (1 if above 50DMA else 0),
    vol_ratio,
    breadth (0..1)
    """
    spy = close[BENCH].dropna()

    rows = []
    for sector_name, etf in sector_symbols.items():
        if etf not in close.columns:
            continue
        s_close = close[etf].dropna()
        s_vol = vol[etf].dropna()

        # Align to spy dates
        df = pd.concat([s_close.rename("sector"), spy.rename("spy")], axis=1).dropna()
        if df.empty or len(df) < 120:
            continue

        # Relative return vs SPY over N days (sector_ret - spy_ret)
        rs20 = pct_change_n(df["sector"], 20) - pct_change_n(df["spy"], 20)
        rs50 = pct_change_n(df["sector"], 50) - pct_change_n(df["spy"], 50)
        rs90 = pct_change_n(df["sector"], 90) - pct_change_n(df["spy"], 90)

        # Ratio series and z-score of ratio ROC
        ratio = (df["sector"] / df["spy"]).dropna()
        ratio_roc20 = ratio.pct_change(20) * 100.0
        rs20_z = rolling_zscore(ratio_roc20, 60)

        # Trend: above 50DMA
        ma50 = df["sector"].rolling(50).mean()
        trend = 1.0 if df["sector"].iloc[-1] > ma50.iloc[-1] else 0.0

        # Volume expansion
        vr = volume_ratio(s_vol, 30)

        # Breadth proxy: % of leaders above 50DMA
        leaders = SECTOR_LEADERS.get(etf, [])
        breadth = np.nan
        if leaders:
            # Pull leaders (cached per run, but can still be heavy—keep list reasonable)
            leaders_close, _ = download_adj_close(leaders, start=start_date)
            latest = leaders_close.dropna(how="all")
            if not latest.empty:
                # Compute above 50DMA for each leader
                above = []
                for sym in leaders:
                    if sym not in leaders_close.columns:
                        continue
                    s = leaders_close[sym].dropna()
                    if len(s) < 60:
                        continue
                    m = s.rolling(50).mean().iloc[-1]
                    above.append(1.0 if s.iloc[-1] > m else 0.0)
                if len(above) >= 3:
                    breadth = float(np.mean(above))  # 0..1

        rows.append({
            "Sector": sector_name,
            "ETF": etf,
            "RS20_vs_SPY_%": rs20,
            "RS50_vs_SPY_%": rs50,
            "RS90_vs_SPY_%": rs90,
            "RS20_Z": rs20_z,
            "Trend_Above_50DMA": trend,
            "Volume_Ratio_30D": vr,
            "Breadth_Leaders_Above_50DMA": breadth,
        })

    return pd.DataFrame(rows).set_index("Sector").sort_index()

def compute_scores(features: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw features into 0..100 Sector Heat Score.
    """
    df = features.copy()

    # Normalize by rank (robust, no magic scaling)
    n_rs20 = normalize_rank(df["RS20_vs_SPY_%"])
    n_rs50 = normalize_rank(df["RS50_vs_SPY_%"])
    n_rs90 = normalize_rank(df["RS90_vs_SPY_%"])
    n_trend = df["Trend_Above_50DMA"]  # already 0/1
    n_vol = normalize_rank(df["Volume_Ratio_30D"])
    n_breadth = df["Breadth_Leaders_Above_50DMA"]  # already 0..1

    # If breadth missing, don’t zero it—just renormalize weights for that row
    scores = []
    drivers = []
    for idx in df.index:
        parts = {
            "RS20": n_rs20.loc[idx],
            "RS50": n_rs50.loc[idx],
            "RS90": n_rs90.loc[idx],
            "Trend": n_trend.loc[idx],
            "Vol": n_vol.loc[idx],
            "Breadth": n_breadth.loc[idx],
        }
        weights = {
            "RS20": W_RS20,
            "RS50": W_RS50,
            "RS90": W_RS90,
            "Trend": W_TREND,
            "Vol": W_VOL,
            "Breadth": W_BREADTH,
        }

        # Drop NaN components, renormalize
        valid = {k: v for k, v in parts.items() if pd.notna(v)}
        if not valid:
            scores.append(np.nan)
            drivers.append("")
            continue

        wsum = sum(weights[k] for k in valid.keys())
        sc = sum((valid[k] * weights[k] / wsum) for k in valid.keys()) * 100.0

        # Driver text (top 3 contributing normalized components)
        contrib = {k: (valid[k] * weights[k] / wsum) for k in valid.keys()}
        top = sorted(contrib.items(), key=lambda x: x[1], reverse=True)[:3]
        driver_txt = ", ".join([f"{k}:{contrib[k]*100:.1f}" for k, _ in top])

        scores.append(sc)
        drivers.append(driver_txt)

    df["Heat_Score"] = scores
    df["Top_Drivers"] = drivers
    return df.sort_values("Heat_Score", ascending=False)

def compute_sector_time_series(close: pd.DataFrame, etf: str) -> pd.DataFrame:
    """
    Builds a dataframe with ratio (ETF/SPY) and breadth time series.
    Breadth time series computed from leaders list.
    """
    if etf not in close.columns or BENCH not in close.columns:
        return pd.DataFrame()

    df = pd.concat([close[etf].rename("ETF"), close[BENCH].rename("SPY")], axis=1).dropna()
    df["Ratio"] = df["ETF"] / df["SPY"]
    df["Ratio_ROC20_%"] = df["Ratio"].pct_change(20) * 100.0

    # Breadth time series (leaders above 50DMA) - computed on leader closes
    leaders = SECTOR_LEADERS.get(etf, [])
    if leaders:
        leaders_close, _ = download_adj_close(leaders, start=start_date)
        leaders_close = leaders_close.dropna(how="all")
        if not leaders_close.empty:
            breadth_series = []
            for t in leaders_close.index:
                above = []
                for sym in leaders:
                    s = leaders_close[sym].loc[:t].dropna()
                    if len(s) < 60:
                        continue
                    m = s.rolling(50).mean().iloc[-1]
                    above.append(1.0 if s.iloc[-1] > m else 0.0)
                breadth_series.append(np.mean(above) if len(above) >= 3 else np.nan)
            df_b = pd.Series(breadth_series, index=leaders_close.index, name="Breadth")
            df = df.join(df_b, how="left")

    return df

def compute_top_stocks_in_sector(etf: str, start_date: str) -> pd.DataFrame:
    """
    Ranks sector leader stocks by RS vs sector ETF over 20/50D and basic filters.
    """
    leaders = SECTOR_LEADERS.get(etf, [])
    if not leaders:
        return pd.DataFrame()

    symbols = [etf] + leaders
    close, vol = download_adj_close(symbols, start=start_date)
    if etf not in close.columns:
        return pd.DataFrame()

    etf_close = close[etf].dropna()
    rows = []
    for sym in leaders:
        if sym not in close.columns:
            continue
        s = close[sym].dropna()
        if len(s) < 120 or len(etf_close) < 120:
            continue
        aligned = pd.concat([s.rename("stk"), etf_close.rename("etf")], axis=1).dropna()
        if len(aligned) < 120:
            continue

        rs20 = pct_change_n(aligned["stk"], 20) - pct_change_n(aligned["etf"], 20)
        rs50 = pct_change_n(aligned["stk"], 50) - pct_change_n(aligned["etf"], 50)

        ma50 = aligned["stk"].rolling(50).mean()
        trend = "Above" if aligned["stk"].iloc[-1] > ma50.iloc[-1] else "Below"

        vr = np.nan
        if sym in vol.columns:
            vr = volume_ratio(vol[sym], 30)

        rows.append({
            "Symbol": sym,
            "RS20_vs_Sector_%": rs20,
            "RS50_vs_Sector_%": rs50,
            "Trend_vs_50DMA": trend,
            "Volume_Ratio_30D": vr,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["RS20_vs_Sector_%", "RS50_vs_Sector_%"], ascending=False)

# =========================
# Streamlit App
# =========================

st.set_page_config(page_title="Sector Heat Dashboard", layout="wide")
st.title("Sector Heat Dashboard (MVP)")

today = datetime.utcnow().date()
start_date = (today - timedelta(days=LOOKBACK_DAYS)).isoformat()

with st.sidebar:
    st.header("Settings")
    lookback = st.selectbox("Lookback window for RS", [20, 50, 90], index=0)
    st.caption("This dashboard ranks sectors using RS vs SPY, trend, volume expansion, and breadth proxy.")
    st.divider()
    st.subheader("Scoring weights")
    st.write({
        "RS20": W_RS20, "RS50": W_RS50, "RS90": W_RS90,
        "Trend": W_TREND, "Volume": W_VOL, "Breadth": W_BREADTH
    })
    st.divider()
    show_raw = st.checkbox("Show raw metrics table", value=False)

# Download sector ETF + SPY
symbols = [BENCH] + list(SECTOR_ETFS.values())
close, vol = download_adj_close(symbols, start=start_date)

features = compute_sector_features(close, vol, SECTOR_ETFS)
ranked = compute_scores(features)

# =========================
# Leaderboard
# =========================
st.subheader("Sector Leaderboard")

colA, colB, colC = st.columns([2, 1, 1])

with colA:
    display_cols = ["ETF", "Heat_Score", "Top_Drivers", "RS20_vs_SPY_%", "RS50_vs_SPY_%", "RS90_vs_SPY_%", "Volume_Ratio_30D", "Breadth_Leaders_Above_50DMA"]
    st.dataframe(
        ranked[display_cols].style.format({
            "Heat_Score": "{:.1f}",
            "RS20_vs_SPY_%": "{:.2f}",
            "RS50_vs_SPY_%": "{:.2f}",
            "RS90_vs_SPY_%": "{:.2f}",
            "Volume_Ratio_30D": "{:.2f}",
            "Breadth_Leaders_Above_50DMA": "{:.2f}",
        }),
        use_container_width=True,
        height=420
    )

with colB:
    top3 = ranked.head(3)
    st.metric("Hot Sectors (Top 1)", f"{top3.index[0]}", f"{top3['Heat_Score'].iloc[0]:.1f}")
    st.write("Top 3:")
    for s in top3.index:
        st.write(f"- {s} ({ranked.loc[s,'ETF']}) — {ranked.loc[s,'Heat_Score']:.1f}")

with colC:
    bot3 = ranked.tail(3).sort_values("Heat_Score")
    st.metric("Cold Sectors (Bottom 1)", f"{bot3.index[0]}", f"{bot3['Heat_Score'].iloc[0]:.1f}")
    st.write("Bottom 3:")
    for s in bot3.index:
        st.write(f"- {s} ({ranked.loc[s,'ETF']}) — {ranked.loc[s,'Heat_Score']:.1f}")

if show_raw:
    st.subheader("Raw metrics")
    st.dataframe(features, use_container_width=True)

st.divider()

# =========================
# Drill-down
# =========================
st.subheader("Sector Drill-down")

sector_names = ranked.index.tolist()
default_sector = sector_names[0] if sector_names else list(SECTOR_ETFS.keys())[0]
picked_sector = st.selectbox("Pick a sector", sector_names, index=0 if sector_names else 0)

picked_etf = ranked.loc[picked_sector, "ETF"]
ts = compute_sector_time_series(close, picked_etf)

left, right = st.columns([1.2, 1])

with left:
    st.markdown(f"### {picked_sector} ({picked_etf}) vs {BENCH}")
    if ts.empty:
        st.warning("Not enough data to plot.")
    else:
        fig_ratio = px.line(ts.reset_index(), x="Date", y="Ratio", title="Ratio: Sector ETF / SPY")
        st.plotly_chart(fig_ratio, use_container_width=True)

        fig_roc = px.line(ts.reset_index(), x="Date", y="Ratio_ROC20_%", title="20D Ratio ROC (%)")
        st.plotly_chart(fig_roc, use_container_width=True)

with right:
    st.markdown("### Breadth (proxy)")
    if ts.empty or "Breadth" not in ts.columns:
        st.info("Breadth uses a leaders list per sector. Expand SECTOR_LEADERS for better coverage.")
    else:
        fig_b = px.line(ts.reset_index(), x="Date", y="Breadth", title="Leaders Above 50DMA (%)")
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("### Top Stocks inside sector (leaders list)")
    stocks = compute_top_stocks_in_sector(picked_etf, start_date=start_date)
    if stocks.empty:
        st.info("No stock list found for this sector, or not enough data.")
    else:
        st.dataframe(
            stocks.style.format({
                "RS20_vs_Sector_%": "{:.2f}",
                "RS50_vs_Sector_%": "{:.2f}",
                "Volume_Ratio_30D": "{:.2f}",
            }),
            use_container_width=True,
            height=360
        )

st.caption("Next upgrade: replace SECTOR_LEADERS with full constituent lists + true breadth (% above 50DMA across all members), then add options metrics (call/put, OI change, IV).")
