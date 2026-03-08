from __future__ import annotations

import textwrap

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analysis_engine import (
    INDICATOR_GLOSSARY,
    DataUnavailableError,
    allocate_portfolio,
    analyze_stock,
    evaluate_holdings,
    normalize_ticker,
    run_backtest,
    scan_universe,
)
from market_overview import (
    analyze_market_themes,
    build_stock_policy_context,
    rank_full_market_main_board_top5,
)
from local_store import (
    ensure_store_dirs,
    list_theme_snapshots,
    list_top_pick_snapshots,
    load_preferences,
    load_theme_snapshot,
    load_top_picks_snapshot,
    save_preferences,
    save_theme_snapshot,
    save_top_picks_snapshot,
)

st.set_page_config(page_title="缅A亏钱理论指南 by doctorbun", page_icon="📉", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;700&display=swap');
    :root {
        --bg: #f7f1e1;
        --card: rgba(255, 252, 244, 0.82);
        --ink: #1a2238;
        --muted: #5f6b7a;
        --accent: #0e8a62;
        --accent-2: #d84b45;
        --line: rgba(26, 34, 56, 0.1);
    }
    .stApp {
        background: radial-gradient(circle at top left, rgba(216, 111, 69, 0.18), transparent 28%), radial-gradient(circle at top right, rgba(14, 138, 98, 0.16), transparent 24%), linear-gradient(180deg, #fbf7ee 0%, #f2ead6 100%);
        color: var(--ink);
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    }
    h1, h2, h3 { font-family: "Space Grotesk", "Segoe UI", sans-serif; letter-spacing: -0.02em; }
    .hero, .metric-card, .section-card, .quote-card {
        border: 1px solid var(--line);
        border-radius: 22px;
        box-shadow: 0 12px 32px rgba(51, 60, 80, 0.06);
    }
    .hero {
        padding: 1.4rem 1.6rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.85), rgba(250, 239, 212, 0.88));
        margin-bottom: 1rem;
    }
    .hero-kicker { color: var(--accent); text-transform: uppercase; font-size: 0.78rem; letter-spacing: 0.18em; font-weight: 700; }
    .hero-copy { color: var(--muted); max-width: 76ch; }
    .metric-card { padding: 1rem 1.1rem; background: var(--card); min-height: 128px; }
    .metric-label { color: var(--muted); font-size: 0.85rem; margin-bottom: 0.25rem; }
    .metric-value { font-family: "Space Grotesk", "Segoe UI", sans-serif; font-size: 1.75rem; font-weight: 700; line-height: 1.05; }
    .metric-note { color: var(--muted); font-size: 0.85rem; margin-top: 0.4rem; }
    .section-card { padding: 1rem 1.2rem; background: rgba(255, 255, 255, 0.78); margin-bottom: 0.8rem; }
    .section-title { font-family: "Space Grotesk", "Segoe UI", sans-serif; font-size: 1rem; margin-bottom: 0.45rem; }
    .pill { display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.35rem 0.7rem; border-radius: 999px; background: rgba(14, 138, 98, 0.1); color: var(--accent); border: 1px solid rgba(14, 138, 98, 0.16); font-size: 0.84rem; font-weight: 600; }
    .quote-card { padding: 1rem 1.15rem; background: linear-gradient(145deg, rgba(255,255,255,0.92), rgba(245,233,205,0.86)); }
    .quote-body { font-size: 1rem; line-height: 1.6; }
    .quote-author { color: var(--muted); margin-top: 0.5rem; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

A_SHARE_DEFAULTS = "600519,000858,600036,601318,600276,000333"
US_DEFAULTS = "AAPL,MSFT,NVDA,AMZN,META,GOOGL"
HK_DEFAULTS = "0700.HK,9988.HK,3690.HK,1810.HK"
PERIOD_LABELS = {
    "1w": "1周",
    "1mo": "1个月",
    "3mo": "3个月",
    "6mo": "6个月",
    "1y": "1年",
    "2y": "2年",
    "5y": "5年",
}

ensure_store_dirs()
PREFERENCES = load_preferences()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_analyze_stock(ticker: str, period: str, capital: float, risk_per_trade: float, stop_loss_pct: float, as_of_date: str | None):
    return analyze_stock(ticker=ticker, period=period, capital=capital, risk_per_trade=risk_per_trade, stop_loss_pct=stop_loss_pct, as_of_date=as_of_date)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_scan_universe(tickers: tuple[str, ...], period: str, capital: float, risk_per_trade: float, stop_loss_pct: float):
    return scan_universe(tickers=tickers, period=period, capital=capital, risk_per_trade=risk_per_trade, stop_loss_pct=stop_loss_pct)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_backtest(ticker: str, period: str, stop_loss_pct: float):
    result = analyze_stock(ticker=ticker, period=period)
    return result, run_backtest(result.frame, stop_loss_pct=stop_loss_pct / 100)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_allocate_portfolio(tickers: tuple[str, ...], period: str, capital: float, risk_per_trade: float, stop_loss_pct: float, total_risk_limit_pct: float, max_position_pct: float, reserve_cash_pct: float, max_positions: int):
    return allocate_portfolio(tickers=tickers, period=period, capital=capital, risk_per_trade=risk_per_trade, stop_loss_pct=stop_loss_pct, total_risk_limit_pct=total_risk_limit_pct, max_position_pct=max_position_pct, reserve_cash_pct=reserve_cash_pct, max_positions=max_positions)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_evaluate_holdings(positions: tuple[tuple[str, float, float], ...], period: str, capital: float, risk_per_trade: float, stop_loss_pct: float):
    payload = [{"ticker": ticker, "shares": shares, "cost": cost} for ticker, shares, cost in positions]
    return evaluate_holdings(payload, period=period, capital=capital, risk_per_trade=risk_per_trade, stop_loss_pct=stop_loss_pct)

@st.cache_data(ttl=21600, show_spinner=False)
def cached_top_picks(period: str, capital: float, risk_per_trade: float, stop_loss_pct: float, refresh_token: int = 0):
    return rank_full_market_main_board_top5(period=period, capital=capital, risk_per_trade=risk_per_trade, stop_loss_pct=stop_loss_pct, top_n=None, candidate_count=100)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_stock_context(ticker: str, period: str, capital: float, risk_per_trade: float, stop_loss_pct: float, as_of_date: str | None):
    result = analyze_stock(ticker=ticker, period=period, capital=capital, risk_per_trade=risk_per_trade, stop_loss_pct=stop_loss_pct, as_of_date=as_of_date)
    return build_stock_policy_context(result)


@st.cache_data(ttl=21600, show_spinner=False)
def cached_market_themes(period: str, capital: float, risk_per_trade: float, stop_loss_pct: float, refresh_token: int = 0):
    return analyze_market_themes(period=period, capital=capital, risk_per_trade=risk_per_trade, stop_loss_pct=stop_loss_pct, max_themes=10)


def pct(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value * 100:.2f}%"


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>", unsafe_allow_html=True)


def section_card(title: str, body: str) -> None:
    st.markdown(f"<div class='section-card'><div class='section-title'>{title}</div><div>{body}</div></div>", unsafe_allow_html=True)


def quote_card(quote: dict[str, str]) -> None:
    st.markdown(f"<div class='quote-card'><div class='section-title'>今日决策台名言</div><div class='quote-body'>“{quote['quote']}”</div><div class='quote-author'>- {quote['author']}</div></div>", unsafe_allow_html=True)


def format_indicator_value(key: str, latest: pd.Series) -> str:
    value = latest.get(key)
    if pd.isna(value):
        return "N/A"
    if key in {"ATRPercent", "Return20", "Return60"}:
        return f"{value * 100:.2f}%"
    if key == "Turnover":
        return f"{value:.2f}%"
    if key in {"MACDHist", "TrendSlope60"}:
        return f"{value:.3f}"
    if key == "OBV":
        return f"{value:,.0f}"
    return f"{value:.2f}"


def build_indicator_table(result) -> pd.DataFrame:
    latest = result.frame.iloc[-1]
    rows = []
    for key in ["SMA20", "SMA50", "SMA200", "RSI14", "MACDHist", "ATRPercent", "VolumeRatio", "Turnover", "OBV", "Return20", "Return60", "TrendLine60"]:
        glossary = INDICATOR_GLOSSARY.get(key, {"meaning": "", "formula": ""})
        rows.append({"指标": key, "当前值": format_indicator_value(key, latest), "意义": glossary["meaning"], "计算方法": glossary["formula"]})
    rows.append({"指标": "支撑位", "当前值": ", ".join(f"{level:.2f}" for level in result.support_levels) if result.support_levels else "N/A", "意义": "近 160 日内更容易被资金承接的位置。", "计算方法": "由近 160 日枢轴低点与 20/60 日低点聚类而成"})
    rows.append({"指标": "压力位", "当前值": ", ".join(f"{level:.2f}" for level in result.resistance_levels) if result.resistance_levels else "N/A", "意义": "价格上行时更容易遇到抛压的位置。", "计算方法": "由近 160 日枢轴高点与 20/60 日高点聚类而成"})
    return pd.DataFrame(rows)

def build_price_chart(result) -> go.Figure:
    frame = result.frame.copy()
    x_values = frame.index.strftime("%Y-%m-%d")
    tickvals, ticktext = build_sparse_trade_ticks(frame.index)
    colors = ["#d84b45" if close >= open_ else "#1f9d55" for open_, close in zip(frame["Open"], frame["Close"])]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.72, 0.28])
    fig.add_trace(go.Candlestick(x=x_values, open=frame["Open"], high=frame["High"], low=frame["Low"], close=frame["Close"], name="K线", increasing_line_color="#d84b45", decreasing_line_color="#1f9d55", increasing_fillcolor="#d84b45", decreasing_fillcolor="#1f9d55"), row=1, col=1)
    for line, label, color, dash in (("SMA20", "20日线", "#1d3557", "solid"), ("SMA50", "50日线", "#0e8a62", "solid"), ("SMA200", "200日线", "#d68c45", "solid"), ("TrendLine60", "60日趋势线", "#6d597a", "dash"), ("TrendUpper60", "趋势上轨", "#8d99ae", "dot"), ("TrendLower60", "趋势下轨", "#8d99ae", "dot")):
        if line in frame.columns:
            fig.add_trace(go.Scatter(x=x_values, y=frame[line], mode="lines", name=label, line=dict(width=2 if "Trend" not in line else 1.8, color=color, dash=dash)), row=1, col=1)
    for idx, level in enumerate(result.support_levels[:2], start=1):
        fig.add_hline(y=level, line_dash="dot", line_color="#0e8a62", annotation_text=f"支撑 {idx}: {level:.2f}", annotation_position="bottom right", row=1, col=1)
    for idx, level in enumerate(result.resistance_levels[:2], start=1):
        fig.add_hline(y=level, line_dash="dot", line_color="#d84b45", annotation_text=f"压力 {idx}: {level:.2f}", annotation_position="top right", row=1, col=1)
    fig.add_trace(go.Bar(x=x_values, y=frame["Volume"], name="成交量", marker_color=colors, opacity=0.75), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_values, y=frame["VolumeMA20"], mode="lines", name="20日均量", line=dict(color="#1d3557", width=1.6)), row=2, col=1)
    fig.update_layout(title=f"{result.stock_name} {result.ticker} K线、趋势线与量能", height=760, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.72)", margin=dict(l=10, r=10, t=55, b=10), xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_xaxes(showgrid=False, type="category", tickmode="array", tickvals=tickvals, ticktext=ticktext, tickangle=0)
    return fig


def build_score_chart(result) -> go.Figure:
    labels = ["技术", "量价", "基本面", "消息", "风险", "综合"]
    values = [result.technical_score, result.volume_score, result.fundamental_score, result.news_score, result.risk_score, result.total_score]
    colors = ["#1d3557", "#d86f45", "#345995", "#0e8a62", "#8d99ae", "#14213d"]
    fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors, text=values, textposition="outside")])
    fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.7)", margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(range=[0, 105], title="分数"), xaxis_title=None)
    return fig


def build_equity_chart(backtest) -> go.Figure:
    frame = backtest.frame
    x_values = frame.index.strftime("%Y-%m-%d")
    tickvals, ticktext = build_sparse_trade_ticks(frame.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=frame["StrategyEquity"], mode="lines", name="策略净值", line=dict(color="#0e8a62", width=2.5)))
    fig.add_trace(go.Scatter(x=x_values, y=frame["BuyHoldEquity"], mode="lines", name="买入持有", line=dict(color="#1d3557", width=2.2)))
    fig.add_trace(go.Scatter(x=x_values, y=frame["EntryMarker"], mode="markers", name="买点", marker=dict(color="#d84b45", size=8, symbol="triangle-up")))
    fig.add_trace(go.Scatter(x=x_values, y=frame["ExitMarker"], mode="markers", name="卖点", marker=dict(color="#1f9d55", size=8, symbol="triangle-down")))
    fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.7)", margin=dict(l=10, r=10, t=20, b=10), yaxis_title="净值", xaxis_title=None)
    fig.update_xaxes(type="category", tickmode="array", tickvals=tickvals, ticktext=ticktext)
    return fig


def build_sparse_trade_ticks(index: pd.Index, max_ticks: int = 9) -> tuple[list[str], list[str]]:
    dates = pd.to_datetime(index)
    labels = dates.strftime("%Y-%m-%d").tolist()
    if not labels:
        return [], []
    if len(labels) <= max_ticks:
        return labels, [date.strftime("%m-%d") for date in dates]
    step = max(1, len(labels) // max_ticks)
    picks = list(range(0, len(labels), step))
    if picks[-1] != len(labels) - 1:
        picks.append(len(labels) - 1)
    tickvals = [labels[idx] for idx in picks]
    ticktext: list[str] = []
    for idx in picks:
        date = dates[idx]
        if len(labels) <= 25:
            ticktext.append(date.strftime("%m-%d"))
        elif date.month == 1 or idx == 0 or idx == len(labels) - 1:
            ticktext.append(date.strftime("%Y-%m"))
        else:
            ticktext.append(date.strftime("%m-%d"))
    return tickvals, ticktext


def build_company_table(profile: dict[str, str]) -> pd.DataFrame:
    if not profile:
        return pd.DataFrame(columns=["项目", "内容"])
    ordered = [key for key in ["公司名称", "所属行业", "所属市场", "上市日期", "总股本", "估算总市值", "实控人", "法人代表", "员工人数", "官方网站", "联系电话", "办公地址", "主营业务", "公司简介"] if key in profile]
    remaining = [key for key in profile.keys() if key not in ordered]
    keys = ordered + remaining
    return pd.DataFrame([{"项目": key, "内容": profile[key]} for key in keys])


def build_valuation_table(result) -> pd.DataFrame:
    frame = pd.DataFrame(result.valuation_items)
    ordered = [column for column in ["方法", "当前值", "判断", "合理价", "安全买点", "潜在空间", "说明"] if column in frame.columns]
    remaining = [column for column in frame.columns if column not in ordered]
    return frame[ordered + remaining] if not frame.empty else frame


def build_news_table(result) -> pd.DataFrame:
    if not result.news_items:
        return pd.DataFrame(columns=["发布时间", "类型", "标题", "来源", "情绪", "链接"])
    frame = pd.DataFrame(result.news_items)
    ordered = [column for column in ["发布时间", "类型", "标题", "来源", "情绪", "链接"] if column in frame.columns]
    remaining = [column for column in frame.columns if column not in ordered]
    return frame[ordered + remaining]


def build_trade_plan(result) -> pd.DataFrame:
    latest = result.frame.iloc[-1]
    sma20 = latest.get("SMA20", result.price)
    trend_line = latest.get("TrendLine60", result.price)
    base_support = result.support_levels[0] if result.support_levels else min(sma20, trend_line)
    base_resistance = next((level for level in result.resistance_levels if level > result.price), latest.get("High20", result.price))
    breakout_buy = max(base_resistance, latest.get("High20", result.price)) * 1.005
    pullback_buy_low = max(base_support * 0.995, min(sma20, trend_line) * 0.995)
    pullback_buy_high = max(base_support * 1.01, pullback_buy_low)
    reduce_price = result.resistance_levels[1] if len(result.resistance_levels) > 1 else breakout_buy + max(result.price - result.stop_price, result.price * 0.05)
    return pd.DataFrame([
        {"动作": "突破买点", "价格/区间": f"站上 {breakout_buy:.2f}", "条件": "当日收盘站上该价位，且量能不能弱于 20 日均量"},
        {"动作": "回踩买点", "价格/区间": f"{pullback_buy_low:.2f} - {pullback_buy_high:.2f}", "条件": "回踩支撑或 SMA20/趋势线附近不破，可小仓试错"},
        {"动作": "减仓观察", "价格/区间": f"{reduce_price:.2f} 附近", "条件": "接近第二压力位或短线拉离均线过快时，优先锁定利润"},
        {"动作": "失效位", "价格/区间": f"跌破 {result.stop_price:.2f}", "条件": "跌破后说明预判失效，仓位要及时收缩"},
    ])


def snapshot_label(filename: str) -> str:
    if filename.startswith("latest_"):
        return "最新结果"
    stem = filename.removesuffix(".csv")
    stamp = stem.split("_")[-2:] if "_" in stem else [stem]
    return " ".join(stamp)


def normalize_top_ranking_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return frame
    normalized = frame.copy()
    if "综合总分" not in normalized.columns and "Score" in normalized.columns:
        normalized["综合总分"] = normalized["Score"]
    if "评分原因" not in normalized.columns:
        normalized["评分原因"] = normalized.get("Decision", "暂无说明")
    if "Name" not in normalized.columns and "名称" in normalized.columns:
        normalized["Name"] = normalized["名称"]
    return normalized


st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">缅A亏钱理论指南 by doctorbun</div>
        <h1>技术、量能、基本面、消息面、估值和仓位控制放到同一张决策台</h1>
        <p class="hero-copy">这版重点偏向 A 股。除了 K 线和指标，它会同时给出公司名称与基本信息、量价关系、估值判断、趋势线与撑压、组合仓位建议、全主板排名、主线板块和规则回测。</p>
        <div class="pill">不能保证赚钱，但能把研究流程和风险控制做得更完整</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("参数")
    market = st.selectbox("市场", options=["A股", "美股", "港股"], index=0)
    analysis_mode = st.selectbox("分析模式", options=["精确优先", "快速缓存"], index=0, help="精确优先会尽量实时抓取更全的信息；快速缓存会复用最近计算结果。")
    default_ticker = {"A股": "600519", "美股": "AAPL", "港股": "0700.HK"}[market]
    default_universe = {"A股": A_SHARE_DEFAULTS, "美股": US_DEFAULTS, "港股": HK_DEFAULTS}[market]
    market_code = {"A股": "cn", "美股": "us", "港股": "hk"}[market]
    ticker_input = st.text_input("股票代码", value=default_ticker, help="A股可直接输入 6 位代码，例如 600519、000858、600036")
    period = st.selectbox("历史区间", options=list(PERIOD_LABELS.keys()), index=4, format_func=lambda item: PERIOD_LABELS[item])
    use_historical_as_of = st.checkbox("按指定日期回看单票", value=False, help="勾选后，单票分析会按你输入日期之前最近的交易日，回看该日期之前所选周期内的量价、消息和财务口径。")
    as_of_input = st.date_input("回看日期", value=pd.Timestamp.now().date(), max_value=pd.Timestamp.now().date(), disabled=not use_historical_as_of)
    capital = st.number_input("账户资金", min_value=1000.0, value=100000.0, step=1000.0)
    risk_per_trade = st.slider("单笔风险占比 %", min_value=0.25, max_value=3.0, value=1.0, step=0.25)
    stop_loss_pct = st.slider("止损比例 %", min_value=3.0, max_value=15.0, value=8.0, step=0.5)
    total_risk_limit_pct = st.slider("组合总风险上限 %", min_value=1.0, max_value=8.0, value=4.0, step=0.5)
    max_position_pct = st.slider("单票最大仓位 %", min_value=5.0, max_value=40.0, value=25.0, step=1.0)
    reserve_cash_pct = st.slider("保留现金 %", min_value=0.0, max_value=40.0, value=15.0, step=1.0)
    ticker = normalize_ticker(ticker_input, market=market_code)
    analysis_as_of = pd.Timestamp(as_of_input).strftime("%Y-%m-%d") if use_historical_as_of else None
    st.caption("研究顺序")
    st.caption("1. 趋势和量价先过关")
    st.caption("2. 基本面和估值不能明显拖后腿")
    st.caption("3. 消息面只做辅助，不替代研报")
    st.caption("4. 先算仓位，再谈买入")

precise_mode = analysis_mode == "精确优先"
st.session_state.setdefault("top_picks_result", None)
st.session_state.setdefault("top_picks_meta", "")
st.session_state.setdefault("top_picks_refresh_token", 0)
st.session_state.setdefault("theme_result", None)
st.session_state.setdefault("theme_rankings", None)
st.session_state.setdefault("theme_meta", "")
st.session_state.setdefault("theme_refresh_token", 0)
st.session_state.setdefault("saved_allocation_inputs", PREFERENCES.get("allocation_inputs", {}))
st.session_state.setdefault("allocation_text", st.session_state["saved_allocation_inputs"].get(market, A_SHARE_DEFAULTS if market == "A股" else default_universe))

if st.session_state["top_picks_result"] is None:
    stored_top_df, stored_top_meta = load_top_picks_snapshot()
    if stored_top_df is not None:
        st.session_state["top_picks_result"] = stored_top_df
        st.session_state["top_picks_meta"] = stored_top_meta.get("label", stored_top_meta.get("timestamp", ""))

if st.session_state["theme_result"] is None:
    stored_theme_df, stored_theme_rankings, stored_theme_meta = load_theme_snapshot()
    if stored_theme_df is not None:
        st.session_state["theme_result"] = stored_theme_df
        st.session_state["theme_rankings"] = stored_theme_rankings
        st.session_state["theme_meta"] = stored_theme_meta.get("label", stored_theme_meta.get("timestamp", ""))

single_tab, holdings_tab, allocation_tab, picks_tab, themes_tab, scanner_tab, backtest_tab, guide_tab = st.tabs(["单票分析", "持仓决策", "资金分配", "全主板排名", "主线板块", "股票池扫描", "策略回放", "使用说明"])

with single_tab:
    try:
        with st.spinner(f"正在分析 {ticker.upper()} ..."):
            if precise_mode:
                result = analyze_stock(ticker=ticker, period=period, capital=capital, risk_per_trade=risk_per_trade, stop_loss_pct=stop_loss_pct, as_of_date=analysis_as_of)
            else:
                result = cached_analyze_stock(ticker, period, capital, risk_per_trade, stop_loss_pct, analysis_as_of)
            if analysis_as_of:
                context = {"summary": "历史回看模式下，政策与主线联动没有做全历史回放，这里不混入当前时点的主线判断。", "policy_items": [], "macro_items": []}
            elif precise_mode:
                context = build_stock_policy_context(result)
            else:
                context = cached_stock_context(ticker, period, capital, risk_per_trade, stop_loss_pct, analysis_as_of)
        st.subheader(f"{result.stock_name} ({result.ticker})")
        if analysis_as_of:
            st.caption(f"历史回看模式: 按 {analysis_as_of} 之前最近交易日，回看此前 {PERIOD_LABELS[period]} 的量价、消息和财务口径。")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            metric_card("最新价格", f"{result.price:,.2f}", f"截至 {result.as_of}")
        with c2:
            metric_card("综合评分", str(result.total_score), f"技术 {result.technical_score} / 量价 {result.volume_score}")
        with c3:
            metric_card("决策结论", result.decision, f"基本面 {result.fundamental_score} / 消息 {result.news_score}")
        with c4:
            metric_card("理论仓位股数", f"{result.position_size:,}", f"参考止损 {result.stop_price:,.2f}" if pd.notna(result.stop_price) else "未启用")
        with c5:
            support = f"{result.support_levels[0]:.2f}" if result.support_levels else "N/A"
            resistance = f"{result.resistance_levels[0]:.2f}" if result.resistance_levels else "N/A"
            metric_card("支撑 / 压力", f"{support} / {resistance}", f"风险分 {result.risk_score}")
        left, right = st.columns([1.75, 1])
        with left:
            st.plotly_chart(build_price_chart(result), width="stretch")
        with right:
            st.plotly_chart(build_score_chart(result), width="stretch")
            quote_card(result.quote)
            section_card("最终建议", result.decision_details["最终建议"])
        info_col, detail_col = st.columns([1.05, 1.25])
        with info_col:
            st.subheader("公司与估值")
            st.dataframe(build_company_table(result.company_profile), width="stretch", hide_index=True)
            st.dataframe(build_valuation_table(result), width="stretch", hide_index=True)
            st.subheader("买卖点提示")
            st.dataframe(build_trade_plan(result), width="stretch", hide_index=True)
        with detail_col:
            for title in ["技术面", "量价关系", "基本面", "估值视角", "消息面", "风险控制"]:
                section_card(title, result.decision_details[title])
            section_card("政策与主线联动", context["summary"])
        st.subheader("指标说明")
        st.dataframe(build_indicator_table(result), width="stretch", hide_index=True)
        news_col, policy_col, summary_col = st.columns([1.15, 1, 1])
        with news_col:
            st.subheader("公司消息")
            news_df = build_news_table(result)
            if news_df.empty:
                st.info("当前没有抓到相关新闻。")
            else:
                st.dataframe(news_df, width="stretch", hide_index=True)
        with policy_col:
            st.subheader("政策 / 宏观催化")
            if context["policy_items"]:
                st.dataframe(pd.DataFrame([{"日期": item["date"], "主题": item["theme"], "来源": item["source"], "摘要": item["summary"]} for item in context["policy_items"]]), width="stretch", hide_index=True)
            if context["macro_items"]:
                st.dataframe(pd.DataFrame(context["macro_items"]), width="stretch", hide_index=True)
            if not context["policy_items"] and not context["macro_items"]:
                st.info("当前没有匹配到明显的政策或宏观催化。")
        with summary_col:
            st.subheader("综合摘要")
            st.write(result.summary)
            st.caption("现在的消息面会合并公司新闻、公告、研报、财联社电报、社交舆情、行业/市场新闻和政策催化；但它仍然是程序化归纳，不是人工研报。")
    except DataUnavailableError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.exception(exc)


with holdings_tab:
    st.subheader("持仓决策")
    st.caption("每行一个持仓，格式: 股票代码, 持仓股数, 成本价。示例: `600519,300,1432.50`")
    holdings_text = st.text_area("输入当前持仓", value="600519,100,1430\n600879,200,9.85", height=130)
    if st.button("分析当前持仓", type="primary"):
        parsed_positions: list[tuple[str, float, float]] = []
        for raw_line in holdings_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [item.strip() for item in line.replace("，", ",").split(",")]
            if len(parts) < 3:
                continue
            try:
                parsed_positions.append((normalize_ticker(parts[0], market=market_code), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
        if not parsed_positions:
            st.warning("没有识别到有效持仓，请按 `代码,股数,成本价` 填写。")
        else:
            with st.spinner("正在分析持仓..."):
                holdings_df = evaluate_holdings(
                    [{"ticker": ticker, "shares": shares, "cost": cost} for ticker, shares, cost in parsed_positions],
                    period=period,
                    capital=capital,
                    risk_per_trade=risk_per_trade,
                    stop_loss_pct=stop_loss_pct,
                ) if precise_mode else cached_evaluate_holdings(tuple(parsed_positions), period, capital, risk_per_trade, stop_loss_pct)
            st.dataframe(holdings_df, width="stretch", hide_index=True)
            st.caption("做 T 提示是基于当前趋势、支撑/压力、你的成本区和程序化风控给出的纪律建议，不是保证盈利的信号。")
with allocation_tab:
    st.subheader("资金分配与严格仓位控制")
    allocation_default = st.session_state["saved_allocation_inputs"].get(market, A_SHARE_DEFAULTS if market == "A股" else default_universe)
    allocation_text = st.text_area("输入准备比较的股票代码，用逗号分隔", value=st.session_state.get(f"allocation_text_{market}", allocation_default), height=110, key=f"allocation_text_{market}")
    st.session_state["saved_allocation_inputs"][market] = allocation_text
    save_preferences({"allocation_inputs": st.session_state["saved_allocation_inputs"]})
    if st.button("生成组合建议", type="primary"):
        tickers = tuple(normalize_ticker(item.strip(), market=market_code) for item in allocation_text.replace("\n", ",").split(",") if item.strip())
        with st.spinner("正在计算组合仓位..."):
            plan = allocate_portfolio(
                tickers=tickers,
                period=period,
                capital=capital,
                risk_per_trade=risk_per_trade,
                stop_loss_pct=stop_loss_pct,
                total_risk_limit_pct=total_risk_limit_pct,
                max_position_pct=max_position_pct,
                reserve_cash_pct=reserve_cash_pct,
                max_positions=5,
            ) if precise_mode else cached_allocate_portfolio(tickers, period, capital, risk_per_trade, stop_loss_pct, total_risk_limit_pct, max_position_pct, reserve_cash_pct, 5)
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            metric_card("预计投入", f"{plan.total_allocated:,.0f}", f"投入比例 {plan.invested_ratio * 100:.2f}%")
        with a2:
            metric_card("剩余现金", f"{plan.cash_left:,.0f}", f"保留现金 {reserve_cash_pct:.1f}% 目标")
        with a3:
            metric_card("组合风险预算", f"{plan.total_risk_budget:,.0f}", f"上限 {total_risk_limit_pct:.1f}%")
        with a4:
            metric_card("预计止损风险", f"{plan.total_estimated_risk:,.0f}", "按建议股数与止损位估算")
        if plan.plan.empty:
            st.warning("当前没有满足门槛的标的。可以放宽候选池，或者继续等待。")
        else:
            st.dataframe(plan.plan, width="stretch", hide_index=True)
            st.caption("逻辑: 先用综合评分筛选，再受组合总风险、单票最大仓位、止损距离和保留现金四重约束。")
        if plan.notes:
            st.info("；".join(plan.notes))

with picks_tab:
    st.subheader("全A股主板排名")
    st.caption("只在你主动点击时刷新。结果会保留在页面状态里，不会因为单票分析 rerun 被覆盖。算法会先扫全主板实时快照，再只对预筛前 100 只候选股做完整多维评分，并把结果落到本地 CSV。")
    history_options = list_top_pick_snapshots()
    if not history_options:
        history_options = ["latest_top_picks.csv"]
    picks_col1, picks_col2 = st.columns([1.2, 1])
    with picks_col1:
        selected_top_snapshot = st.selectbox("本地结果", options=history_options, format_func=snapshot_label, key="top_picks_snapshot_select")
    with picks_col2:
        display_count = st.selectbox("展示数量", options=[10, 20, 50, 100], index=0, key="top_picks_display_count")
    if st.button("刷新全主板排名"):
        with st.spinner("正在扫描全主板候选股，这一步会比较慢..."):
            st.session_state["top_picks_refresh_token"] += 1
            st.session_state["top_picks_result"] = rank_full_market_main_board_top5(
                period=period,
                capital=capital,
                risk_per_trade=risk_per_trade,
                stop_loss_pct=stop_loss_pct,
                top_n=None,
                candidate_count=100,
            ) if precise_mode else cached_top_picks(period, capital, risk_per_trade, stop_loss_pct, st.session_state["top_picks_refresh_token"])
            now_label = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            meta = {"timestamp": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"), "label": f"最近刷新: {now_label}", "rows": int(len(st.session_state["top_picks_result"]))}
            save_top_picks_snapshot(st.session_state["top_picks_result"], meta)
            st.session_state["top_picks_meta"] = meta["label"]
    if selected_top_snapshot != "latest_top_picks.csv":
        top_df, top_meta = load_top_picks_snapshot(selected_top_snapshot)
    else:
        top_df, top_meta = st.session_state.get("top_picks_result"), {"label": st.session_state.get("top_picks_meta", "")}
    top_df = normalize_top_ranking_frame(top_df)
    if top_df is None:
        st.info("还没有刷新过全主板排名。")
    elif top_df.empty:
        st.warning("当前没有拿到全主板排名结果。")
    else:
        st.caption(top_meta.get("label", st.session_state.get("top_picks_meta", "")))
        display_df = top_df.copy().head(display_count)
        st.dataframe(display_df, width="stretch", hide_index=True)
        name_col = "Name" if "Name" in display_df.columns else "Ticker"
        score_col = "综合总分" if "综合总分" in display_df.columns else "Score"
        reason_col = "评分原因" if "评分原因" in display_df.columns else "Decision"
        chart = go.Figure(data=[go.Bar(x=display_df[name_col], y=display_df[score_col], marker_color="#0e8a62", text=display_df[reason_col], hovertemplate="%{x}<br>综合总分=%{y:.1f}<br>%{text}<extra></extra>")])
        chart.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.7)", margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(title="综合总分"), xaxis_title=None)
        st.plotly_chart(chart, width="stretch")
        st.download_button("下载当前全主板结果 CSV", data=display_df.to_csv(index=False).encode("utf-8-sig"), file_name=f"a_main_board_top_{display_count}.csv", mime="text/csv")
        st.caption("这里更偏向‘有爆发潜力且能过基本风控’的标的，不是单纯看涨幅榜。")

with themes_tab:
    st.subheader("当前主线板块")
    st.caption("只在你主动点击时刷新。这里会优先扫描前 10 个主线主题，把政策催化、宏观/行业新闻热度、主题池强弱和主题池内个股评分合在一起看。")
    theme_history_options = list_theme_snapshots()
    if not theme_history_options:
        theme_history_options = ["latest_themes_summary.csv"]
    theme_col1, theme_col2 = st.columns([1.2, 1])
    with theme_col1:
        selected_theme_snapshot = st.selectbox("本地主线结果", options=theme_history_options, format_func=snapshot_label, key="theme_snapshot_select")
    with theme_col2:
        theme_stock_limit = st.selectbox("板块内展示数量", options=[5, 10], index=1, key="theme_stock_limit")
    if st.button("刷新主线板块"):
        with st.spinner("正在汇总政策、新闻和主题池评分，这一步会比较慢..."):
            st.session_state["theme_refresh_token"] += 1
            theme_df, theme_rankings = analyze_market_themes(
                period=period,
                capital=capital,
                risk_per_trade=risk_per_trade,
                stop_loss_pct=stop_loss_pct,
                max_themes=10,
            ) if precise_mode else cached_market_themes(period, capital, risk_per_trade, stop_loss_pct, st.session_state["theme_refresh_token"])
            st.session_state["theme_result"] = theme_df
            st.session_state["theme_rankings"] = theme_rankings
            now_label = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            meta = {"timestamp": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"), "label": f"最近刷新: {now_label}", "themes": int(len(theme_df))}
            save_theme_snapshot(theme_df, theme_rankings, meta)
            st.session_state["theme_meta"] = meta["label"]
    if selected_theme_snapshot != "latest_themes_summary.csv":
        theme_df, theme_rankings, theme_meta = load_theme_snapshot(selected_theme_snapshot)
    else:
        theme_df, theme_rankings, theme_meta = st.session_state.get("theme_result"), st.session_state.get("theme_rankings"), {"label": st.session_state.get("theme_meta", "")}
    if theme_df is None:
        st.info("还没有刷新过主线板块。")
    elif theme_df.empty:
        st.warning("当前没有拿到主线板块结果。")
    else:
        st.caption(theme_meta.get("label", st.session_state.get("theme_meta", "")))
        st.dataframe(theme_df, width="stretch", hide_index=True)
        st.download_button("下载当前主线板块 CSV", data=theme_df.to_csv(index=False).encode("utf-8-sig"), file_name="theme_summary.csv", mime="text/csv")
        for _, row in theme_df.iterrows():
            theme_name = row["主线主题"]
            st.markdown(f"### {theme_name}")
            section_card("主题判断", str(row["判断"]))
            ranking = theme_rankings.get(theme_name) if theme_rankings else None
            if ranking is not None and not ranking.empty:
                display_cols = [column for column in ["Ticker", "Name", "Score", "Technical", "Volume", "Fundamental", "News", "Risk", "Decision", "评分原因"] if column in ranking.columns]
                st.dataframe(ranking[display_cols].head(theme_stock_limit), width="stretch", hide_index=True)

with scanner_tab:
    st.subheader("股票池排序")
    universe_text = st.text_area("输入一组代码，用逗号分隔", value=default_universe, height=110)
    if st.button("扫描股票池"):
        tickers = tuple(normalize_ticker(item.strip(), market=market_code) for item in universe_text.replace("\n", ",").split(",") if item.strip())
        with st.spinner("正在抓取股票池数据..."):
            ranking = scan_universe(tickers=tickers, period=period, capital=capital, risk_per_trade=risk_per_trade, stop_loss_pct=stop_loss_pct) if precise_mode else cached_scan_universe(tickers, period, capital, risk_per_trade, stop_loss_pct)
        if ranking.empty:
            st.warning("没有可用结果。")
        else:
            st.dataframe(ranking, width="stretch", hide_index=True)
            chart_df = ranking[ranking["Score"] >= 0].head(10)
            chart = go.Figure(data=[go.Bar(x=chart_df["Name"], y=chart_df["Score"], marker_color="#0e8a62", text=chart_df["Decision"], hovertemplate="%{x}<br>Score=%{y}<br>%{text}<extra></extra>")])
            chart.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.7)", margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(range=[0, 100], title="综合评分"), xaxis_title=None)
            st.plotly_chart(chart, width="stretch")

with backtest_tab:
    st.subheader("规则回放")
    st.write("回放规则: `站上 SMA50` + `SMA20 > SMA50` + `MACD 柱状图 > 0` + `量能高于 20 日均量` + `OBV 强于自身均线` + `价格站上 60 日趋势线`；离场用 `跌破 SMA20 / 趋势线 / 移动止损`。")
    try:
        with st.spinner("正在生成回放结果..."):
            if precise_mode:
                result = analyze_stock(ticker=ticker, period=period, capital=capital, risk_per_trade=risk_per_trade, stop_loss_pct=stop_loss_pct)
                backtest = run_backtest(result.frame, stop_loss_pct=stop_loss_pct / 100)
            else:
                _, backtest = cached_backtest(ticker, period, stop_loss_pct)
        b1, b2, b3, b4, b5 = st.columns(5)
        with b1:
            metric_card("策略 CAGR", pct(backtest.strategy_cagr), f"买入持有 {pct(backtest.buy_hold_cagr)}")
        with b2:
            metric_card("最大回撤", pct(backtest.strategy_max_drawdown), f"买入持有 {pct(backtest.buy_hold_max_drawdown)}")
        with b3:
            metric_card("策略 Sharpe", f"{backtest.strategy_sharpe:.2f}" if pd.notna(backtest.strategy_sharpe) else "N/A", f"买入持有 {backtest.buy_hold_sharpe:.2f}" if pd.notna(backtest.buy_hold_sharpe) else "N/A")
        with b4:
            metric_card("胜率", pct(backtest.win_rate), f"交易次数 {len(backtest.trades)}")
        with b5:
            metric_card("持仓暴露", pct(backtest.exposure), f"成本率 {backtest.trade_cost_rate * 100:.2f}%/次换手")
        st.plotly_chart(build_equity_chart(backtest), width="stretch")
        if not backtest.trades.empty:
            trades = backtest.trades.copy()
            trades["PnL"] = trades["PnL"].map(lambda value: f"{value * 100:.2f}%")
            st.dataframe(trades.tail(15), width="stretch", hide_index=True)
        st.caption("回测仍然只回放价格和量价规则，不包含新闻与财务因子的历史可得性，所以只能验证纪律，不能证明未来一定盈利。")
    except DataUnavailableError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.exception(exc)

with guide_tab:
    st.subheader("建议怎么用")
    st.markdown(textwrap.dedent("""
    1. 先在 `全主板排名`、`主线板块` 或 `股票池扫描` 找出综合分靠前的标的。
    2. 再看 `单票分析`，确认量价、趋势、基本面、估值和消息是否同向。
    3. 最后在 `资金分配` 里按账户资金和总风险上限算仓位，不要先看好再硬凑仓位。
    4. 回测只用来验证规则是否有历史约束力，不能替代未来判断，更不能保证稳赚。
    """))
    st.info("A股可以直接输入 6 位代码，例如 `600519`、`000858`、`600036`。")
    st.warning("没有任何工具可以保证你一定挣钱。这个版本的目标是把分析维度和仓位纪律做完整，而不是制造确定性幻觉。")















