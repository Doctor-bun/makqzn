from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from math import sqrt
import time
from typing import Iterable

import akshare as ak
import numpy as np
import pandas as pd
import requests
import yfinance as yf

TRADING_DAYS = 252
PERIOD_TO_DAYS = {
    "1w": 7,
    "1mo": 31,
    "3mo": 92,
    "6mo": 183,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
}
POSITIVE_NEWS_KEYWORDS = ["回购", "增持", "中标", "预增", "增长", "签约", "订单", "突破", "分红", "获批", "上调", "扩产", "创新高"]
NEGATIVE_NEWS_KEYWORDS = ["减持", "立案", "亏损", "下滑", "处罚", "问询", "诉讼", "风险", "终止", "违约", "暴跌", "跌停", "爆雷", "退市", "质押"]
NOTICE_POSITIVE_KEYWORDS = ["中标", "回购", "增持", "分红", "签订", "订单", "预增", "股权激励", "获批", "扭亏", "增长"]
NOTICE_NEGATIVE_KEYWORDS = ["减持", "问询", "处罚", "立案", "风险", "诉讼", "亏损", "终止", "延期", "质押", "冻结", "退市"]
RESEARCH_POSITIVE_RATINGS = ["买入", "增持", "强烈推荐", "推荐", "跑赢行业", "优于大市"]
RESEARCH_NEGATIVE_RATINGS = ["减持", "卖出", "回避", "弱于大市"]
SOCIAL_SENTIMENT_UPPER = 1.0
SOCIAL_SENTIMENT_LOWER = -1.0
A_MAIN_BOARD_CORE_POOL = (
    "600519", "000858", "600036", "601318", "600276", "600900", "000333", "002594",
    "601899", "600030", "600887", "600809", "601166", "600309", "601088", "603259",
    "603288", "600031", "600570", "601888",
)
FINANCE_QUOTES = [
    {"quote": "价格是你付出的，价值是你得到的。", "author": "沃伦·巴菲特"},
    {"quote": "市场短期像投票机，长期像称重机。", "author": "本杰明·格雷厄姆"},
    {"quote": "最重要的不是你对了多少次，而是你对的时候赚了多少。", "author": "乔治·索罗斯"},
    {"quote": "买股票前，先想清楚如果它跌了 20% 你还愿不愿意拿。", "author": "彼得·林奇"},
    {"quote": "真正的风险不是波动，而是永久性亏损。", "author": "霍华德·马克斯"},
    {"quote": "牛市在悲观中诞生，在怀疑中成长，在乐观中成熟。", "author": "约翰·邓普顿"},
    {"quote": "顺势而为，直到市场证明你错了。", "author": "杰西·利弗莫尔"},
    {"quote": "没有人能够持续正确预测市场，但可以持续控制风险。", "author": "雷·达里奥"},
]
INDICATOR_GLOSSARY = {
    "SMA20": {"meaning": "20 日均线，用来看短线成本区和短期趋势。", "formula": "最近 20 个交易日收盘价的简单平均值"},
    "SMA50": {"meaning": "50 日均线，用来看中期趋势是否健康。", "formula": "最近 50 个交易日收盘价的简单平均值"},
    "SMA200": {"meaning": "200 日均线，常用来区分长期多空。", "formula": "最近 200 个交易日收盘价的简单平均值"},
    "RSI14": {"meaning": "14 日相对强弱指标，衡量涨跌动能是否过热。", "formula": "100 - 100 / (1 + 14 日平均上涨幅度 / 14 日平均下跌幅度)"},
    "MACDHist": {"meaning": "MACD 柱状图，衡量短中期均线差是否继续扩张。", "formula": "MACD 线减去信号线；MACD = EMA12 - EMA26"},
    "ATRPercent": {"meaning": "ATR 占股价比例，反映单日真实波动有多大。", "formula": "14 日 ATR / 当前收盘价"},
    "VolumeRatio": {"meaning": "5 日均量相对 20 日均量的放大倍数。", "formula": "5 日平均成交量 / 20 日平均成交量"},
    "Turnover": {"meaning": "换手率，衡量筹码交换速度。", "formula": "当日成交量 / 流通股本"},
    "OBV": {"meaning": "能量潮指标，用涨跌方向累计成交量，观察资金累积。", "formula": "上涨日加成交量，下跌日减成交量后累计"},
    "Return20": {"meaning": "近 20 个交易日涨跌幅。", "formula": "当前收盘价 / 20 日前收盘价 - 1"},
    "Return60": {"meaning": "近 60 个交易日涨跌幅。", "formula": "当前收盘价 / 60 日前收盘价 - 1"},
    "TrendLine60": {"meaning": "60 日回归趋势线，用来看价格是否仍站在上升通道内。", "formula": "对最近 60 个交易日收盘价做线性回归得到的拟合线"},
}
_CN_NAME_CACHE: dict[str, str] | None = None


@dataclass
class AnalysisResult:
    ticker: str
    stock_name: str
    market: str
    price: float
    as_of: str
    decision: str
    total_score: int
    technical_score: int
    trend_score: int
    momentum_score: int
    volume_score: int
    fundamental_score: int
    news_score: int
    risk_score: int
    position_size: int
    stop_price: float
    risk_budget: float
    summary: str
    annual_volatility: float
    max_drawdown: float
    atr_percent: float
    frame: pd.DataFrame
    decision_details: dict[str, str]
    fundamental_snapshot: dict[str, str]
    company_profile: dict[str, str]
    valuation_items: list[dict[str, str]]
    news_items: list[dict[str, str]]
    support_levels: list[float]
    resistance_levels: list[float]
    quote: dict[str, str]


@dataclass
class BacktestResult:
    frame: pd.DataFrame
    trades: pd.DataFrame
    strategy_cagr: float
    buy_hold_cagr: float
    strategy_max_drawdown: float
    buy_hold_max_drawdown: float
    strategy_sharpe: float
    buy_hold_sharpe: float
    win_rate: float
    exposure: float
    trade_cost_rate: float


@dataclass
class PortfolioPlan:
    plan: pd.DataFrame
    total_allocated: float
    cash_left: float
    total_risk_budget: float
    total_estimated_risk: float
    invested_ratio: float
    notes: list[str]


class DataUnavailableError(RuntimeError):
    pass


def normalize_ticker(ticker: str, market: str = "auto") -> str:
    raw = ticker.strip().upper()
    if not raw:
        return raw
    if market == "cn" and "." not in raw and raw.isdigit() and len(raw) == 6:
        if raw.startswith(("6", "9")):
            return f"{raw}.SS"
        if raw.startswith(("0", "2", "3")):
            return f"{raw}.SZ"
    if market == "auto" and "." not in raw and raw.isdigit() and len(raw) == 6:
        if raw.startswith(("6", "9")):
            return f"{raw}.SS"
        if raw.startswith(("0", "2", "3")):
            return f"{raw}.SZ"
    return raw


def infer_market(ticker: str) -> str:
    normalized = ticker.strip().upper()
    if normalized.endswith((".SS", ".SZ", ".BJ")):
        return "cn"
    if normalized.endswith(".HK"):
        return "hk"
    if normalized.isdigit() and len(normalized) == 6:
        return "cn"
    return "us"


def fetch_price_history(ticker: str, period: str = "2y", as_of_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    normalized = normalize_ticker(ticker)
    market = infer_market(normalized)
    fetchers = [_fetch_akshare_a_history, _fetch_akshare_tx_history, _fetch_yfinance_history, _fetch_stooq_history] if market == "cn" else [_fetch_yfinance_history, _fetch_stooq_history]
    failures: list[str] = []
    as_of_ts = _normalize_as_of_date(as_of_date)
    for fetcher in fetchers:
        try:
            frame = fetcher(ticker=normalized, period=period, as_of_date=as_of_ts)
            if as_of_ts is not None:
                frame = frame.loc[frame.index <= as_of_ts]
            if not frame.empty:
                return frame
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{fetcher.__name__}: {exc}")
    raise DataUnavailableError(f"无法获取 {normalized} 的行情数据。{' ; '.join(failures)}")


def _fetch_akshare_a_history(ticker: str, period: str, as_of_date: pd.Timestamp | None = None) -> pd.DataFrame:
    symbol = ticker.split(".")[0]
    start_ts, _, end_ts = _analysis_window(period, as_of_date)
    end_date = end_ts.strftime("%Y%m%d")
    start_date = start_ts.strftime("%Y%m%d")
    frame = _retry(lambda: ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq"), retries=3, pause=1.0)
    if frame is None or frame.empty:
        raise DataUnavailableError(f"AkShare 没有返回 {ticker} 的历史数据")
    renamed = frame.rename(columns={"日期": "Date", "开盘": "Open", "收盘": "Close", "最高": "High", "最低": "Low", "成交量": "Volume", "成交额": "Amount", "振幅": "Amplitude", "涨跌幅": "PctChange", "涨跌额": "ChangeAmount", "换手率": "Turnover"})
    renamed["Date"] = pd.to_datetime(renamed["Date"])
    return renamed.set_index("Date").sort_index()


def _fetch_akshare_tx_history(ticker: str, period: str, as_of_date: pd.Timestamp | None = None) -> pd.DataFrame:
    symbol = ticker.split(".")[0]
    exchange = "sh" if symbol.startswith(("6", "9")) else "sz"
    start_ts, _, end_ts = _analysis_window(period, as_of_date)
    start_date = start_ts.strftime("%Y%m%d")
    end_date = end_ts.strftime("%Y%m%d")
    frame = _retry(lambda: ak.stock_zh_a_hist_tx(symbol=f"{exchange}{symbol}", start_date=start_date, end_date=end_date, adjust="qfq"), retries=2, pause=0.8)
    if frame is None or frame.empty:
        raise DataUnavailableError(f"腾讯行情没有返回 {ticker} 的历史数据")
    renamed = frame.rename(columns={"date": "Date", "open": "Open", "close": "Close", "high": "High", "low": "Low", "amount": "Volume"})
    renamed["Date"] = pd.to_datetime(renamed["Date"])
    renamed["Volume"] = pd.to_numeric(renamed["Volume"], errors="coerce")
    return renamed.set_index("Date").sort_index()


def _fetch_yfinance_history(ticker: str, period: str, as_of_date: pd.Timestamp | None = None) -> pd.DataFrame:
    if as_of_date is None:
        frame = yf.download(tickers=ticker, period=_to_yfinance_period(period), interval="1d", auto_adjust=True, progress=False, threads=False)
    else:
        start_ts, _, end_ts = _analysis_window(period, as_of_date)
        frame = yf.download(tickers=ticker, start=start_ts.strftime("%Y-%m-%d"), end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), interval="1d", auto_adjust=True, progress=False, threads=False)
    if frame is None or frame.empty:
        raise DataUnavailableError(f"Yahoo 没有返回 {ticker} 的数据")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.rename(columns=lambda name: str(name).title())
    expected = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise DataUnavailableError(f"Yahoo 返回字段不完整: {', '.join(missing)}")
    frame = frame[expected].dropna().copy()
    frame.index = pd.to_datetime(frame.index)
    return frame


def _fetch_stooq_history(ticker: str, period: str, as_of_date: pd.Timestamp | None = None) -> pd.DataFrame:
    symbol = _to_stooq_symbol(ticker)
    response = requests.get(f"https://stooq.com/q/d/l/?s={symbol}&i=d", timeout=20)
    response.raise_for_status()
    payload = response.text.strip()
    if not payload or payload == "No data":
        raise DataUnavailableError(f"Stooq 没有 {ticker} 的数据")
    frame = pd.read_csv(StringIO(payload))
    frame["Date"] = pd.to_datetime(frame["Date"])
    start_ts, _, end_ts = _analysis_window(period, as_of_date)
    frame = frame.loc[(frame["Date"] >= start_ts) & (frame["Date"] <= end_ts)]
    if frame.empty:
        raise DataUnavailableError("Stooq 返回了空区间")
    return frame.set_index("Date").sort_index()


def _to_stooq_symbol(ticker: str) -> str:
    normalized = ticker.strip().lower()
    if normalized.endswith((".ss", ".sz", ".bj")):
        return f"{normalized.split('.')[0]}.cn"
    if "." not in normalized:
        if normalized.isdigit() and len(normalized) == 6:
            return f"{normalized}.cn"
        return f"{normalized}.us"
    return normalized


def _to_yfinance_period(period: str) -> str:
    return {
        "1w": "5d",
        "1mo": "1mo",
        "3mo": "3mo",
        "6mo": "6mo",
        "1y": "1y",
        "2y": "2y",
        "5y": "5y",
    }.get(period, period)


def _normalize_as_of_date(as_of_date: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if as_of_date in (None, ""):
        return None
    ts = pd.Timestamp(as_of_date)
    if pd.isna(ts):
        return None
    return ts.normalize()


def _analysis_window(period: str, as_of_date: str | pd.Timestamp | None = None, padding_days: int = 320) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    end_ts = _normalize_as_of_date(as_of_date) or pd.Timestamp.now().normalize()
    visible_start = end_ts - pd.Timedelta(days=PERIOD_TO_DAYS.get(period, 365))
    fetch_start = visible_start - pd.Timedelta(days=padding_days)
    return fetch_start, visible_start, end_ts


def _slice_analysis_window(frame: pd.DataFrame, period: str, as_of_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame
    _, visible_start, end_ts = _analysis_window(period, as_of_date, padding_days=0)
    sliced = frame.loc[(frame.index >= visible_start) & (frame.index <= end_ts)].copy()
    return sliced if not sliced.empty else frame.loc[frame.index <= end_ts].tail(min(len(frame), PERIOD_TO_DAYS.get(period, 365))).copy()


def compute_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    close, high, low = df["Close"], df["High"], df["Low"]
    df["Return"] = close.pct_change()
    df["Return5"] = close.pct_change(5)
    df["Return20"] = close.pct_change(20)
    df["Return60"] = close.pct_change(60)
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()
    df["High20"] = high.rolling(20).max()
    df["High60"] = high.rolling(60).max()
    df["Low20"] = low.rolling(20).min()
    df["Low60"] = low.rolling(60).min()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACDSignal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACDHist"] = df["MACD"] - df["MACDSignal"]
    delta = close.diff()
    gains, losses = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = (100 - (100 / (1 + rs))).fillna(100)
    previous_close = close.shift(1)
    true_range = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
    df["ATR14"] = true_range.rolling(14).mean()
    df["ATRPercent"] = df["ATR14"] / close
    df["VolumeMA5"] = df["Volume"].rolling(5).mean()
    df["VolumeMA20"] = df["Volume"].rolling(20).mean()
    df["VolumeRatio"] = df["VolumeMA5"] / df["VolumeMA20"]
    df["DailyVolumeRatio"] = df["Volume"] / df["VolumeMA20"]
    if "Turnover" in df.columns:
        df["TurnoverMA5"] = df["Turnover"].rolling(5).mean()
        df["TurnoverMA20"] = df["Turnover"].rolling(20).mean()
    else:
        df["Turnover"] = np.nan
        df["TurnoverMA5"] = np.nan
        df["TurnoverMA20"] = np.nan
    obv_direction = np.sign(close.diff().fillna(0))
    df["OBV"] = (obv_direction * df["Volume"]).fillna(0).cumsum()
    df["OBVEMA20"] = df["OBV"].ewm(span=20, adjust=False).mean()
    df["Breakout60"] = close >= df["High60"].shift(1)
    df["TrendLine60"] = _rolling_linear_regression(close, 60)
    price_std = close.rolling(60).std()
    df["TrendUpper60"] = df["TrendLine60"] + price_std
    df["TrendLower60"] = df["TrendLine60"] - price_std
    df["TrendSlope60"] = _rolling_slope(close, 60)
    df["PivotHigh"] = np.where(high == high.rolling(5, center=True).max(), high, np.nan)
    df["PivotLow"] = np.where(low == low.rolling(5, center=True).min(), low, np.nan)
    df["AnnualVol20"] = df["Return"].rolling(20).std() * sqrt(TRADING_DAYS)
    df["RollingHigh252"] = close.rolling(252).max()
    wealth = (1 + df["Return"].fillna(0)).cumprod()
    peak = wealth.cummax()
    df["Drawdown"] = wealth / peak - 1
    return df

def fetch_news_sentiment(ticker: str, stock_name: str = "", limit: int = 80, period: str = "2y", as_of_date: str | pd.Timestamp | None = None) -> tuple[int, str, list[dict[str, str]]]:
    if infer_market(ticker) != "cn":
        return 50, "当前版本的消息面主要针对 A 股接入。", []
    symbol = ticker.split(".")[0]
    raw_items: list[dict[str, object]] = []
    parts: list[str] = []
    as_of_ts = _normalize_as_of_date(as_of_date)

    try:
        news_df = _retry(lambda: ak.stock_news_em(symbol=symbol), retries=3, pause=1.1)
        if news_df is not None and not news_df.empty:
            for _, row in news_df.drop_duplicates(subset=["新闻标题"]).head(limit).iterrows():
                headline = str(row.get("新闻标题", ""))
                content = str(row.get("新闻内容", ""))
                text = f"{headline} {content}"
                item_score = sum(keyword in text for keyword in POSITIVE_NEWS_KEYWORDS) - sum(keyword in text for keyword in NEGATIVE_NEWS_KEYWORDS)
                raw_items.append({"发布时间": str(row.get("发布时间", "")), "标题": headline, "来源": str(row.get("文章来源", "东财新闻")), "情绪": "偏多" if item_score > 0 else "偏空" if item_score < 0 else "中性", "链接": str(row.get("新闻链接", "")), "类型": "公司新闻", "score": float(item_score)})
            parts.append(f"公司新闻 {min(len(news_df), limit)} 条")
    except Exception as exc:  # noqa: BLE001
        parts.append(f"公司新闻抓取失败: {exc}")

    notice_items = _fetch_company_notices(symbol, limit=max(limit, 50), lookback_days=30)
    for notice in notice_items:
        text = f"{notice['标题']} {notice['类型']}"
        item_score = sum(keyword in text for keyword in NOTICE_POSITIVE_KEYWORDS) - sum(keyword in text for keyword in NOTICE_NEGATIVE_KEYWORDS)
        raw_items.append({"发布时间": notice["发布时间"], "标题": notice["标题"], "来源": notice["来源"], "情绪": "偏多" if item_score > 0 else "偏空" if item_score < 0 else "中性", "链接": notice["链接"], "类型": "公告", "score": float(item_score * 1.5)})
    if notice_items:
        parts.append(f"公告 {len(notice_items)} 条")

    research_items = _fetch_research_reports(symbol, limit=max(12, limit // 2))
    for report in research_items:
        text = f"{report['标题']} {report['评级']} {report['机构']}"
        item_score = sum(keyword in text for keyword in RESEARCH_POSITIVE_RATINGS) - sum(keyword in text for keyword in RESEARCH_NEGATIVE_RATINGS)
        raw_items.append({"发布时间": report["发布时间"], "标题": report["标题"], "来源": report["来源"], "情绪": "偏多" if item_score > 0 else "偏空" if item_score < 0 else "中性", "链接": report["链接"], "类型": "券商研报", "score": float(item_score * 1.2)})
    if research_items:
        parts.append(f"券商研报 {len(research_items)} 条")

    social_item = _fetch_social_sentiment(stock_name)
    if social_item is not None:
        raw_items.append({
            "发布时间": social_item["发布时间"],
            "标题": social_item["标题"],
            "来源": social_item["来源"],
            "情绪": social_item["情绪"],
            "链接": "",
            "类型": "社交舆情",
            "score": float(social_item["score"]),
        })
        parts.append("微博舆情 1 条")

    cls_items = _fetch_cls_company_telegrams(stock_name, limit=max(10, limit // 5))
    for cls_item in cls_items:
        raw_items.append({
            "发布时间": cls_item["发布时间"],
            "标题": cls_item["标题"],
            "来源": cls_item["来源"],
            "情绪": cls_item["情绪"],
            "链接": cls_item["链接"],
            "类型": "财联社电报",
            "score": float(cls_item["score"] * 0.9),
        })
    if cls_items:
        parts.append(f"财联社电报 {len(cls_items)} 条")

    related_market_news = _fetch_related_market_news(stock_name, limit=12)
    for news_item in related_market_news:
        raw_items.append({
            "发布时间": news_item["发布时间"],
            "标题": news_item["标题"],
            "来源": news_item["来源"],
            "情绪": news_item["情绪"],
            "链接": news_item["链接"],
            "类型": "行业/市场新闻",
            "score": float(news_item["score"] * 0.8),
        })
    if related_market_news:
        parts.append(f"行业/市场新闻 {len(related_market_news)} 条")

    if as_of_ts is not None:
        start_ts = as_of_ts - pd.Timedelta(days=PERIOD_TO_DAYS.get(period, 365))
        filtered_items = []
        for item in raw_items:
            published_at = _parse_item_datetime(item.get("发布时间", ""))
            if published_at is pd.NaT or pd.isna(published_at):
                continue
            if start_ts <= published_at.normalize() <= as_of_ts:
                filtered_items.append(item)
        raw_items = filtered_items

    raw_items = sorted(raw_items, key=lambda item: _safe_sort_datetime(item.get("发布时间", "")), reverse=True)
    if not raw_items:
        return 50, "该回看区间内没有抓到足够的公司消息或公告。" if as_of_ts is not None else "最近没有抓到足够的公司消息或公告。", []

    raw_score = sum(float(item.get("score", 0.0)) for item in raw_items)
    items = [{key: value for key, value in item.items() if key != "score"} for item in raw_items]

    normalized = int(max(0, min(100, 50 + raw_score * 5)))
    if normalized >= 65:
        summary = "近期公司新闻、公告、研报和快讯整体偏多，催化较集中。"
    elif normalized <= 40:
        summary = "近期公告、新闻、研报或快讯里的风险词偏多，做交易前要更重视减仓和止损。"
    else:
        summary = "近期消息面偏中性，更多还是要靠价格和量能确认。"
    detail = "；".join(parts) if parts else ""
    return normalized, f"{summary}{detail and '；' + detail}", items[:limit]



def _fetch_company_notices(symbol: str, limit: int = 30, lookback_days: int = 14) -> list[dict[str, str]]:
    notices: list[dict[str, str]] = []
    url = "https://np-anotice-stock.eastmoney.com/api/security/ann"
    today = pd.Timestamp.now().normalize()
    begin_time = (today - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end_time = today.strftime("%Y-%m-%d")
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}
    for page_index in range(1, 8):
        params = {
            "sr": "-1",
            "page_size": "100",
            "page_index": str(page_index),
            "ann_type": "A",
            "client_source": "web",
            "f_node": "0",
            "s_node": "0",
            "begin_time": begin_time,
            "end_time": end_time,
        }
        try:
            response = requests.get(url, params=params, headers=headers, timeout=20)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            continue
        for item in payload.get("data", {}).get("list", []):
            codes = item.get("codes", []) or []
            matched = None
            for code_item in codes:
                if str(code_item.get("stock_code", "")) == symbol:
                    matched = code_item
                    break
            if not matched:
                continue
            columns = item.get("columns", []) or []
            title = str(item.get("title", ""))
            notices.append({
                "发布时间": str(item.get("notice_date", ""))[:10],
                "标题": title,
                "来源": "东财公告",
                "类型": str(columns[0].get("column_name", "公司公告")) if columns else "公司公告",
                "链接": f"https://data.eastmoney.com/notices/detail/{symbol}/{item.get('art_code','')}.html",
            })
            if len(notices) >= limit:
                return notices[:limit]
        if not payload.get("data", {}).get("list"):
            break
    return notices[:limit]


def _fetch_research_reports(symbol: str, limit: int = 15) -> list[dict[str, str]]:
    try:
        df = _retry(lambda: ak.stock_research_report_em(symbol=symbol), retries=2, pause=1.0)
    except Exception:
        return []
    if df is None or df.empty:
        return []
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, row in df.head(limit).iterrows():
        title = str(row.get("报告名称", "")).strip()
        report_date = str(row.get("日期", "")).strip()
        dedupe_key = f"{report_date}|{title}"
        if not title or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        rating = str(row.get("东财评级", "")).strip()
        broker = str(row.get("机构", "")).strip()
        items.append({
            "发布时间": report_date,
            "标题": title,
            "来源": broker or "东方财富研报",
            "评级": rating or "未评级",
            "机构": broker or "未知机构",
            "链接": str(row.get("报告PDF链接", "")).strip(),
        })
    return items[:limit]


def _fetch_social_sentiment(stock_name: str) -> dict[str, str | float] | None:
    if not stock_name:
        return None
    try:
        df = _retry(lambda: ak.stock_js_weibo_report(time_period="CNHOUR24"), retries=2, pause=0.8)
    except Exception:
        return None
    if df is None or df.empty or "name" not in df.columns:
        return None
    matched = df.loc[df["name"].astype(str) == stock_name]
    if matched.empty:
        return None
    rate = _safe_float(matched.iloc[0].get("rate"))
    if pd.isna(rate):
        return None
    score = 2 if rate >= SOCIAL_SENTIMENT_UPPER else -2 if rate <= SOCIAL_SENTIMENT_LOWER else 0
    return {
        "发布时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "标题": f"{stock_name} 微博舆情热度 {rate:.2f}",
        "来源": "金十微博舆情",
        "情绪": "偏多" if score > 0 else "偏空" if score < 0 else "中性",
        "score": score,
    }


def _fetch_related_market_news(stock_name: str, limit: int = 8) -> list[dict[str, str | int]]:
    if not stock_name or len(stock_name) < 2:
        return []
    rows: list[dict[str, str | int]] = []
    seen: set[str] = set()
    try:
        df = _retry(lambda: ak.stock_news_main_cx(), retries=2, pause=1.0)
        if df is not None and not df.empty:
            for _, row in df.head(180).iterrows():
                title = str(row.get("summary", ""))
                if stock_name not in title:
                    continue
                dedupe_key = f"cx|{title}"
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                item_score = sum(keyword in title for keyword in POSITIVE_NEWS_KEYWORDS) - sum(keyword in title for keyword in NEGATIVE_NEWS_KEYWORDS)
                rows.append({
                    "发布时间": pd.Timestamp.now().strftime("%Y-%m-%d"),
                    "标题": title,
                    "来源": "财新主线新闻",
                    "情绪": "偏多" if item_score > 0 else "偏空" if item_score < 0 else "中性",
                    "链接": str(row.get("url", "")),
                    "score": item_score,
                })
                if len(rows) >= limit:
                    return rows[:limit]
    except Exception:
        pass
    for item in _fetch_cls_company_telegrams(stock_name, limit=limit):
        dedupe_key = f"cls|{item['标题']}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        rows.append(item)
        if len(rows) >= limit:
            break
    return rows[:limit]


def _fetch_cls_company_telegrams(stock_name: str, limit: int = 10) -> list[dict[str, str | int]]:
    if not stock_name or len(stock_name) < 2:
        return []
    try:
        df = _retry(lambda: ak.stock_info_global_cls(), retries=2, pause=0.8)
    except Exception:
        return []
    if df is None or df.empty:
        return []
    rows: list[dict[str, str | int]] = []
    seen: set[str] = set()
    for _, row in df.head(220).iterrows():
        title = str(row.get("标题", "")).strip()
        content = str(row.get("内容", "")).strip()
        text = f"{title} {content}"
        if stock_name not in text:
            continue
        dedupe_key = f"{row.get('发布日期', '')}|{title}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        item_score = sum(keyword in text for keyword in POSITIVE_NEWS_KEYWORDS) - sum(keyword in text for keyword in NEGATIVE_NEWS_KEYWORDS)
        rows.append({
            "发布时间": f"{row.get('发布日期', '')} {row.get('发布时间', '')}".strip(),
            "标题": title or content[:36],
            "来源": "财联社电报",
            "情绪": "偏多" if item_score > 0 else "偏空" if item_score < 0 else "中性",
            "链接": "",
            "score": item_score,
        })
        if len(rows) >= limit:
            break
    return rows[:limit]


def _parse_item_datetime(value: object) -> pd.Timestamp:
    if value in (None, "", "nan", "None"):
        return pd.NaT
    try:
        return pd.to_datetime(value, errors="coerce")
    except Exception:
        return pd.NaT


def _safe_sort_datetime(value: object) -> pd.Timestamp:
    parsed = _parse_item_datetime(value)
    return parsed if pd.notna(parsed) else pd.Timestamp("1900-01-01")
def fetch_fundamental_snapshot(ticker: str, as_of_date: str | pd.Timestamp | None = None) -> tuple[int, dict[str, str], str, dict[str, float]]:
    if infer_market(ticker) != "cn":
        return 50, {}, "当前版本的财务摘要主要针对 A 股接入。", {}
    symbol = ticker.split(".")[0]
    details: dict[str, str] = {}
    raw: dict[str, float] = {}
    score = 50
    summary_parts: list[str] = []
    as_of_ts = _normalize_as_of_date(as_of_date)
    try:
        ths_df = _retry(lambda: ak.stock_financial_abstract_ths(symbol=symbol, indicator="按报告期"), retries=2, pause=0.8)
        if ths_df is not None and not ths_df.empty:
            report_df = ths_df.copy()
            report_df["报告期"] = pd.to_datetime(report_df["报告期"], errors="coerce")
            if as_of_ts is not None:
                report_df = report_df.loc[report_df["报告期"] <= as_of_ts]
            if report_df.empty:
                raise DataUnavailableError("该日期之前没有可用财报摘要")
            latest = report_df.sort_values("报告期").iloc[-1]
            if pd.notna(latest.get("报告期")):
                details["财报口径"] = str(pd.Timestamp(latest["报告期"]).date())
                summary_parts.append(f"财报口径 {pd.Timestamp(latest['报告期']).date()}")
            raw["roe"] = _parse_percent(latest.get("净资产收益率"))
            raw["gross_margin"] = _parse_percent(latest.get("销售毛利率"))
            raw["debt_ratio"] = _parse_percent(latest.get("资产负债率"))
            raw["cashflow_per_share"] = _safe_float(latest.get("每股经营现金流"))
            raw["revenue"] = _safe_float(latest.get("营业总收入"))
            raw["profit"] = _safe_float(latest.get("净利润"))
            raw["eps"] = _safe_float(latest.get("基本每股收益"))
            raw["bps"] = _safe_float(latest.get("每股净资产"))
            raw["revenue_growth"] = _parse_percent(latest.get("营业总收入同比增长率"))
            raw["profit_growth"] = _parse_percent(latest.get("净利润同比增长率"))
            if pd.notna(raw.get("roe", np.nan)):
                details["ROE"] = f"{raw['roe']:.2f}%"
                if raw["roe"] >= 15:
                    score += 20
                    summary_parts.append("ROE 较高")
                elif raw["roe"] >= 10:
                    score += 10
                    summary_parts.append("ROE 尚可")
                else:
                    score -= 10
                    summary_parts.append("ROE 偏弱")
            if pd.notna(raw.get("gross_margin", np.nan)):
                details["毛利率"] = f"{raw['gross_margin']:.2f}%"
                if raw["gross_margin"] >= 25:
                    score += 15
                    summary_parts.append("毛利率健康")
                elif raw["gross_margin"] < 10:
                    score -= 10
                    summary_parts.append("毛利率偏低")
            if pd.notna(raw.get("debt_ratio", np.nan)):
                details["资产负债率"] = f"{raw['debt_ratio']:.2f}%"
                score += 10 if raw["debt_ratio"] <= 60 else -10
                if raw["debt_ratio"] > 60:
                    summary_parts.append("负债率偏高")
            if pd.notna(raw.get("cashflow_per_share", np.nan)):
                details["每股经营现金流"] = f"{raw['cashflow_per_share']:.2f}"
                if raw["cashflow_per_share"] > 0:
                    score += 10
                    summary_parts.append("经营现金流为正")
                else:
                    score -= 10
                    summary_parts.append("经营现金流承压")
            if pd.notna(raw.get("revenue", np.nan)):
                details["营业总收入"] = _format_large_number(raw["revenue"])
            if pd.notna(raw.get("profit", np.nan)):
                details["归母净利润"] = _format_large_number(raw["profit"])
            if pd.notna(raw.get("eps", np.nan)):
                details["EPS"] = f"{raw['eps']:.2f}"
            if pd.notna(raw.get("bps", np.nan)):
                details["每股净资产"] = f"{raw['bps']:.2f}"
            if raw.get("revenue_growth") is not None:
                details["营收同比"] = f"{raw['revenue_growth']:.2f}%"
                if raw["revenue_growth"] >= 10:
                    score += 8
                    summary_parts.append("营收同比增长")
                elif raw["revenue_growth"] < 0:
                    score -= 8
                    summary_parts.append("营收同比下滑")
            if raw.get("profit_growth") is not None:
                details["净利同比"] = f"{raw['profit_growth']:.2f}%"
                if raw["profit_growth"] >= 10:
                    score += 12
                    summary_parts.append("净利同比增长")
                elif raw["profit_growth"] < 0:
                    score -= 12
                    summary_parts.append("净利同比下滑")
    except Exception:
        summary_parts.append("财务主数据源暂时不可用，当前只展示可回退到的指标。")
    score = int(max(0, min(100, score)))
    summary = "；".join(summary_parts) if summary_parts else "财务数据没有给出明显偏向。"
    normalized_raw = {key: float(value) for key, value in raw.items() if value is not None and pd.notna(value)}
    return score, details, summary, normalized_raw


def fetch_company_profile(ticker: str, latest_price: float) -> tuple[str, dict[str, str], float]:
    market = infer_market(ticker)
    symbol = ticker.split(".")[0]
    stock_name = _get_cn_name_map().get(symbol, ticker.upper()) if market == "cn" else ticker.upper()
    profile: dict[str, str] = {}
    shares_outstanding = np.nan
    if market != "cn":
        return stock_name, profile, shares_outstanding
    try:
        basic_df = _retry(lambda: ak.stock_individual_basic_info_xq(symbol=_to_xq_symbol(ticker)))
        if basic_df is not None and not basic_df.empty:
            basic_map = basic_df.set_index("item")["value"]
            stock_name = str(basic_map.get("org_short_name_cn", stock_name))
            shares_candidate = _safe_float(basic_map.get("reg_asset"))
            if pd.notna(shares_candidate):
                shares_outstanding = shares_candidate
            for key, label in {"actual_controller": "实控人", "legal_representative": "法人代表", "staff_num": "员工人数", "org_website": "官方网站", "main_operation_business": "主营业务", "provincial_name": "所在省份"}.items():
                value = basic_map.get(key)
                if value not in (None, "None", "nan"):
                    profile[label] = str(value)
    except Exception:
        pass
    try:
        cninfo_df = _retry(lambda: ak.stock_profile_cninfo(symbol=symbol))
        if cninfo_df is not None and not cninfo_df.empty:
            row = cninfo_df.iloc[0]
            stock_name = str(row.get("A股简称", stock_name))
            mappings = {"公司名称": "公司名称", "所属行业": "所属行业", "所属市场": "所属市场", "上市日期": "上市日期", "主营业务": "主营业务", "办公地址": "办公地址", "官方网站": "官方网站", "联系电话": "联系电话"}
            for source, target in mappings.items():
                value = row.get(source)
                if pd.notna(value) and str(value).strip() and target not in profile:
                    profile[target] = str(value)
            intro = str(row.get("机构简介", "")).strip()
            if intro:
                profile["公司简介"] = intro[:140] + ("..." if len(intro) > 140 else "")
    except Exception:
        pass
    if pd.isna(shares_outstanding):
        try:
            info_df = _retry(lambda: ak.stock_individual_info_em(symbol=symbol))
            if info_df is not None and not info_df.empty:
                info_map = info_df.set_index("item")["value"]
                stock_name = str(info_map.get("股票简称", stock_name))
                shares_candidate = _safe_float(info_map.get("总股本"))
                if pd.notna(shares_candidate):
                    shares_outstanding = shares_candidate
        except Exception:
            pass
    if pd.notna(shares_outstanding):
        profile.setdefault("总股本", f"{shares_outstanding / 100000000:.2f} 亿股")
        if latest_price > 0:
            profile.setdefault("估算总市值", _format_large_number(latest_price * shares_outstanding))
    return stock_name, profile, shares_outstanding


def build_valuation_snapshot(ticker: str, price: float, shares_outstanding: float, fundamentals: dict[str, float], as_of_date: str | pd.Timestamp | None = None) -> tuple[list[dict[str, str]], str]:
    items: list[dict[str, str]] = []
    summary_parts: list[str] = []
    symbol = ticker.split(".")[0]
    market_cap = price * shares_outstanding if price > 0 and pd.notna(shares_outstanding) else np.nan
    eps = fundamentals.get("eps", np.nan)
    bps = fundamentals.get("bps", np.nan)
    revenue = fundamentals.get("revenue", np.nan)
    profit = fundamentals.get("profit", np.nan)
    profit_growth = fundamentals.get("profit_growth", np.nan)
    revenue_growth = fundamentals.get("revenue_growth", np.nan)
    gross_margin = fundamentals.get("gross_margin", np.nan)
    cashflow = fundamentals.get("cashflow", np.nan)
    if pd.isna(cashflow):
        cashflow_per_share = fundamentals.get("cashflow_per_share", np.nan)
        if pd.notna(cashflow_per_share) and pd.notna(shares_outstanding):
            cashflow = cashflow_per_share * shares_outstanding
    valuation_history = _fetch_valuation_history(symbol)
    as_of_ts = _normalize_as_of_date(as_of_date)
    if as_of_ts is not None and not valuation_history.empty and "数据日期" in valuation_history.columns:
        valuation_history = valuation_history.loc[valuation_history["数据日期"] <= as_of_ts].copy()
    fair_prices: list[float] = []
    buy_prices: list[float] = []
    if pd.notna(eps) and eps > 0:
        pe = price / eps
        hist_stats = _historical_valuation_stats(valuation_history, "PE(TTM)")
        fair_price = _price_from_multiple(price, pe, hist_stats.get("median", np.nan))
        buy_price = _price_from_multiple(price, pe, hist_stats.get("q25", np.nan))
        judgement = _value_judgement(price, fair_price, buy_price, fallback=_bucket_value(pe, 15, 30, "偏便宜", "大致合理", "偏贵"))
        items.append(_valuation_item("PE 市盈率", pe, judgement, "股价 / 每股收益，适合盈利稳定公司。", fair_price, buy_price, price, hist_stats, "历史 PE 中位数回归视角。"))
        summary_parts.append(f"PE {pe:.2f}，{judgement}")
        if pd.notna(fair_price):
            fair_prices.append(fair_price)
        if pd.notna(buy_price):
            buy_prices.append(buy_price)
    if pd.notna(bps) and bps > 0:
        pb = price / bps
        hist_stats = _historical_valuation_stats(valuation_history, "市净率")
        fair_price = _price_from_multiple(price, pb, hist_stats.get("median", np.nan))
        buy_price = _price_from_multiple(price, pb, hist_stats.get("q25", np.nan))
        judgement = _value_judgement(price, fair_price, buy_price, fallback=_bucket_value(pb, 1.5, 4.0, "偏便宜", "大致合理", "偏贵"))
        items.append(_valuation_item("PB 市净率", pb, judgement, "股价 / 每股净资产，适合银行、周期、重资产行业。", fair_price, buy_price, price, hist_stats, "历史 PB 分位用于判断贵贱。"))
        summary_parts.append(f"PB {pb:.2f}，{judgement}")
        if pd.notna(fair_price):
            fair_prices.append(fair_price)
        if pd.notna(buy_price):
            buy_prices.append(buy_price)
    if pd.notna(market_cap) and pd.notna(revenue) and revenue > 0:
        ps = market_cap / revenue
        hist_stats = _historical_valuation_stats(valuation_history, "市销率")
        fair_price = _price_from_multiple(price, ps, hist_stats.get("median", np.nan))
        buy_price = _price_from_multiple(price, ps, hist_stats.get("q25", np.nan))
        judgement = _value_judgement(price, fair_price, buy_price, fallback=_bucket_value(ps, 2.0, 6.0, "偏便宜", "大致合理", "偏贵"))
        items.append(_valuation_item("PS 市销率", ps, judgement, "总市值 / 营业收入，适合利润波动较大的成长公司。", fair_price, buy_price, price, hist_stats, "历史 PS 分位和营收质量一起看。"))
        summary_parts.append(f"PS {ps:.2f}，{judgement}")
        if pd.notna(fair_price):
            fair_prices.append(fair_price)
        if pd.notna(buy_price):
            buy_prices.append(buy_price)
    if pd.notna(market_cap) and pd.notna(profit) and market_cap > 0:
        earnings_yield = profit / market_cap
        items.append({"方法": "盈利收益率", "当前值": f"{earnings_yield * 100:.2f}%", "判断": "收益率较高，估值更有吸引力" if earnings_yield >= 0.06 else "收益率一般" if earnings_yield >= 0.03 else "收益率偏低，估值压力较大", "合理价": "N/A", "安全买点": "N/A", "潜在空间": "N/A", "说明": "归母净利润 / 总市值，是 PE 的倒数视角。"})
        summary_parts.append(f"盈利收益率 {earnings_yield * 100:.2f}%")
    if pd.notna(market_cap) and pd.notna(cashflow) and cashflow > 0:
        pcf = market_cap / cashflow
        hist_stats = _historical_valuation_stats(valuation_history, "市现率")
        fair_price = _price_from_multiple(price, pcf, hist_stats.get("median", np.nan))
        buy_price = _price_from_multiple(price, pcf, hist_stats.get("q25", np.nan))
        judgement = _value_judgement(price, fair_price, buy_price, fallback=_bucket_value(pcf, 8, 20, "现金流估值偏便宜", "现金流估值大致合理", "现金流估值偏贵"))
        items.append(_valuation_item("PCF 市现率", pcf, judgement, "总市值 / 经营现金流，更适合现金流稳定的公司。", fair_price, buy_price, price, hist_stats, "历史市现率分位可辅助判断现金流定价。"))
        summary_parts.append(f"PCF {pcf:.2f}，{judgement}")
        if pd.notna(fair_price):
            fair_prices.append(fair_price)
        if pd.notna(buy_price):
            buy_prices.append(buy_price)
    if pd.notna(eps) and eps > 0 and pd.notna(profit_growth) and profit_growth > 0:
        peg = (price / eps) / profit_growth
        hist_stats = _historical_valuation_stats(valuation_history, "PEG值")
        fair_price = _price_from_multiple(price, peg, hist_stats.get("median", np.nan))
        buy_price = _price_from_multiple(price, peg, hist_stats.get("q25", np.nan))
        judgement = _value_judgement(price, fair_price, buy_price, fallback=_bucket_value(peg, 1.0, 2.0, "成长匹配较好", "成长与估值基本匹配", "估值透支增长"))
        items.append(_valuation_item("PEG", peg, judgement, "PE / 净利增速，用来看增长是否足以支撑估值。", fair_price, buy_price, price, hist_stats, "PEG 越低，增长对估值越有支撑。"))
        summary_parts.append(f"PEG {peg:.2f}，{judgement}")
        if pd.notna(fair_price):
            fair_prices.append(fair_price)
        if pd.notna(buy_price):
            buy_prices.append(buy_price)
    if pd.notna(profit) and pd.notna(shares_outstanding) and shares_outstanding > 0:
        owner_earnings = np.nan
        positives = [value for value in [profit, cashflow] if pd.notna(value) and value > 0]
        if positives:
            owner_earnings = min(positives) if len(positives) > 1 else positives[0]
        if pd.notna(owner_earnings) and owner_earnings > 0:
            fair_price = (owner_earnings / 0.08) / shares_outstanding
            buy_price = fair_price * 0.7
            judgement = _value_judgement(price, fair_price, buy_price)
            items.append(_valuation_item("巴菲特收益法", owner_earnings, judgement, "用保守所有者收益按 8% 要求回报折算合理市值。", fair_price, buy_price, price, {}, "所有者收益用净利润与经营现金流的保守值近似。", value_formatter=_format_large_number))
            summary_parts.append(f"巴菲特收益法合理价约 {fair_price:.2f}")
            fair_prices.append(fair_price)
            buy_prices.append(buy_price)
    roe = fundamentals.get("roe", np.nan)
    if pd.notna(roe) and roe > 0 and pd.notna(bps) and bps > 0:
        fair_pb = min(max(roe / 10.0, 0.8), 4.5)
        fair_price = bps * fair_pb
        buy_price = fair_price * 0.8
        judgement = _value_judgement(price, fair_price, buy_price)
        items.append(_valuation_item("ROE-PB 锚定", fair_pb, judgement, "用 ROE 对应合理 PB，再用每股净资产折算股价。", fair_price, buy_price, price, {}, "当 ROE 稳定时，这个方法适合看长期合理区间。"))
        summary_parts.append(f"ROE-PB 合理价约 {fair_price:.2f}")
        fair_prices.append(fair_price)
        buy_prices.append(buy_price)
    if pd.notna(eps) and eps > 0 and pd.notna(profit_growth):
        growth = max(0.0, min(profit_growth, 20.0))
        fair_price = eps * (8.5 + 2 * growth)
        buy_price = fair_price * 0.7
        judgement = _value_judgement(price, fair_price, buy_price)
        items.append(_valuation_item("格雷厄姆公式", eps, judgement, "内在价值 = EPS x (8.5 + 2g)，适合盈利相对稳定公司。", fair_price, buy_price, price, {}, f"这里把 g 取为净利增速，并上限截断到 20%。", value_formatter=lambda value: f"EPS {value:.2f}"))
        summary_parts.append(f"格雷厄姆合理价约 {fair_price:.2f}")
        fair_prices.append(fair_price)
        buy_prices.append(buy_price)
    if pd.notna(eps) and eps > 0 and pd.notna(profit_growth) and profit_growth > 0:
        lynch_pe = min(max(profit_growth, 8.0), 25.0)
        fair_price = eps * lynch_pe
        buy_price = fair_price * 0.78
        judgement = _value_judgement(price, fair_price, buy_price)
        items.append(_valuation_item("彼得林奇估值", lynch_pe, judgement, "把合理 PE 近似看成净利增速，用增长匹配估值。", fair_price, buy_price, price, {}, "更适合盈利持续增长、商业模式清晰的公司。"))
        summary_parts.append(f"彼得林奇合理价约 {fair_price:.2f}")
        fair_prices.append(fair_price)
        buy_prices.append(buy_price)
    if pd.notna(revenue) and revenue > 0 and pd.notna(shares_outstanding) and shares_outstanding > 0:
        target_ps = 1.0
        if pd.notna(gross_margin):
            target_ps = 0.9 if gross_margin < 15 else 1.5 if gross_margin < 25 else 2.5 if gross_margin < 40 else 3.5
        if pd.notna(revenue_growth):
            target_ps *= 1.15 if revenue_growth >= 20 else 1.05 if revenue_growth >= 10 else 0.9 if revenue_growth < 0 else 1.0
        target_ps = float(min(max(target_ps, 0.8), 6.0))
        fair_price = (revenue * target_ps) / shares_outstanding
        buy_price = fair_price * 0.75
        judgement = _value_judgement(price, fair_price, buy_price)
        items.append(_valuation_item("业务市值法", target_ps, judgement, "按营收规模、毛利率和增速匹配目标市销率，再折算合理市值。", fair_price, buy_price, price, {}, "更适合成长行业，用来看业务扩张后对应的合理市值。"))
        summary_parts.append(f"业务市值法合理价约 {fair_price:.2f}")
        fair_prices.append(fair_price)
        buy_prices.append(buy_price)
    if pd.notna(cashflow) and cashflow > 0 and pd.notna(shares_outstanding) and shares_outstanding > 0:
        growth_rate = max(0.03, min(max((profit_growth if pd.notna(profit_growth) else 0.0), (revenue_growth if pd.notna(revenue_growth) else 0.0), 5.0) / 100, 0.12))
        discount_rate = 0.11
        terminal_growth = 0.03
        discounted = sum((cashflow * ((1 + growth_rate) ** year)) / ((1 + discount_rate) ** year) for year in range(1, 6))
        terminal_cashflow = cashflow * ((1 + growth_rate) ** 5) * (1 + terminal_growth)
        terminal_value = terminal_cashflow / max(discount_rate - terminal_growth, 0.02)
        fair_price = (discounted + (terminal_value / ((1 + discount_rate) ** 5))) / shares_outstanding
        buy_price = fair_price * 0.72
        judgement = _value_judgement(price, fair_price, buy_price)
        items.append(_valuation_item("简化 DCF", growth_rate * 100, judgement, "用经营现金流做 5 年增长 + 永续增长折现。", fair_price, buy_price, price, {}, f"折现率 11%，永续增长率 3%，增长假设 {growth_rate * 100:.1f}%。", value_formatter=lambda value: f"{value:.1f}%"))
        summary_parts.append(f"DCF 合理价约 {fair_price:.2f}")
        fair_prices.append(fair_price)
        buy_prices.append(buy_price)
    valid_fair = [value for value in fair_prices if pd.notna(value) and value > 0]
    valid_buy = [value for value in buy_prices if pd.notna(value) and value > 0]
    if valid_fair:
        value_low = float(np.nanpercentile(valid_fair, 25))
        value_mid = float(np.nanmedian(valid_fair))
        value_high = float(np.nanpercentile(valid_fair, 75))
        safe_buy = float(np.nanmedian(valid_buy)) if valid_buy else value_low * 0.9
        judgement = _value_judgement(price, value_mid, safe_buy)
        items.insert(0, {
            "方法": "综合价值区间",
            "当前值": f"{price:.2f}",
            "判断": judgement,
            "合理价": f"{value_low:.2f} - {value_high:.2f}",
            "安全买点": f"{safe_buy:.2f}",
            "潜在空间": _format_percent_delta(value_mid, price),
            "说明": "综合历史估值分位、现金流估值、巴菲特收益法、格雷厄姆公式、彼得林奇估值、ROE-PB 和业务市值法的中枢结果。",
        })
        summary_parts.insert(0, f"综合合理价值区间约 {value_low:.2f}-{value_high:.2f}，安全买点约 {safe_buy:.2f}")
    if not items:
        items.append({"方法": "估值缺失", "当前值": "N/A", "判断": "暂无法判断", "合理价": "N/A", "安全买点": "N/A", "潜在空间": "N/A", "说明": "缺少 EPS、净资产或营收数据时，估值判断会失真。"})
        summary_parts.append("当前可用估值数据不足")
    return items, "；".join(summary_parts)


def analyze_stock(ticker: str, period: str = "2y", capital: float = 100000.0, risk_per_trade: float = 1.0, stop_loss_pct: float = 8.0, as_of_date: str | pd.Timestamp | None = None) -> AnalysisResult:
    normalized = normalize_ticker(ticker)
    market = infer_market(normalized)
    history = fetch_price_history(normalized, period=period, as_of_date=as_of_date)
    full_df = compute_indicators(history)
    df = _slice_analysis_window(full_df, period=period, as_of_date=as_of_date)
    if df.empty:
        raise DataUnavailableError(f"{normalized} 在指定日期之前没有可用交易数据。")
    latest = df.iloc[-1]
    trend_score = _score_trend(latest)
    momentum_score = _score_momentum(latest)
    technical_score = int(round((trend_score * 0.6) + (momentum_score * 0.4)))
    volume_score, volume_summary = _score_volume(latest)
    fundamental_score, fundamental_snapshot, fundamental_summary, fundamentals_raw = fetch_fundamental_snapshot(normalized, as_of_date=latest.name)
    stock_name, company_profile, shares_outstanding = fetch_company_profile(normalized, float(latest["Close"]))
    valuation_items, valuation_summary = build_valuation_snapshot(normalized, float(latest["Close"]), shares_outstanding, fundamentals_raw, as_of_date=latest.name)
    news_score, news_summary, news_items = fetch_news_sentiment(normalized, stock_name=stock_name, period=period, as_of_date=latest.name)
    risk_score = _score_risk(latest, df)
    support_source = full_df.loc[full_df.index <= latest.name].tail(max(160, PERIOD_TO_DAYS.get(period, 365) + 40)).copy()
    support_levels, resistance_levels = extract_support_resistance(support_source, float(latest["Close"]))
    total_score = int(round((technical_score * 0.27) + (volume_score * 0.20) + (fundamental_score * 0.18) + (news_score * 0.12) + (risk_score * 0.13) + (_score_valuation(valuation_items) * 0.10)))
    decision = _decision_label(total_score, risk_score)
    position_size, stop_price, risk_budget = _position_sizing(float(latest["Close"]), capital, risk_per_trade, stop_loss_pct, market=market)
    decision_details = {
        "技术面": _build_technical_summary(latest, trend_score, momentum_score),
        "量价关系": volume_summary,
        "基本面": fundamental_summary,
        "估值视角": valuation_summary,
        "消息面": news_summary,
        "风险控制": _build_risk_summary(latest, df, stop_price, position_size, support_levels),
        "最终建议": _build_decision_summary(total_score, decision, position_size, stop_price, stock_name),
    }
    return AnalysisResult(
        ticker=normalized.upper(), stock_name=stock_name, market=market, price=float(latest["Close"]), as_of=df.index[-1].strftime("%Y-%m-%d"), decision=decision, total_score=total_score,
        technical_score=technical_score, trend_score=trend_score, momentum_score=momentum_score, volume_score=volume_score, fundamental_score=fundamental_score, news_score=news_score,
        risk_score=risk_score, position_size=position_size, stop_price=stop_price, risk_budget=risk_budget, summary="；".join(decision_details.values()), annual_volatility=float(latest["AnnualVol20"]) if pd.notna(latest["AnnualVol20"]) else np.nan,
        max_drawdown=float(df["Drawdown"].min()) if not df.empty else np.nan, atr_percent=float(latest["ATRPercent"]) if pd.notna(latest["ATRPercent"]) else np.nan, frame=df,
        decision_details=decision_details, fundamental_snapshot=fundamental_snapshot, company_profile=company_profile, valuation_items=valuation_items, news_items=news_items,
        support_levels=support_levels, resistance_levels=resistance_levels, quote=select_finance_quote(stock_name, df.index[-1]),
    )

def scan_universe(tickers: Iterable[str], period: str = "2y", capital: float = 100000.0, risk_per_trade: float = 1.0, stop_loss_pct: float = 8.0) -> pd.DataFrame:
    rows: list[dict] = []
    for item in tickers:
        ticker = normalize_ticker(item.strip())
        if not ticker:
            continue
        try:
            result = analyze_stock(ticker, period=period, capital=capital, risk_per_trade=risk_per_trade, stop_loss_pct=stop_loss_pct)
            rows.append({
                "Ticker": result.ticker, "Name": result.stock_name, "Price": round(result.price, 2), "Score": result.total_score, "Technical": result.technical_score,
                "Volume": result.volume_score, "Fundamental": result.fundamental_score, "News": result.news_score, "Risk": result.risk_score, "Decision": result.decision,
                "Volatility20": round(result.annual_volatility * 100, 2) if pd.notna(result.annual_volatility) else np.nan,
                "MaxDrawdown": round(result.max_drawdown * 100, 2) if pd.notna(result.max_drawdown) else np.nan,
                "ATR%": round(result.atr_percent * 100, 2) if pd.notna(result.atr_percent) else np.nan,
                "Support": round(result.support_levels[0], 2) if result.support_levels else np.nan,
                "Resistance": round(result.resistance_levels[0], 2) if result.resistance_levels else np.nan,
                "Shares": result.position_size,
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({"Ticker": ticker, "Name": _get_cn_name_map().get(ticker.split(".")[0], ticker.split(".")[0]), "Price": np.nan, "Score": -1, "Technical": np.nan, "Volume": np.nan, "Fundamental": np.nan, "News": np.nan, "Risk": np.nan, "Decision": f"失败: {exc}", "Volatility20": np.nan, "MaxDrawdown": np.nan, "ATR%": np.nan, "Support": np.nan, "Resistance": np.nan, "Shares": np.nan})
    ranking = pd.DataFrame(rows)
    return ranking.sort_values(["Score", "Risk", "Volume"], ascending=[False, False, False], na_position="last") if not ranking.empty else ranking


def allocate_portfolio(tickers: Iterable[str], period: str = "1y", capital: float = 100000.0, risk_per_trade: float = 1.0, stop_loss_pct: float = 8.0, total_risk_limit_pct: float = 4.0, max_position_pct: float = 25.0, reserve_cash_pct: float = 15.0, max_positions: int = 5) -> PortfolioPlan:
    results: list[AnalysisResult] = []
    notes: list[str] = []
    for item in tickers:
        ticker = normalize_ticker(item.strip())
        if not ticker:
            continue
        try:
            results.append(analyze_stock(ticker=ticker, period=period, capital=capital, risk_per_trade=risk_per_trade, stop_loss_pct=stop_loss_pct))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{ticker}: {exc}")
    eligible = [item for item in results if item.total_score >= 60 and item.risk_score >= 45 and item.decision != "回避"]
    eligible = sorted(eligible, key=lambda item: (item.total_score, item.volume_score, item.fundamental_score), reverse=True)[:max_positions]
    if not eligible:
        return PortfolioPlan(plan=pd.DataFrame(columns=["代码", "名称", "评分", "建议", "建议股数", "预计投入", "止损价", "单票风险", "仓位占比"]), total_allocated=0.0, cash_left=capital, total_risk_budget=capital * (total_risk_limit_pct / 100), total_estimated_risk=0.0, invested_ratio=0.0, notes=notes + ["当前没有满足综合评分和风控门槛的标的。"])
    target_capital = capital * (1 - reserve_cash_pct / 100)
    total_risk_budget = capital * (total_risk_limit_pct / 100)
    score_sum = sum(max(item.total_score, 1) for item in eligible)
    rows: list[dict[str, object]] = []
    total_allocated = 0.0
    total_estimated_risk = 0.0
    for item in eligible:
        weight = max(item.total_score, 1) / score_sum
        price = item.price
        per_share_risk = max(price - item.stop_price, 0)
        board_lot = 100 if item.market == "cn" else 1
        risk_budget = total_risk_budget * weight
        capital_budget = min(target_capital * weight * 1.15, capital * (max_position_pct / 100))
        if per_share_risk <= 0 or price <= 0:
            shares = 0
        else:
            risk_shares = int((risk_budget // per_share_risk) // board_lot) * board_lot
            capital_shares = int((capital_budget // price) // board_lot) * board_lot
            raw_limit = int((item.position_size // board_lot)) * board_lot if board_lot > 1 else item.position_size
            if raw_limit <= 0:
                raw_limit = item.position_size
            shares = max(0, min(risk_shares, capital_shares, raw_limit))
        allocated = shares * price
        estimated_risk = shares * per_share_risk
        total_allocated += allocated
        total_estimated_risk += estimated_risk
        rows.append({"代码": item.ticker, "名称": item.stock_name, "评分": item.total_score, "建议": item.decision, "建议股数": int(shares), "预计投入": round(allocated, 2), "止损价": round(item.stop_price, 2) if pd.notna(item.stop_price) else np.nan, "单票风险": round(estimated_risk, 2), "仓位占比": round((allocated / capital) * 100, 2) if capital > 0 else 0.0, "说明": f"按评分权重 {weight * 100:.1f}% 分配，且受单票仓位与总风险双重约束。"})
    plan = pd.DataFrame(rows).sort_values(["评分", "预计投入"], ascending=[False, False])
    cash_left = max(capital - total_allocated, 0.0)
    return PortfolioPlan(plan=plan, total_allocated=round(total_allocated, 2), cash_left=round(cash_left, 2), total_risk_budget=round(total_risk_budget, 2), total_estimated_risk=round(total_estimated_risk, 2), invested_ratio=round((total_allocated / capital), 4) if capital > 0 else 0.0, notes=notes)



def evaluate_holdings(positions: Iterable[dict[str, float | str]], period: str = "1y", capital: float = 100000.0, risk_per_trade: float = 1.0, stop_loss_pct: float = 8.0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for position in positions:
        ticker = normalize_ticker(str(position.get("ticker", "")), market="cn")
        shares = int(float(position.get("shares", 0) or 0))
        cost = float(position.get("cost", 0) or 0)
        if not ticker or shares <= 0 or cost <= 0:
            continue
        try:
            result = analyze_stock(ticker=ticker, period=period, capital=capital, risk_per_trade=risk_per_trade, stop_loss_pct=stop_loss_pct)
            latest = result.frame.iloc[-1]
            support = result.support_levels[0] if result.support_levels else float(latest.get("SMA20", result.price))
            resistance = next((level for level in result.resistance_levels if level >= result.price), result.price * 1.03)
            t_buy = max(support, float(latest.get("TrendLine60", support)) * 0.995)
            t_sell = max(resistance, result.price * 1.015)
            pnl_pct = (result.price / cost - 1) if cost > 0 else np.nan
            action, reason = _holding_action(result, pnl_pct, support, resistance)
            rows.append({
                "代码": result.ticker,
                "名称": result.stock_name,
                "持仓股数": shares,
                "成本价": round(cost, 2),
                "现价": round(result.price, 2),
                "浮盈亏": f"{pnl_pct * 100:.2f}%" if pd.notna(pnl_pct) else "N/A",
                "当前决策": action,
                "做T低吸区": f"{t_buy:.2f} 附近",
                "做T高抛区": f"{t_sell:.2f} 附近",
                "观察止盈位": f"{max(cost * 1.06, resistance):.2f}",
                "风险止损位": f"{result.stop_price:.2f}",
                "说明": reason,
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({"代码": ticker, "名称": "", "持仓股数": shares, "成本价": round(cost, 2), "现价": np.nan, "浮盈亏": "N/A", "当前决策": f"分析失败: {exc}", "做T低吸区": "N/A", "做T高抛区": "N/A", "观察止盈位": "N/A", "风险止损位": "N/A", "说明": "该持仓未能完成分析"})
    return pd.DataFrame(rows)


def _holding_action(result: AnalysisResult, pnl_pct: float, support: float, resistance: float) -> tuple[str, str]:
    if result.total_score < 50:
        return "反弹减仓/止损", f"综合分 {result.total_score} 偏弱，当前更像反弹处理，不适合继续恋战。"
    if pd.notna(pnl_pct) and pnl_pct >= 0.08 and result.price >= resistance * 0.98:
        return "优先高抛做T", f"已经有一定浮盈，且价格接近压力位 {resistance:.2f}，更适合先卖后看回补。"
    if result.total_score >= 68 and result.price <= support * 1.015:
        return "可考虑低吸做T", f"综合分 {result.total_score} 不低，价格接近支撑位 {support:.2f}，可考虑小仓低吸做T。"
    if result.total_score >= 68:
        return "以持有为主", f"趋势和量价尚可，等待接近 {resistance:.2f} 再考虑减仓，回踩 {support:.2f} 看承接。"
    return "观望，不急于操作", f"分数中性，当前更适合等价格接近 {support:.2f} 或突破 {resistance:.2f} 后再动作。"
def pick_main_board_top5(period: str = "1y", capital: float = 100000.0, risk_per_trade: float = 1.0, stop_loss_pct: float = 8.0, limit: int = 5) -> pd.DataFrame:
    ranking = scan_universe(A_MAIN_BOARD_CORE_POOL, period=period, capital=capital, risk_per_trade=risk_per_trade, stop_loss_pct=stop_loss_pct)
    if ranking.empty:
        return ranking
    ranking = ranking.loc[(ranking["Score"] >= 0) & (~ranking["Decision"].astype(str).str.startswith("失败"))].copy()
    ranking.insert(0, "候选池", f"主板核心池 {len(A_MAIN_BOARD_CORE_POOL)} 只")
    return ranking.head(limit).reset_index(drop=True)


def run_backtest(df: pd.DataFrame, stop_loss_pct: float = 0.08, trade_cost_rate: float = 0.001) -> BacktestResult:
    data = df.copy()
    data["Signal"] = 0
    data["Position"] = 0
    data["EntryMarker"] = np.nan
    data["ExitMarker"] = np.nan
    in_position = False
    trailing_stop = np.nan
    rolling_high = np.nan
    for idx in range(1, len(data)):
        today = data.iloc[idx]
        yesterday = data.iloc[idx - 1]
        can_enter = today["Close"] > today["SMA50"] and today["SMA20"] > today["SMA50"] and today["MACDHist"] > 0 and today["VolumeRatio"] > 1.05 and today["OBV"] > today["OBVEMA20"] and today["Close"] > today["TrendLine60"] and (today["Breakout60"] or today["Close"] >= yesterday["High20"] * 0.995)
        must_exit = False
        if in_position:
            rolling_high = max(rolling_high, today["Close"])
            trailing_stop = max(trailing_stop, rolling_high * (1 - stop_loss_pct))
            must_exit = today["Close"] < today["SMA20"] or today["MACDHist"] < 0 or today["Close"] < trailing_stop or today["Close"] < today["TrendLine60"]
        if not in_position and can_enter:
            in_position = True
            rolling_high = today["Close"]
            trailing_stop = today["Close"] * (1 - stop_loss_pct)
            data.iloc[idx, data.columns.get_loc("Signal")] = 1
            data.iloc[idx, data.columns.get_loc("EntryMarker")] = today["Close"]
        elif in_position and must_exit:
            in_position = False
            trailing_stop = np.nan
            rolling_high = np.nan
            data.iloc[idx, data.columns.get_loc("Signal")] = -1
            data.iloc[idx, data.columns.get_loc("ExitMarker")] = today["Close"]
        data.iloc[idx, data.columns.get_loc("Position")] = 1 if in_position else 0
    data["Position"] = data["Position"].shift(1).fillna(0)
    turnover = data["Position"].diff().abs().fillna(data["Position"])
    data["StrategyReturn"] = (data["Return"].fillna(0) * data["Position"]) - (turnover * trade_cost_rate)
    data["StrategyEquity"] = (1 + data["StrategyReturn"]).cumprod()
    data["BuyHoldEquity"] = (1 + data["Return"].fillna(0)).cumprod()
    trades = _extract_trades(data)
    return BacktestResult(frame=data, trades=trades, strategy_cagr=_cagr(data["StrategyEquity"]), buy_hold_cagr=_cagr(data["BuyHoldEquity"]), strategy_max_drawdown=_max_drawdown_from_equity(data["StrategyEquity"]), buy_hold_max_drawdown=_max_drawdown_from_equity(data["BuyHoldEquity"]), strategy_sharpe=_sharpe(data["StrategyReturn"]), buy_hold_sharpe=_sharpe(data["Return"].fillna(0)), win_rate=float((trades["PnL"] > 0).mean()) if not trades.empty else np.nan, exposure=float(data["Position"].mean()) if not data.empty else np.nan, trade_cost_rate=trade_cost_rate)


def extract_support_resistance(df: pd.DataFrame, price: float) -> tuple[list[float], list[float]]:
    recent = df.tail(160).copy()
    support_candidates = recent["PivotLow"].dropna().tolist() + recent["Low20"].dropna().tail(3).tolist() + recent["Low60"].dropna().tail(2).tolist()
    resistance_candidates = recent["PivotHigh"].dropna().tolist() + recent["High20"].dropna().tail(3).tolist() + recent["High60"].dropna().tail(2).tolist()
    supports = _deduplicate_levels([value for value in support_candidates if value < price * 1.02], reverse=True)
    resistances = _deduplicate_levels([value for value in resistance_candidates if value > price * 0.98], reverse=False)
    return supports[:3], resistances[:3]


def select_finance_quote(seed_text: str, timestamp: pd.Timestamp) -> dict[str, str]:
    seed = int(timestamp.strftime("%Y%m%d")) + sum(ord(char) for char in seed_text)
    return FINANCE_QUOTES[seed % len(FINANCE_QUOTES)]


def _fetch_valuation_history(symbol: str) -> pd.DataFrame:
    try:
        df = _retry(lambda: ak.stock_value_em(symbol=symbol), retries=2, pause=0.8)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    frame = df.copy()
    if "数据日期" in frame.columns:
        frame["数据日期"] = pd.to_datetime(frame["数据日期"], errors="coerce")
    for column in ["PE(TTM)", "PE(静)", "市净率", "PEG值", "市现率", "市销率"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("数据日期") if "数据日期" in frame.columns else frame


def _historical_valuation_stats(frame: pd.DataFrame, column: str) -> dict[str, float]:
    if frame is None or frame.empty or column not in frame.columns:
        return {}
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    series = series[series > 0]
    if series.empty:
        return {}
    latest = float(series.iloc[-1])
    return {
        "latest": latest,
        "median": float(series.median()),
        "q25": float(series.quantile(0.25)),
        "q75": float(series.quantile(0.75)),
        "percentile": float((series <= latest).mean()),
    }


def _price_from_multiple(price: float, current_multiple: float, target_multiple: float) -> float:
    if pd.isna(price) or pd.isna(current_multiple) or pd.isna(target_multiple) or current_multiple <= 0 or target_multiple <= 0:
        return np.nan
    return float(price * target_multiple / current_multiple)


def _value_judgement(price: float, fair_price: float, buy_price: float, fallback: str = "暂无法判断") -> str:
    if pd.isna(price) or pd.isna(fair_price) or fair_price <= 0:
        return fallback
    if pd.notna(buy_price) and price <= buy_price:
        return "低于安全买点，可重点跟踪"
    if price <= fair_price * 0.92:
        return "低于合理价值，可分批关注"
    if price <= fair_price * 1.05:
        return "接近合理价值"
    return "高于合理价值，耐心等价"


def _valuation_item(method: str, current_value: float, judgement: str, description: str, fair_price: float, buy_price: float, current_price: float, hist_stats: dict[str, float], extra_note: str, value_formatter=None) -> dict[str, str]:
    formatter = value_formatter or (lambda value: f"{value:.2f}")
    percentile_note = ""
    if hist_stats and pd.notna(hist_stats.get("percentile", np.nan)):
        percentile_note = f"历史分位 {hist_stats['percentile'] * 100:.1f}%"
    note = "；".join(part for part in [description, percentile_note, extra_note] if part)
    return {
        "方法": method,
        "当前值": formatter(current_value),
        "判断": judgement,
        "合理价": _format_price(fair_price),
        "安全买点": _format_price(buy_price),
        "潜在空间": _format_percent_delta(fair_price, current_price),
        "说明": note,
    }


def _format_price(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.2f}"


def _format_percent_delta(target_price: float, base_price: float) -> str:
    if pd.isna(target_price) or pd.isna(base_price) or base_price <= 0:
        return "N/A"
    return f"{((target_price / base_price) - 1) * 100:.2f}%"

def _score_trend(latest: pd.Series) -> int:
    score = 0
    if latest["Close"] > latest["SMA20"]:
        score += 20
    if latest["Close"] > latest["SMA50"]:
        score += 20
    if pd.notna(latest["SMA200"]) and latest["Close"] > latest["SMA200"]:
        score += 15
    if latest["SMA20"] > latest["SMA50"]:
        score += 20
    if pd.notna(latest["SMA200"]) and latest["SMA50"] > latest["SMA200"]:
        score += 15
    if pd.notna(latest["RollingHigh252"]) and latest["Close"] / latest["RollingHigh252"] >= 0.9:
        score += 10
    if pd.notna(latest.get("TrendSlope60", np.nan)) and latest["TrendSlope60"] > 0:
        score += 10
    return int(min(score, 100))


def _score_momentum(latest: pd.Series) -> int:
    score = 0
    rsi = latest["RSI14"]
    if 52 <= rsi <= 68:
        score += 30
    elif 45 <= rsi <= 75:
        score += 20
    elif rsi < 40:
        score += 5
    if latest["MACDHist"] > 0:
        score += 30
    if latest["Return20"] > 0:
        score += 20
    if latest["Return60"] > 0:
        score += 15
    if rsi > 78:
        score -= 15
    return int(max(0, min(score, 100)))


def _score_volume(latest: pd.Series) -> tuple[int, str]:
    score = 50
    parts: list[str] = []
    volume_ratio = latest.get("VolumeRatio", np.nan)
    daily_volume_ratio = latest.get("DailyVolumeRatio", np.nan)
    turnover = latest.get("Turnover", np.nan)
    turnover_avg = latest.get("TurnoverMA20", np.nan)
    if pd.notna(volume_ratio):
        if volume_ratio >= 1.2 and latest.get("Return20", 0) > 0:
            score += 20
            parts.append("5 日均量明显高于 20 日均量，放量方向与上涨一致")
        elif volume_ratio < 0.85 and latest.get("Return20", 0) > 0:
            score -= 10
            parts.append("上涨但均量没有同步放大，趋势延续性一般")
        elif volume_ratio > 1.3 and latest.get("Return20", 0) < 0:
            score -= 12
            parts.append("放量下跌，筹码分歧偏大")
    if pd.notna(daily_volume_ratio) and daily_volume_ratio >= 1.5:
        score += 8
        parts.append("当日成交量显著高于 20 日均量")
    if latest.get("OBV", 0) > latest.get("OBVEMA20", 0):
        score += 15
        parts.append("OBV 仍在累积，资金没有明显背离")
    else:
        score -= 10
        parts.append("OBV 没有明显走强")
    if bool(latest.get("Breakout60", False)):
        score += 15
        parts.append("价格接近或突破近 60 日高点")
    if pd.notna(turnover) and pd.notna(turnover_avg):
        if turnover > turnover_avg:
            score += 10
            parts.append("换手率高于近 20 日均值，筹码交换更活跃")
        else:
            parts.append("换手率没有明显放大")
    return int(max(0, min(score, 100))), "；".join(parts) if parts else "量价关系暂无明显优势。"


def _score_risk(latest: pd.Series, df: pd.DataFrame) -> int:
    score = 100
    annual_vol = latest["AnnualVol20"]
    atr_pct = latest["ATRPercent"]
    max_drawdown = df["Drawdown"].min()
    if pd.notna(annual_vol):
        score -= 30 if annual_vol > 0.60 else 20 if annual_vol > 0.40 else 10 if annual_vol > 0.25 else 0
    if pd.notna(max_drawdown):
        score -= 35 if max_drawdown < -0.50 else 25 if max_drawdown < -0.30 else 10 if max_drawdown < -0.15 else 0
    if pd.notna(atr_pct):
        score -= 20 if atr_pct > 0.05 else 10 if atr_pct > 0.03 else 0
    return int(max(0, min(score, 100)))


def _score_valuation(valuation_items: list[dict[str, str]]) -> int:
    score = 50
    for item in valuation_items:
        judgement = item.get("判断", "")
        if "低于安全买点" in judgement or "偏便宜" in judgement or "成长匹配较好" in judgement:
            score += 18
        elif "低于合理价值" in judgement:
            score += 15
        elif "合理" in judgement or "匹配" in judgement:
            score += 5
        elif "高于合理价值" in judgement or "偏贵" in judgement or "透支" in judgement:
            score -= 10
    return int(max(0, min(score, 100)))


def _decision_label(total_score: int, risk_score: int) -> str:
    if total_score >= 82 and risk_score >= 55:
        return "偏强，可分批跟踪"
    if total_score >= 68 and risk_score >= 45:
        return "中性偏多，等确认"
    if total_score >= 52:
        return "观察"
    return "回避"


def _build_technical_summary(latest: pd.Series, trend_score: int, momentum_score: int) -> str:
    parts = ["均线结构偏强" if trend_score >= 70 else "均线结构一般" if trend_score >= 45 else "均线结构偏弱", "60 日趋势线向上" if latest.get("TrendSlope60", 0) > 0 else "60 日趋势线走平或走弱"]
    if 52 <= latest["RSI14"] <= 68:
        parts.append("RSI 落在健康动量区")
    elif latest["RSI14"] > 78:
        parts.append("RSI 偏热，不适合追高")
    else:
        parts.append("RSI 没有形成极端超买")
    parts.append("MACD 偏多" if latest["MACDHist"] > 0 else "MACD 仍未翻强")
    parts.append(f"趋势分 {trend_score}，动量分 {momentum_score}")
    return "；".join(parts)


def _build_risk_summary(latest: pd.Series, df: pd.DataFrame, stop_price: float, position_size: int, supports: list[float]) -> str:
    parts = [f"20 日年化波动 {latest['AnnualVol20'] * 100:.2f}%" if pd.notna(latest["AnnualVol20"]) else "波动率缺失", f"历史最大回撤 {df['Drawdown'].min() * 100:.2f}%", f"ATR 占比 {latest['ATRPercent'] * 100:.2f}%" if pd.notna(latest["ATRPercent"]) else "ATR 缺失"]
    if supports:
        parts.append(f"最近支撑位参考 {supports[0]:.2f}")
    if pd.notna(stop_price):
        parts.append(f"参考止损价 {stop_price:.2f}")
    if position_size > 0:
        parts.append(f"按当前单笔风险规则，理论上限 {position_size} 股")
    return "；".join(parts)


def _build_decision_summary(total_score: int, decision: str, position_size: int, stop_price: float, stock_name: str) -> str:
    action = "先观察，等趋势、量能或基本面继续改善再出手。" if decision == "观察" else "可以放进重点跟踪名单，但更适合等放量突破或回踩确认。" if decision == "中性偏多，等确认" else "具备跟踪价值，仍然只能分批介入，不能一次性满仓。" if decision == "偏强，可分批跟踪" else "当前不值得主动承担风险。"
    position_text = f"建议按理论仓位 {position_size} 股以内控制" if position_size > 0 else "当前不建议扩大仓位"
    stop_text = f"，参考止损 {stop_price:.2f}" if pd.notna(stop_price) else ""
    return f"{stock_name} 当前综合分 {total_score}，结论为“{decision}”。{action}{position_text}{stop_text}。"


def _extract_trades(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=["EntryDate", "ExitDate", "EntryPrice", "ExitPrice", "PnL", "HoldingDays"])
    trades: list[dict] = []
    entry_date = None
    entry_price = None
    for index, row in data.iterrows():
        if row["Signal"] == 1 and entry_date is None:
            entry_date, entry_price = index, row["Close"]
        elif row["Signal"] == -1 and entry_date is not None:
            exit_price = row["Close"]
            trades.append({"EntryDate": entry_date.strftime("%Y-%m-%d"), "ExitDate": index.strftime("%Y-%m-%d"), "EntryPrice": round(float(entry_price), 2), "ExitPrice": round(float(exit_price), 2), "PnL": float(exit_price / entry_price - 1), "HoldingDays": int((index - entry_date).days)})
            entry_date = None
            entry_price = None
    if entry_date is not None and entry_price is not None:
        exit_price = data.iloc[-1]["Close"]
        trades.append({"EntryDate": entry_date.strftime("%Y-%m-%d"), "ExitDate": data.index[-1].strftime("%Y-%m-%d"), "EntryPrice": round(float(entry_price), 2), "ExitPrice": round(float(exit_price), 2), "PnL": float(exit_price / entry_price - 1), "HoldingDays": int((data.index[-1] - entry_date).days)})
    return pd.DataFrame(trades)


def _get_cn_name_map() -> dict[str, str]:
    global _CN_NAME_CACHE
    if _CN_NAME_CACHE is not None:
        return _CN_NAME_CACHE
    try:
        df = ak.stock_info_a_code_name()
        _CN_NAME_CACHE = {str(row["code"]): str(row["name"]) for _, row in df.iterrows()}
    except Exception:
        _CN_NAME_CACHE = {}
    return _CN_NAME_CACHE


def _retry(func, retries: int = 2, pause: float = 0.8):
    for attempt in range(retries + 1):
        try:
            return func()
        except Exception:  # noqa: BLE001
            if attempt >= retries:
                raise
            time.sleep(pause)


def _to_xq_symbol(ticker: str) -> str:
    normalized = normalize_ticker(ticker)
    if normalized.endswith(".SS"):
        return f"SH{normalized.split('.')[0]}"
    if normalized.endswith(".SZ"):
        return f"SZ{normalized.split('.')[0]}"
    return normalized.replace(".", "")


def _rolling_linear_regression(series: pd.Series, window: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    x = np.arange(window)
    for idx in range(window - 1, len(values)):
        y = values[idx - window + 1 : idx + 1]
        if np.isnan(y).any():
            continue
        slope, intercept = np.polyfit(x, y, 1)
        result[idx] = (slope * x[-1]) + intercept
    return pd.Series(result, index=series.index)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    x = np.arange(window)
    for idx in range(window - 1, len(values)):
        y = values[idx - window + 1 : idx + 1]
        if np.isnan(y).any():
            continue
        slope, _ = np.polyfit(x, y, 1)
        result[idx] = slope
    return pd.Series(result, index=series.index)


def _bucket_value(value: float, low: float, high: float, low_label: str, mid_label: str, high_label: str) -> str:
    return low_label if value <= low else mid_label if value <= high else high_label


def _deduplicate_levels(levels: list[float], reverse: bool) -> list[float]:
    cleaned: list[float] = []
    for level in sorted(levels, reverse=reverse):
        if not any(abs(level - existing) / max(existing, 1e-6) < 0.02 for existing in cleaned):
            cleaned.append(float(level))
    return cleaned


def _format_large_number(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    absolute = abs(float(value))
    if absolute >= 100000000:
        return f"{value / 100000000:.2f} 亿"
    if absolute >= 10000:
        return f"{value / 10000:.2f} 万"
    return f"{value:.2f}"


def _safe_float(value) -> float:
    if value in (None, "None", "", False):
        return np.nan
    text = str(value).replace(",", "").replace("%", "").replace("元", "").strip()
    if not text or text.lower() == "nan":
        return np.nan
    multiplier = 1.0
    if text.endswith("万亿"):
        multiplier = 1000000000000.0
        text = text[:-2]
    elif text.endswith("亿"):
        multiplier = 100000000.0
        text = text[:-1]
    elif text.endswith("万"):
        multiplier = 10000.0
        text = text[:-1]
    elif text.endswith("千"):
        multiplier = 1000.0
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


def _parse_percent(value) -> float | None:
    parsed = _safe_float(value)
    return None if pd.isna(parsed) else float(parsed)


def _cagr(equity_curve: pd.Series) -> float:
    clean = equity_curve.dropna()
    if clean.empty or len(clean) < 2:
        return np.nan
    years = len(clean) / TRADING_DAYS
    return np.nan if years <= 0 else float(clean.iloc[-1] ** (1 / years) - 1)


def _max_drawdown_from_equity(equity_curve: pd.Series) -> float:
    clean = equity_curve.dropna()
    return np.nan if clean.empty else float((clean / clean.cummax() - 1).min())


def _sharpe(returns: pd.Series) -> float:
    clean = returns.dropna()
    return np.nan if clean.empty or clean.std() == 0 else float((clean.mean() / clean.std()) * sqrt(TRADING_DAYS))


def _position_sizing(price: float, capital: float, risk_per_trade: float, stop_loss_pct: float, market: str = "cn") -> tuple[int, float, float]:
    if capital <= 0 or risk_per_trade <= 0 or stop_loss_pct <= 0:
        return 0, np.nan, np.nan
    risk_budget = capital * (risk_per_trade / 100)
    stop_price = price * (1 - stop_loss_pct / 100)
    per_share_risk = max(price - stop_price, 0)
    if per_share_risk == 0:
        return 0, stop_price, risk_budget
    shares = int(risk_budget // per_share_risk)
    if market == "cn":
        shares = (shares // 100) * 100
    return shares, stop_price, risk_budget








