from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
import re
import time
from math import isnan

import akshare as ak
import numpy as np
import pandas as pd
import requests

from analysis_engine import analyze_stock, normalize_ticker, scan_universe

POLICY_SIGNALS = [
    {
        "theme": "AI智能体与算力",
        "date": "2026-03-05",
        "source": "2026政府工作报告",
        "summary": "深化拓展人工智能+，促进新一代智能终端和智能体加快推广，推动重点行业人工智能商业化规模化应用。",
        "keywords": ["人工智能", "AI", "智能体", "算力", "终端", "大模型", "服务器", "液冷", "芯片"],
        "source_url": "https://www.moe.gov.cn/jyb_xwfb/moe_1946/2026/202603/t20260305_1430052.html",
    },
    {
        "theme": "人形机器人与具身智能",
        "date": "2026-02-28",
        "source": "人形机器人与具身智能标准体系",
        "summary": "我国首个人形机器人与具身智能标准体系发布，覆盖全产业链和全生命周期。",
        "keywords": ["机器人", "具身智能", "人形", "伺服", "减速器", "传感器", "自动化"],
        "source_url": "https://www.ncsti.gov.cn/kjdt/kjrd/202603/t20260302_239474.html",
    },
    {
        "theme": "能源安全与高股息",
        "date": "2026-03-07",
        "source": "地缘冲突与能源安全",
        "summary": "中东局势升级推升原油和天然气价格，能源安全与资源保供重新成为市场关注焦点。",
        "keywords": ["原油", "油气", "天然气", "煤炭", "电力", "能源安全", "航运", "高股息"],
        "source_url": "https://companies.caixin.com/2026-03-07/102420649.html",
    },
    {
        "theme": "军工与商业航天",
        "date": "2026-02-28",
        "source": "低空经济与商业航天地方政策",
        "summary": "地方政府继续推进低空经济与商业航天示范和城市空中交通试点，相关产业链景气度被持续讨论。",
        "keywords": ["商业航天", "卫星", "航天", "低空", "军工", "火箭", "无人机"],
        "source_url": "https://www.gz.gov.cn/attachment/7/7976/7976695/10699545.pdf",
    },
]
_SNAPSHOT_CACHE: pd.DataFrame | None = None
_SNAPSHOT_CACHE_AT = 0.0

THEME_POOLS = {
    "AI智能体与算力": ["603019", "600588", "600845", "601360", "600536", "600941", "603000", "601728"],
    "人形机器人与具身智能": ["603728", "603662", "603583", "600031", "601766", "600835", "603960", "600885"],
    "能源安全与高股息": ["601088", "600938", "600900", "600188", "601857", "601898", "600011", "601225"],
    "储能电网": ["600406", "601877", "600089", "603659", "600884", "002594", "600312", "603606"],
    "军工与商业航天": ["600760", "600893", "600879", "600118", "601698", "600501", "601989", "600372"],
    "半导体与设备": ["603986", "600584", "603501", "688981", "603283", "600460", "603005", "601012"],
    "创新药与医药服务": ["600276", "603259", "600196", "603127", "600332", "600161", "600763", "603707"],
    "消费龙头与白酒": ["600519", "000858", "600809", "600887", "603288", "600600", "600690", "000333"],
    "券商与金融高股息": ["600030", "601318", "600036", "601601", "601688", "601398", "601288", "601166"],
    "黄金与有色资源": ["600489", "601600", "600547", "601168", "603993", "000975", "600111", "601899"],
    "基建出海与高端装备": ["601390", "601800", "600039", "601117", "600150", "601766", "600760", "601186"],
    "低空经济与无人机": ["600372", "600879", "600118", "600893", "601698", "603766", "600316", "600038"],
}


def fetch_full_a_market_snapshot() -> pd.DataFrame:
    global _SNAPSHOT_CACHE, _SNAPSHOT_CACHE_AT
    if _SNAPSHOT_CACHE is not None and (time.time() - _SNAPSHOT_CACHE_AT) < 900:
        return _SNAPSHOT_CACHE.copy()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Referer": "https://finance.sina.com.cn/",
        "Upgrade-Insecure-Requests": "1",
    }
    count_url = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeStockCount?node=hs_a"
    data_url = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
    try:
        count_response = requests.get(count_url, headers=headers, timeout=20)
        count_response.raise_for_status()
        page_count = int(re.findall(r"\d+", count_response.text)[0])
    except Exception:
        return _SNAPSHOT_CACHE.copy() if _SNAPSHOT_CACHE is not None else pd.DataFrame()
    page_count = (page_count // 100) + (1 if page_count % 100 else 0)
    rows = []
    for page in range(1, page_count + 1):
        params = {"page": str(page), "num": "100", "sort": "symbol", "asc": "1", "node": "hs_a", "symbol": "", "_s_r_a": "page"}
        success = False
        for _ in range(3):
            try:
                response = requests.get(data_url, params=params, headers=headers, timeout=20)
                response.raise_for_status()
                batch = response.json()
                if batch:
                    rows.extend(batch)
                    success = True
                    break
            except Exception:
                time.sleep(0.6)
                continue
        if not success:
            continue
    if not rows:
        return _SNAPSHOT_CACHE.copy() if _SNAPSHOT_CACHE is not None else pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.rename(columns={"symbol": "代码", "code": "6位代码", "name": "名称", "trade": "最新价", "pricechange": "涨跌额", "changepercent": "涨跌幅", "buy": "买入", "sell": "卖出", "settlement": "昨收", "open": "今开", "high": "最高", "low": "最低", "volume": "成交量", "amount": "成交额", "ticktime": "时间戳", "turnoverratio": "换手率"})
    df["代码"] = df["代码"].astype(str)
    df["代码"] = df["代码"].astype(str)
    df["6位代码"] = df["代码"].str[-6:]
    df["市场后缀"] = np.where(df["代码"].str.startswith("sh"), ".SS", np.where(df["代码"].str.startswith("sz"), ".SZ", ""))
    df["Ticker"] = df["6位代码"] + df["市场后缀"]
    numeric_cols = ["最新价", "涨跌额", "涨跌幅", "买入", "卖出", "昨收", "今开", "最高", "最低", "成交量", "成交额"]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    _SNAPSHOT_CACHE = df.copy()
    _SNAPSHOT_CACHE_AT = time.time()
    return df


def rank_full_market_main_board_top5(period: str = "1y", capital: float = 100000.0, risk_per_trade: float = 1.0, stop_loss_pct: float = 8.0, candidate_count: int = 100, top_n: int | None = None) -> pd.DataFrame:
    snapshot = fetch_full_a_market_snapshot()
    if snapshot.empty:
        fallback = _silent_scan_universe(_fallback_main_board_pool(), period=period, capital=capital, risk_per_trade=risk_per_trade, stop_loss_pct=stop_loss_pct)
        if fallback.empty:
            return fallback
        fallback = fallback.loc[fallback["Score"] >= 0].copy()
        fallback["综合总分"] = fallback["Score"]
        fallback["评分原因"] = fallback.apply(_build_row_reason, axis=1) + "（实时全市场快照不可用，已回退到扩展候选池）"
        ordered = fallback.sort_values(["综合总分", "Volume"], ascending=[False, False]).reset_index(drop=True)
        return ordered if top_n is None else ordered.head(top_n).reset_index(drop=True)
    main = snapshot.loc[snapshot["Ticker"].map(_is_main_board_ticker)].copy()
    main = main.loc[~main["名称"].astype(str).str.contains("ST|退", na=False)]
    main = main.loc[(main["最新价"] > 2) & (main["成交额"] > 100000000)]
    intraday_range = (main["最高"] - main["最低"]).replace(0, np.nan)
    main["收盘强度"] = ((main["最新价"] - main["最低"]) / intraday_range).fillna(0)
    main["开盘缺口"] = ((main["今开"] - main["昨收"]) / main["昨收"].replace(0, np.nan)).fillna(0)
    main["涨跌幅分位"] = main["涨跌幅"].rank(pct=True)
    main["成交额分位"] = main["成交额"].rank(pct=True)
    main["爆发预筛分"] = (main["涨跌幅分位"] * 40) + (main["成交额分位"] * 30) + (main["收盘强度"] * 20) + (main["开盘缺口"].clip(lower=-0.03, upper=0.05) * 200)
    candidate_df = main.sort_values(["爆发预筛分", "涨跌幅", "成交额"], ascending=[False, False, False]).head(candidate_count)
    tickers = candidate_df["Ticker"].tolist()
    results = _parallel_scan(tickers, period, capital, risk_per_trade, stop_loss_pct)
    if results.empty:
        return results
    merged = results.merge(candidate_df[["Ticker", "名称", "涨跌幅", "成交额", "爆发预筛分"]], on="Ticker", how="left")
    merged["综合总分"] = (merged["Score"] * 0.72) + (merged["爆发预筛分"] * 0.28)
    merged["评分原因"] = merged.apply(_build_row_reason, axis=1)
    ordered = merged.sort_values(["综合总分", "Score", "Volume"], ascending=[False, False, False]).reset_index(drop=True)
    return ordered if top_n is None else ordered.head(top_n).reset_index(drop=True)


def fetch_macro_market_news(limit: int = 80) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    try:
        df = ak.stock_news_main_cx().head(limit)
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                title = str(row.get("summary", "")).strip()
                if not title or title in seen:
                    continue
                seen.add(title)
                rows.append({"标签": str(row.get("tag", "")), "标题": title, "链接": str(row.get("url", ""))})
                if len(rows) >= limit:
                    return rows[:limit]
    except Exception:
        pass
    try:
        cls_df = ak.stock_info_global_cls().head(limit)
        if cls_df is not None and not cls_df.empty:
            for _, row in cls_df.iterrows():
                title = str(row.get("标题", "")).strip()
                if not title or title in seen:
                    continue
                seen.add(title)
                rows.append({"标签": "财联社", "标题": title, "链接": ""})
                if len(rows) >= limit:
                    break
    except Exception:
        pass
    return rows[:limit]


def build_stock_policy_context(result) -> dict[str, object]:
    profile_text = " ".join(str(result.company_profile.get(key, "")) for key in ["所属行业", "主营业务", "公司简介", "公司名称"])
    company_news_text = " ".join(item.get("标题", "") for item in result.news_items if item.get("类型") in {"公司新闻", "公告", "券商研报"})
    policy_hits = []
    for signal in POLICY_SIGNALS:
        theme_pool = [normalize_ticker(code, market="cn") for code in THEME_POOLS.get(signal["theme"], [])]
        profile_hits = sum(keyword in profile_text for keyword in signal["keywords"])
        news_hits = sum(keyword in company_news_text for keyword in signal["keywords"])
        if result.ticker in theme_pool or profile_hits >= 2 or (profile_hits >= 1 and news_hits >= 1):
            policy_hits.append(signal)
    macro_hits = []
    for item in fetch_macro_market_news():
        if any(keyword in item["标题"] for signal in policy_hits for keyword in signal["keywords"]):
            macro_hits.append(item)
    summary_parts = []
    if policy_hits:
        summary_parts.append("政策匹配: " + "；".join(f"{item['theme']}({item['source']})" for item in policy_hits[:3]))
    if macro_hits:
        summary_parts.append("宏观/行业新闻匹配: " + "；".join(item["标题"][:26] for item in macro_hits[:3]))
    if not summary_parts:
        summary_parts.append("当前没有匹配到明显的政策催化或主线新闻。")
    return {"summary": "；".join(summary_parts), "policy_items": policy_hits[:5], "macro_items": macro_hits[:5]}

def analyze_market_themes(period: str = "1y", capital: float = 100000.0, risk_per_trade: float = 1.0, stop_loss_pct: float = 8.0, max_themes: int = 10) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    snapshot = fetch_full_a_market_snapshot()
    macro_news = fetch_macro_market_news(limit=100)
    theme_rows: list[dict[str, object]] = []
    theme_rankings: dict[str, pd.DataFrame] = {}
    for theme, pool in list(THEME_POOLS.items())[:max_themes]:
        ranking = _parallel_scan([normalize_ticker(code, market="cn") for code in pool], period, capital, risk_per_trade, stop_loss_pct)
        ranking = ranking.loc[ranking["Score"] >= 0].copy()
        if ranking.empty:
            continue
        theme_rankings[theme] = ranking.head(10).copy()
        policy_hits = [item for item in POLICY_SIGNALS if item["theme"] == theme]
        keywords = sorted({keyword for item in policy_hits for keyword in item["keywords"]}) or [theme]
        macro_hits = [item for item in macro_news if any(keyword in item["标题"] for keyword in keywords)]
        tickers = [normalize_ticker(code, market="cn") for code in pool]
        member_snapshot = snapshot.loc[snapshot["Ticker"].isin(tickers)].copy() if not snapshot.empty else pd.DataFrame()
        avg_move = float(member_snapshot["涨跌幅"].mean()) if not member_snapshot.empty else 0.0
        avg_amount = float(member_snapshot["成交额"].mean()) if not member_snapshot.empty else 0.0
        avg_score = float(ranking["Score"].head(3).mean())
        theme_score = 50 + (len(policy_hits) * 10) + (min(len(macro_hits), 5) * 4) + max(min(avg_move * 2.5, 15), -10) + max(min((avg_score - 50) * 0.45, 20), -10)
        theme_rows.append({
            "主线主题": theme,
            "主题分": round(max(0, min(theme_score, 100)), 1),
            "政策驱动": len(policy_hits),
            "新闻热度": len(macro_hits),
            "主题池平均涨跌幅": round(avg_move, 2),
            "主题池平均成交额": round(avg_amount / 100000000, 2),
            "代表股": "、".join(ranking["Name"].head(3).tolist()),
            "判断": _build_theme_reason(theme, policy_hits, macro_hits, avg_move, avg_score),
        })
        ranked = theme_rankings[theme].copy()
        ranked["评分原因"] = ranked.apply(_build_row_reason, axis=1)
        theme_rankings[theme] = ranked
    theme_df = pd.DataFrame(theme_rows).sort_values(["主题分", "政策驱动", "新闻热度"], ascending=[False, False, False]).head(max_themes).reset_index(drop=True) if theme_rows else pd.DataFrame()
    if theme_df.empty:
        return theme_df, {}
    ordered_themes = theme_df["主线主题"].tolist()
    filtered_rankings = {theme: theme_rankings[theme] for theme in ordered_themes if theme in theme_rankings}
    return theme_df, filtered_rankings


def _parallel_scan(tickers: list[str], period: str, capital: float, risk_per_trade: float, stop_loss_pct: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(_analyze_stock_silent, ticker, period, capital, risk_per_trade, stop_loss_pct): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result()
                rows.append({
                    "Ticker": result.ticker,
                    "Name": result.stock_name,
                    "Price": round(result.price, 2),
                    "Score": result.total_score,
                    "Technical": result.technical_score,
                    "Volume": result.volume_score,
                    "Fundamental": result.fundamental_score,
                    "News": result.news_score,
                    "Risk": result.risk_score,
                    "Decision": result.decision,
                    "Support": round(result.support_levels[0], 2) if result.support_levels else np.nan,
                    "Resistance": round(result.resistance_levels[0], 2) if result.resistance_levels else np.nan,
                })
            except Exception:
                continue
    return pd.DataFrame(rows)


def _silent_scan_universe(tickers: list[str], period: str, capital: float, risk_per_trade: float, stop_loss_pct: float) -> pd.DataFrame:
    sink = StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return scan_universe(tickers, period=period, capital=capital, risk_per_trade=risk_per_trade, stop_loss_pct=stop_loss_pct)


def _analyze_stock_silent(ticker: str, period: str, capital: float, risk_per_trade: float, stop_loss_pct: float):
    sink = StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return analyze_stock(ticker, period, capital, risk_per_trade, stop_loss_pct)


def _build_row_reason(row: pd.Series) -> str:
    parts = []
    if row.get("Technical", 0) >= 70:
        parts.append("趋势强")
    elif row.get("Technical", 0) >= 60:
        parts.append("趋势尚可")
    if row.get("Volume", 0) >= 65:
        parts.append("量价配合")
    if row.get("Fundamental", 0) >= 65:
        parts.append("基本面不弱")
    if row.get("News", 0) >= 60:
        parts.append("消息偏多")
    if row.get("Risk", 0) <= 45:
        parts.append("波动偏大")
    if row.get("涨跌幅", 0) >= 3:
        parts.append("当日强势")
    return "、".join(parts) if parts else str(row.get("Decision", ""))


def _build_theme_reason(theme: str, policy_hits: list[dict[str, object]], macro_hits: list[dict[str, str]], avg_move: float, avg_score: float) -> str:
    parts = []
    if policy_hits:
        parts.append(f"有 {len(policy_hits)} 条政策催化")
    if macro_hits:
        parts.append(f"近端新闻热度 {len(macro_hits)}")
    if avg_move > 1:
        parts.append("主题池短线强于大盘")
    elif avg_move < -1:
        parts.append("主题池短线走弱")
    if avg_score >= 68:
        parts.append("板块内高分个股较多")
    elif avg_score < 55:
        parts.append("板块内高分个股不足")
    return f"{theme}: " + "；".join(parts) if parts else f"{theme}: 暂无明显主线优势"


def _is_main_board_ticker(ticker: str) -> bool:
    code = ticker.split(".")[0]
    return code.startswith(("600", "601", "603", "605", "000", "001", "002", "003"))




















def _fallback_main_board_pool() -> list[str]:
    base = ["600519", "000858", "600036", "601318", "600276", "600900", "000333", "002594", "601899", "600030", "600887", "600809", "601166", "600309", "601088", "603259", "603288", "600031", "600570", "601888"]
    theme_members = [code for members in THEME_POOLS.values() for code in members if not str(code).startswith("688")]
    return sorted(set(base + theme_members))

