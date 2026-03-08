from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_store"
TOP_DIR = DATA_DIR / "top_picks"
THEME_DIR = DATA_DIR / "themes"
PREFS_FILE = DATA_DIR / "preferences.json"


def ensure_store_dirs() -> None:
    for folder in [DATA_DIR, TOP_DIR, THEME_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def load_preferences() -> dict[str, Any]:
    ensure_store_dirs()
    if not PREFS_FILE.exists():
        return {}
    try:
        return json.loads(PREFS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_preferences(updates: dict[str, Any]) -> dict[str, Any]:
    ensure_store_dirs()
    prefs = load_preferences()
    prefs.update(updates)
    PREFS_FILE.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")
    return prefs


def save_top_picks_snapshot(df: pd.DataFrame, meta: dict[str, Any]) -> Path:
    ensure_store_dirs()
    stamp = meta.get("timestamp", pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
    payload = df.copy()
    payload.to_csv(TOP_DIR / f"top_picks_{stamp}.csv", index=False, encoding="utf-8-sig")
    payload.to_csv(TOP_DIR / "latest_top_picks.csv", index=False, encoding="utf-8-sig")
    (TOP_DIR / "latest_top_picks.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return TOP_DIR / f"top_picks_{stamp}.csv"


def load_top_picks_snapshot(filename: str = "latest_top_picks.csv") -> tuple[pd.DataFrame | None, dict[str, Any]]:
    ensure_store_dirs()
    csv_path = TOP_DIR / filename
    meta_path = TOP_DIR / "latest_top_picks.json"
    if not csv_path.exists():
        return None, {}
    try:
        frame = pd.read_csv(csv_path)
    except Exception:
        return None, {}
    meta = {}
    if meta_path.exists() and filename == "latest_top_picks.csv":
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    elif filename != "latest_top_picks.csv":
        meta = {"timestamp": filename.removeprefix("top_picks_").removesuffix(".csv")}
    return frame, meta


def list_top_pick_snapshots() -> list[str]:
    ensure_store_dirs()
    files = sorted(
        [path.name for path in TOP_DIR.glob("top_picks_*.csv")],
        reverse=True,
    )
    return ["latest_top_picks.csv"] + files if (TOP_DIR / "latest_top_picks.csv").exists() else files


def save_theme_snapshot(summary_df: pd.DataFrame, rankings: dict[str, pd.DataFrame], meta: dict[str, Any]) -> tuple[Path, Path]:
    ensure_store_dirs()
    stamp = meta.get("timestamp", pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
    summary_path = THEME_DIR / f"themes_summary_{stamp}.csv"
    ranking_path = THEME_DIR / f"themes_rankings_{stamp}.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(THEME_DIR / "latest_themes_summary.csv", index=False, encoding="utf-8-sig")
    ranking_rows: list[pd.DataFrame] = []
    for theme, frame in rankings.items():
        copy = frame.copy()
        copy.insert(0, "主线主题", theme)
        ranking_rows.append(copy)
    ranking_df = pd.concat(ranking_rows, ignore_index=True) if ranking_rows else pd.DataFrame()
    ranking_df.to_csv(ranking_path, index=False, encoding="utf-8-sig")
    ranking_df.to_csv(THEME_DIR / "latest_themes_rankings.csv", index=False, encoding="utf-8-sig")
    (THEME_DIR / "latest_themes.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path, ranking_path


def load_theme_snapshot(summary_filename: str = "latest_themes_summary.csv", ranking_filename: str | None = None) -> tuple[pd.DataFrame | None, dict[str, pd.DataFrame], dict[str, Any]]:
    ensure_store_dirs()
    summary_path = THEME_DIR / summary_filename
    ranking_path = THEME_DIR / (ranking_filename or ("latest_themes_rankings.csv" if summary_filename == "latest_themes_summary.csv" else summary_filename.replace("summary", "rankings")))
    if not summary_path.exists():
        return None, {}, {}
    try:
        summary_df = pd.read_csv(summary_path)
    except Exception:
        return None, {}, {}
    rankings: dict[str, pd.DataFrame] = {}
    if ranking_path.exists():
        try:
            ranking_df = pd.read_csv(ranking_path)
            if "主线主题" in ranking_df.columns:
                for theme, frame in ranking_df.groupby("主线主题", dropna=False):
                    rankings[str(theme)] = frame.drop(columns=["主线主题"]).reset_index(drop=True)
        except Exception:
            rankings = {}
    meta = {}
    meta_path = THEME_DIR / "latest_themes.json"
    if meta_path.exists() and summary_filename == "latest_themes_summary.csv":
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    elif summary_filename != "latest_themes_summary.csv":
        meta = {"timestamp": summary_filename.removeprefix("themes_summary_").removesuffix(".csv")}
    return summary_df, rankings, meta


def list_theme_snapshots() -> list[str]:
    ensure_store_dirs()
    files = sorted(
        [path.name for path in THEME_DIR.glob("themes_summary_*.csv")],
        reverse=True,
    )
    return ["latest_themes_summary.csv"] + files if (THEME_DIR / "latest_themes_summary.csv").exists() else files
