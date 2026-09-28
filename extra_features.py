"""Additional in-window User-Agent and search-query features."""
from __future__ import annotations
import pandas as pd


def make_ua_query_features(cookies: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    ids = pd.Index(cookies.cookie_id, name="cookie_id")
    bounds = cookies.set_index("cookie_id")[["window_start_ts", "window_end_ts"]].apply(pd.to_datetime)
    e = events.join(bounds, on="cookie_id", how="inner")
    e["event_ts"] = pd.to_datetime(e.event_ts)
    e = e.loc[(e.event_ts >= e.window_start_ts) & (e.event_ts < e.window_end_ts)].copy()
    e = e.sort_values(["cookie_id", "event_ts"], kind="stable")
    out = pd.DataFrame(index=ids)
    ua = e.user_agent.fillna("").str.lower()
    flags = {"headless": "headless", "selenium": "selenium",
             "playwright": "playwright", "requests": "requests",
             "python": "python", "curl": "curl", "scrapy": "scrapy",
             "avito_app": "avito"}
    for name, token in flags.items():
        out[f"ua_{name}_share"] = ua.str.contains(token, regex=False).groupby(e.cookie_id).mean()
    changed = ua.ne(ua.groupby(e.cookie_id).shift()) & e.groupby("cookie_id").cumcount().gt(0)
    out["ua_changes"] = changed.groupby(e.cookie_id).sum()
    query = e.search_query.dropna().astype(str)
    if not query.empty:
        qid = e.loc[query.index, "cookie_id"]
        length = query.str.len().clip(lower=1)
        out["query_digit_share"] = query.str.count(r"\d").div(length).groupby(qid).mean()
        out["query_symbol_share"] = query.str.count(r"[^\w\s]").div(length).groupby(qid).mean()
        out["query_length_cv"] = query.str.len().groupby(qid).std() / query.str.len().groupby(qid).mean()
    return out
