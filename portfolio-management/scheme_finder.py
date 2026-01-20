
import json
import os
import time
from typing import List, Dict, Any
from difflib import SequenceMatcher

import pandas as pd
import requests
import streamlit as st

MF_MASTER_URL = "https://api.mfapi.in/mf"


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


@st.cache_data(show_spinner=False)
def _load_master_cached(cache_path: str, max_age_hours: int) -> List[Dict[str, Any]]:
    """
    Streamlit-cached wrapper. Internally uses a file cache too.
    """
    # File cache first (persists across reruns)
    if os.path.exists(cache_path):
        age_seconds = time.time() - os.path.getmtime(cache_path)
        if age_seconds < max_age_hours * 3600:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

    # Fetch fresh
    data = requests.get(MF_MASTER_URL, timeout=30).json()

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return data


def find_schemes(query: str, schemes: List[Dict[str, Any]], top_n: int = 20) -> pd.DataFrame:
    q = (query or "").strip().lower()
    if not q:
        return pd.DataFrame(columns=["schemeCode", "schemeName", "score"])

    rows = []
    for s in schemes:
        name = str(s.get("schemeName", ""))
        code = s.get("schemeCode", None)

        name_l = name.lower()
        contains = 1.0 if q in name_l else 0.0
        score = max(_similarity(q, name_l), contains)

        rows.append({"schemeCode": code, "schemeName": name, "score": float(score)})

    df = pd.DataFrame(rows)
    df = df.sort_values(["score", "schemeName"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    return df


def render_scheme_finder(cache_path: str = "mf_schemes_cache.json", max_age_hours: int = 24) -> None:
    st.subheader("Find Mutual Fund Scheme Code")
    st.caption("Search mfapi.in master list by scheme name (full or partial).")

    colA, colB = st.columns([1, 1])
    with colA:
        query = st.text_input("Enter MF scheme name", value="Aditya Birla Sun Life Liquid Fund")
    with colB:
        refresh = st.button("Refresh master list")

    if refresh:
        # Force refresh by deleting file cache; Streamlit cache will rebuild
        if os.path.exists(cache_path):
            os.remove(cache_path)
        st.cache_data.clear()
        st.success("Cache cleared. Click Search to fetch fresh list.")

    top_n = st.number_input("Top results", min_value=5, max_value=50, value=20, step=5)
    min_score = st.slider("Minimum match score", min_value=0.0, max_value=1.0, value=0.55, step=0.05)

    if st.button("Search"):
        with st.spinner("Loading scheme master..."):
            schemes = _load_master_cached(cache_path=cache_path, max_age_hours=max_age_hours)

        df = find_schemes(query=query, schemes=schemes, top_n=int(top_n))
        df = df[df["score"] >= float(min_score)].copy()

        if df.empty:
            st.warning("No matches found. Try a shorter/alternate name.")
        else:
            st.dataframe(df, use_container_width=True)
            st.info("Copy the schemeCode above and paste it into the Mutual Fund Portfolio table.")