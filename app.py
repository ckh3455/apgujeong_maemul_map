import os
import json
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# =============================
# 기본 설정
# =============================
st.set_page_config(page_title="압구정 지도+활성매물+거래내역", layout="wide")

DEFAULT_CENTER = [37.5275, 127.0300]
DEFAULT_ZOOM = 15

TAB_LISTING = "매매물건 목록"
TAB_LOC = "압구정 위치"


# =============================
# 유틸
# =============================
def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def norm_area(x):
    """구역값을 비교하기 위한 정규화: 숫자만 남겨서 '1', '2' 형태로."""
    s = norm_text(x)
    if not s:
        return ""
    s = re.sub(r"[^\d]", "", s)
    return s


def parse_price_to_num(x):
    """
    '가격' 텍스트를 숫자로 변환(만원/억 등 혼합 텍스트를 최대한 견고하게 처리).
    기존 동작을 유지하기 위해 원본 로직을 건드리지 않는 범위에서만 사용됨.
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan

    s = s.replace(",", "")
    # '억' 단위 처리
    m = re.search(r"(\d+(?:\.\d+)?)\s*억", s)
    if m:
        try:
            eok = float(m.group(1))
        except Exception:
            eok = np.nan
        # '억' 뒤의 '천/만' 보정(간단 처리)
        rest = s[m.end():]
        rest = rest.replace("만원", "").replace("만", "").replace("천", "")
        rest = re.sub(r"[^\d\.]", "", rest)
        try:
            man = float(rest) if rest else 0.0
        except Exception:
            man = 0.0
        # 억을 '만원' 기준으로 환산: 1억 = 10,000만원
        return eok * 10000 + man

    # 만원 단위
    s2 = s.replace("만원", "").replace("만", "")
    s2 = re.sub(r"[^\d\.]", "", s2)
    try:
        return float(s2)
    except Exception:
        return np.nan


def price_display_eok(manwon):
    """만원 단위 숫자를 'x.xx억' 표시 문자열로."""
    if pd.isna(manwon):
        return ""
    try:
        manwon = float(manwon)
    except Exception:
        return ""
    eok = manwon / 10000.0
    return f"{eok:.2f}"


def dataframe_height(df, row_height=35, header_height=40, max_height=520):
    if df is None or len(df) == 0:
        return 120
    h = header_height + len(df) * row_height
    return int(min(max_height, max(160, h)))


def resolve_clicked_meta(lat, lng, marker_rows, eps=1e-9):
    """
    st_folium 클릭 좌표와 marker_rows(리스트[dict])의 위경도를 매칭.
    """
    if marker_rows is None:
        return None
    for r in marker_rows:
        try:
            if abs(float(r["위도"]) - float(lat)) < eps and abs(float(r["경도"]) - float(lng)) < eps:
                return r
        except Exception:
            continue
    return None


def recent_trades(df_trade, area, complex_name, pyeong, n=5):
    """구역/단지/평형이 일치하는 최신 거래 n건."""
    if df_trade is None or df_trade.empty:
        return pd.DataFrame()

    area_norm = norm_area(area)
    complex_norm = norm_text(complex_name)
    pyeong_norm = norm_text(pyeong)

    dft = df_trade.copy()

    # 구역 비교(정규화)
    if "구역" in dft.columns:
        dft["_area_norm"] = dft["구역"].astype(str).map(norm_area)
        dft = dft[dft["_area_norm"] == area_norm]

    # 단지명
    if "단지명" in dft.columns:
        dft = dft[dft["단지명"].astype(str).str.strip() == complex_norm]

    # 평형
    if "평형" in dft.columns:
        dft = dft[dft["평형"].astype(str).str.strip() == pyeong_norm]
    elif "평형대" in dft.columns:
        dft = dft[dft["평형대"].astype(str).str.strip() == pyeong_norm]

    # 날짜 정렬
    date_col = None
    for c in ["거래일", "계약일", "날짜", "일자", "거래일자"]:
        if c in dft.columns:
            date_col = c
            break

    if date_col:
        dft["_dt"] = pd.to_datetime(dft[date_col], errors="coerce")
        dft = dft.sort_values("_dt", ascending=False)
    else:
        dft = dft.copy()

    # 표시 컬럼
    show_cols = [c for c in ["거래일", "계약일", "일자", "가격", "층", "면적", "평형", "평형대"] if c in dft.columns]
    if not show_cols:
        show_cols = list(dft.columns)

    return dft[show_cols].head(n)


def summarize_area_by_size(df_view, area):
    """선택 구역 내 활성매물(매매) 평형별 요약."""
    if df_view is None or df_view.empty:
        return pd.DataFrame()

    area_norm = norm_area(area)
    dfx = df_view.copy()
    dfx["_area_norm"] = dfx["구역"].astype(str).map(norm_area)
    dfx = dfx[dfx["_area_norm"] == area_norm].copy()

    if dfx.empty:
        return pd.DataFrame()

    # 가격 숫자
    if "가격_num" not in dfx.columns:
        dfx["가격_num"] = dfx["가격"].apply(parse_price_to_num)

    # 평형: 우선 '평형' 사용, 없으면 '평형대'
    if "평형" in dfx.columns:
        gcol = "평형"
    elif "평형대" in dfx.columns:
        gcol = "평형대"
    else:
        return pd.DataFrame()

    grp = dfx.groupby(gcol, dropna=False)
    out = grp["가격_num"].agg(["count", "min", "max"]).reset_index()
    out = out.rename(columns={gcol: "평형", "count": "매물건수", "min": "최저가격", "max": "최고가격"})
    out["가격대(최저~최고)"] = out.apply(
        lambda r: f"{price_display_eok(r['최저가격'])}~{price_display_eok(r['최고가격'])}", axis=1
    )
    out = out.sort_values("평형")
    return out


# =============================
# 데이터 로드/전처리 (요청에 따라 건드리지 않음)
# =============================
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path(".").resolve()
DATA_DIR = BASE_DIR

LISTING_FILE = DATA_DIR / "listing.csv"
TRADE_FILE = DATA_DIR / "trades.csv"
GEO_FILE = DATA_DIR / "geo.json"

@st.cache_data(show_spinner=False)
def load_data(listing_path: Path, trade_path: Path):
    df_listing = pd.read_csv(listing_path, dtype=str) if listing_path.exists() else pd.DataFrame()
    df_trade = pd.read_csv(trade_path, dtype=str) if trade_path.exists() else pd.DataFrame()
    return df_listing, df_trade

df_listing, df_trade = load_data(LISTING_FILE, TRADE_FILE)

# 전처리
df_view = df_listing.copy()

# 필수 컬럼 보정(없으면 생성)
for c in ["위도", "경도", "단지명", "동", "층/호", "가격", "요약내용", "부동산", "상태", "구역"]:
    if c not in df_view.columns:
        df_view[c] = ""

# 동_key
if "동_key" not in df_view.columns:
    if "동" in df_view.columns:
        df_view["동_key"] = df_view["동"].astype(str).str.strip()
    else:
        df_view["동_key"] = ""

# 평형대_bucket
if "평형대_bucket" not in df_view.columns:
    if "평형대" in df_view.columns:
        def _bucket(x):
            s = norm_text(x)
            s = re.sub(r"[^\d]", "", s)
            try:
                v = int(s)
            except Exception:
                return np.nan
            # 20,30,40,... 10단위 버킷
            return int(v / 10) * 10
        df_view["평형대_bucket"] = df_view["평형대"].apply(_bucket)
    else:
        df_view["평형대_bucket"] = np.nan

# 가격_num, 가격(억)표시
if "가격_num" not in df_view.columns:
    df_view["가격_num"] = df_view["가격"].apply(parse_price_to_num)
if "가격(억)표시" not in df_view.columns:
    df_view["가격(억)표시"] = df_view["가격_num"].apply(price_display_eok)

# 위경도 숫자
df_view["위도"] = pd.to_numeric(df_view["위도"], errors="coerce")
df_view["경도"] = pd.to_numeric(df_view["경도"], errors="coerce")

df_view = df_view[df_view["위도"].notna() & df_view["경도"].notna()].copy()

# =============================
# 세션 상태
# =============================
if "map_center" not in st.session_state:
    st.session_state["map_center"] = DEFAULT_CENTER
if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = DEFAULT_ZOOM
if "selected_meta" not in st.session_state:
    st.session_state["selected_meta"] = None
if "last_click_sig" not in st.session_state:
    st.session_state["last_click_sig"] = ""
if "quick_filter_mode" not in st.session_state:
    st.session_state["quick_filter_mode"] = "size"
if "quick_filter_bucket" not in st.session_state:
    st.session_state["quick_filter_bucket"] = 30
if "last_table_sel_sig" not in st.session_state:
    st.session_state["last_table_sel_sig"] = ""


# =============================
# 지도 생성
# =============================
m = folium.Map(location=st.session_state["map_center"], zoom_start=st.session_state["map_zoom"], control_scale=True)

marker_rows = []
cluster = MarkerCluster(name="매물").add_to(m)

for _, r in df_view.iterrows():
    try:
        lat = float(r["위도"])
        lng = float(r["경도"])
    except Exception:
        continue

    area = norm_text(r.get("구역", ""))
    complex_name = norm_text(r.get("단지명", ""))
    dong = norm_text(r.get("동_key", ""))

    tooltip = f"{complex_name} {dong} / {area}"
    popup = folium.Popup(tooltip, max_width=400)

    folium.Marker(
        location=[lat, lng],
        tooltip=tooltip,
        popup=popup,
    ).add_to(cluster)

    marker_rows.append(
        {
            "위도": lat,
            "경도": lng,
            "구역": r.get("구역", ""),
            "단지명": complex_name,
            "동_key": dong,
        }
    )

folium.LayerControl(collapsed=True).add_to(m)


# =============================
# UI 레이아웃: 상하 배치로 변경
# 1) 지도(상단)
# 2) 선택한 동의 활성매물
# 3) 거래내역 최신 5건
# 4) (하단) 좌: 선택구역 평형별 요약 / 우: 빠른필터(선택구역 내)
# =============================

st.subheader("지도")
out = st_folium(
    m,
    height=650,
    width=None,
    returned_objects=["last_object_clicked"],
    key="map",
)

if out:
    clicked = out.get("last_object_clicked", None)
    if clicked:
        lat = clicked.get("lat")
        lng = clicked.get("lng")
        if lat is not None and lng is not None:
            click_sig = f"{round(float(lat), 6)},{round(float(lng), 6)}"
            if st.session_state["last_click_sig"] != click_sig:
                meta = resolve_clicked_meta(lat, lng, marker_rows)
                if meta:
                    st.session_state["selected_meta"] = meta
                    st.session_state["map_center"] = [float(meta["위도"]), float(meta["경도"])]
                    st.session_state["map_zoom"] = int(st.session_state.get("map_zoom") or DEFAULT_ZOOM)
                    st.session_state["last_click_sig"] = click_sig
                    st.rerun()

meta = st.session_state.get("selected_meta", None)
if not meta:
    st.info("지도에서 마커를 클릭하면 아래에 상세가 표시됩니다.")
    st.stop()

complex_name = meta["단지명"]
dong = meta["동_key"]
area_value = str(meta["구역"]) if pd.notna(meta["구역"]) else ""
area_norm = norm_area(area_value)

df_pick = df_view[(df_view["단지명"] == complex_name) & (df_view["동_key"] == dong)].copy()

# -----------------------------
# (2) 선택한 동의 활성매물
# -----------------------------
st.subheader("선택한 동의 활성매물")

# - '평형대','구역' 제거
# - '부동산'을 '요약내용' 앞에 배치
show_cols = ["단지명", "평형", "대지지분", "동", "층/호", "가격", "부동산", "요약내용", "상태"]
show_cols = [c for c in show_cols if c in df_pick.columns]
view_pick = df_pick[show_cols].reset_index(drop=True)

col_cfg = None
try:
    col_cfg = {
        "단지명": st.column_config.TextColumn("단지명", width="small"),
        "평형": st.column_config.TextColumn("평형", width="small"),
        "대지지분": st.column_config.TextColumn("대지지분", width="small"),
        "동": st.column_config.TextColumn("동", width="small"),
        "층/호": st.column_config.TextColumn("층/호", width="small"),
        "가격": st.column_config.NumberColumn("가격", width="small"),
        "부동산": st.column_config.TextColumn("부동산", width="small"),
        "요약내용": st.column_config.TextColumn("요약내용", width="large"),
        "상태": st.column_config.TextColumn("상태", width="small"),
    }
    col_cfg = {k: v for k, v in col_cfg.items() if k in view_pick.columns}
except Exception:
    col_cfg = None

try:
    st.dataframe(
        view_pick,
        use_container_width=True,
        height=dataframe_height(view_pick, max_height=520),
        column_config=col_cfg,
    )
except TypeError:
    st.dataframe(
        view_pick,
        use_container_width=True,
        height=dataframe_height(view_pick, max_height=520),
    )

st.divider()

# -----------------------------
# (3) 거래내역 최신 5건
# -----------------------------
st.subheader("거래내역 최신 5건 (구역/단지/평형 일치)")

pyeong_candidates = []
if "평형" in df_pick.columns:
    pyeong_candidates = sorted(df_pick["평형"].astype(str).str.strip().dropna().unique().tolist())
elif "평형대" in df_pick.columns:
    pyeong_candidates = sorted(df_pick["평형대"].astype(str).str.strip().dropna().unique().tolist())

if not pyeong_candidates:
    st.info("선택한 동에서 평형 정보를 찾을 수 없습니다.")
else:
    sel_key = f"sel_pyeong_{norm_text(complex_name)}_{dong}"
    sel_pyeong = st.selectbox("평형 선택", pyeong_candidates, index=0, key=sel_key)

    trades = recent_trades(df_trade, area_value, complex_name, sel_pyeong)  # 함수는 이미 head(5)
    if trades.empty:
        st.info("일치하는 거래내역이 없습니다.")
    else:
        trades2 = trades.reset_index(drop=True)
        styled = trades2.style.set_properties(**{"color": "red"})
        try:
            styled = styled.hide(axis="index")
        except Exception:
            pass

        st.dataframe(
            styled,
            use_container_width=True,
            height=dataframe_height(trades2, max_height=280),
        )

st.divider()

# -----------------------------
# (4) 하단: 좌/우 분할
# -----------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("선택구역 평형별 요약 (활성 매물)")
    if not area_value:
        st.info("선택한 마커의 구역 정보가 없습니다.")
    else:
        summary = summarize_area_by_size(df_view, area_value)
        if summary.empty:
            st.info("해당 구역에서 요약할 데이터가 없습니다.")
        else:
            summary_view = summary[["평형", "매물건수", "가격대(최저~최고)", "최저가격", "최고가격"]].reset_index(drop=True)
            st.dataframe(
                summary_view,
                use_container_width=True,
                height=dataframe_height(summary_view, max_height=420),
            )

with col_right:
    st.subheader("빠른필터 (활성매물)")

    if not area_value:
        st.info("구역 정보가 없어 빠른필터를 표시할 수 없습니다.")
    else:
        c0, c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1, 1, 1, 1, 1, 1, 1.2])

        buckets = [20, 30, 40, 50, 60, 70, 80]
        cols = [c0, c1, c2, c3, c4, c5, c6]
        for col, b in zip(cols, buckets):
            if col.button(f"{b}평대", use_container_width=True):
                st.session_state["quick_filter_mode"] = "size"
                st.session_state["quick_filter_bucket"] = b
                st.rerun()

        if c7.button("가격순", use_container_width=True):
            st.session_state["quick_filter_mode"] = "price"
            st.rerun()

        mode = st.session_state["quick_filter_mode"]

        area_display = f"{area_norm}구역" if area_norm else str(area_value).strip()
        if mode == "size":
            st.caption(f"현재: {area_display} / {st.session_state['quick_filter_bucket']}평대 (가격 낮은 순)")
        else:
            st.caption(f"현재: {area_display} / 전체 (가격 낮은 순)")

        # ---- 핵심 변경: '선택된 구역' 내 매물만 필터링 ----
        dfq = df_view.copy()
        dfq = dfq[dfq["가격_num"].notna()].copy()

        dfq["_area_norm"] = dfq["구역"].astype(str).map(norm_area)
        dfq = dfq[dfq["_area_norm"] == area_norm].copy()

        if mode == "size":
            b = st.session_state["quick_filter_bucket"]
            dfq = dfq[dfq["평형대_bucket"] == b].copy()

        dfq = dfq.sort_values("가격_num", ascending=True).reset_index(drop=True)

        if dfq.empty:
            st.info("조건에 맞는 매물이 없습니다.")
        else:
            display_cols = ["구역", "평형대", "단지명", "동", "층/호", "가격(억)표시", "요약내용", "부동산"]
            display_cols = [c for c in display_cols if c in dfq.columns]

            df_show = dfq[display_cols + ["위도", "경도", "동_key", "가격_num"]].copy().reset_index(drop=True)
            df_table = df_show[display_cols].copy().rename(columns={"가격(억)표시": "가격(억)"})

            st.markdown("표에서 행을 클릭하면 해당 동 위치로 지도가 이동합니다.")

            event = st.dataframe(
                df_table,
                height=dataframe_height(df_table, max_height=520),
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row",
            )

            try:
                if event and event.selection and event.selection.rows:
                    ridx = event.selection.rows[0]
                    row = df_show.iloc[ridx]

                    sel_sig = f"{row.get('단지명','')}|{row.get('동_key','')}|{row.get('평형대','')}|{row.get('가격_num','')}"
                    if st.session_state["last_table_sel_sig"] != sel_sig:
                        st.session_state["map_center"] = [float(row["위도"]), float(row["경도"])]
                        st.session_state["map_zoom"] = int(st.session_state.get("map_zoom") or DEFAULT_ZOOM)

                        st.session_state["selected_meta"] = {
                            "단지명": row["단지명"],
                            "동_key": row["동_key"],
                            "구역": row["구역"],
                            "위도": float(row["위도"]),
                            "경도": float(row["경도"]),
                        }
                        st.session_state["last_table_sel_sig"] = sel_sig
                        st.session_state["last_click_sig"] = ""
                        st.rerun()
            except Exception:
                pass
