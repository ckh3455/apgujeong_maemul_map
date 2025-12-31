import os
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

import folium
from streamlit_folium import st_folium

import gspread
from google.oauth2.service_account import Credentials


# ====== 로컬 개발용(배포에서는 Secrets 사용 권장) ======
SERVICE_ACCOUNT_FILE = r"D:\OneDrive\office work\naver crawling\naver-crawling-476404-fcf4b10bc63e 클라우드 서비스계정.txt"
SPREADSHEET_ID_DEFAULT = "1QP56lm5kPBdsUhrgcgY2U-JdmukXIkKCSxefd1QExKE"

TAB_LISTING = "매매물건 목록"
TAB_LOC = "압구정 위치정보"
TAB_TRADE = "거래내역"
# =====================================================


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\n", "").strip() for c in df.columns]
    return df


def dong_key(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    m = re.findall(r"\d+", s)
    return m[0].lstrip("0") if m else s.strip()


def norm_area(x) -> str:
    """'1', '1구역', '01구역' => '1' 로 통일"""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    m = re.findall(r"\d+", s)
    if not m:
        return s
    return m[0].lstrip("0") or "0"


def norm_text(x: str) -> str:
    """단지명 비교용 정규화"""
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = s.replace("아파트", "").replace("apt", "").replace("apartment", "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[(){}\[\]\-_/·.,]", "", s)
    return s


def norm_size(x: str) -> str:
    """평형 비교용 정규화"""
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = s.replace("㎡", "").replace("m2", "").replace("m²", "").replace("평", "")
    s = re.sub(r"\s+", "", s)
    return s


def parse_pyeong_num(x) -> float | None:
    """'35평', '35', '35.5평' -> 35.5"""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def pyeong_bucket_10(pyeong: float | None) -> int | None:
    """35.5 -> 30 (30평대). NaN 안전 처리."""
    if pyeong is None or pd.isna(pyeong):
        return None
    return int(float(pyeong) // 10) * 10


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def fmt_decimal(x, nd=2) -> str:
    """59.500000 -> 59.5 / 62.100000 -> 62.1"""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    num = pd.to_numeric(x, errors="coerce")
    if pd.isna(num):
        return str(x)
    return f"{num:.{nd}f}".rstrip("0").rstrip(".")


def dataframe_height(df: pd.DataFrame, max_height: int = 700, row_height: int = 34, header_height: int = 42) -> int:
    """가능하면 스크롤 없이 보이도록 DataFrame 행 수에 맞춰 높이 계산(상한 max_height)"""
    n = 0 if df is None else int(len(df))
    h = header_height + (n * row_height)
    return max(160, min(h, max_height))


def st_df(obj, **kwargs):
    """
    Streamlit 버전에 따라 hide_index 지원 여부가 달라서 안전하게 처리.
    - 지원하면 hide_index=True 적용
    - 미지원이면 기존 동작 유지 (인덱스는 CSS로 숨김)
    """
    try:
        return st.dataframe(obj, hide_index=True, **kwargs)
    except TypeError:
        return st.dataframe(obj, **kwargs)


def to_eok_display(value) -> str:
    """원 단위면 억으로 환산, 이미 억이면 그대로"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return ""
    if num >= 1e8:
        num = num / 1e8
    return fmt_decimal(num, nd=2)


def make_circle_label_html(label: str, bg_color: str) -> str:
    size = 30
    return f"""
    <div style="
        background:{bg_color};
        width:{size}px;height:{size}px;
        border-radius:50%;
        border:2px solid rgba(0,0,0,0.45);
        display:flex;align-items:center;justify-content:center;
        font-weight:700;font-size:14px;
        color:#ffffff;
        box-shadow:0 1px 4px rgba(0,0,0,0.35);
        ">
        {label}
    </div>
    """


def pick_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def extract_spreadsheet_id(url_or_id: str) -> str:
    """스프레드시트 URL 또는 ID에서 ID만 추출"""
    if not url_or_id:
        return ""
    s = str(url_or_id).strip()
    if "docs.google.com" not in s and "/" not in s and len(s) >= 20:
        return s
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", s)
    if m:
        return m.group(1)
    s = s.split("#", 1)[0]
    return s


def get_spreadsheet_id() -> str:
    if "SPREADSHEET_ID" in st.secrets:
        return extract_spreadsheet_id(st.secrets["SPREADSHEET_ID"])

    if "connections" in st.secrets and "gsheets" in st.secrets["connections"]:
        gs = st.secrets["connections"]["gsheets"]
        if "spreadsheet" in gs:
            sid = extract_spreadsheet_id(gs["spreadsheet"])
            if sid:
                return sid

    env = os.getenv("SPREADSHEET_ID")
    if env:
        return extract_spreadsheet_id(env)

    return SPREADSHEET_ID_DEFAULT


def get_service_account_info():
    if "GCP_SERVICE_ACCOUNT_JSON" in st.secrets:
        v = st.secrets["GCP_SERVICE_ACCOUNT_JSON"]
        if isinstance(v, dict):
            return v
        return json.loads(str(v))

    if "connections" in st.secrets and "gsheets" in st.secrets["connections"]:
        gs = st.secrets["connections"]["gsheets"]
        keys = [
            "type", "project_id", "private_key_id", "private_key",
            "client_email", "client_id",
            "auth_uri", "token_uri",
            "auth_provider_x509_cert_url", "client_x509_cert_url",
        ]
        sa = {k: gs[k] for k in keys if k in gs}
        required = ["type", "project_id", "private_key", "client_email", "token_uri"]
        if all(k in sa and str(sa[k]).strip() for k in required):
            return sa
        raise RuntimeError(
            "Streamlit Secrets의 [connections.gsheets]에 서비스계정 필수 항목이 부족합니다. "
            "type/project_id/private_key/client_email/token_uri를 확인하세요."
        )

    env = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if env:
        return json.loads(env)

    p = Path(SERVICE_ACCOUNT_FILE)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8").strip())

    raise RuntimeError(
        "서비스계정 Secrets가 설정되지 않았습니다. "
        "Streamlit Cloud > Manage app > Settings > Secrets에 "
        "GCP_SERVICE_ACCOUNT_JSON 또는 [connections.gsheets]를 등록하세요."
    )


@st.cache_data(ttl=600)
def load_data():
    sa = get_service_account_info()
    spreadsheet_id = get_spreadsheet_id()

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(sa, scopes=scopes)
    gc = gspread.authorize(creds)

    sh = gc.open_by_key(spreadsheet_id)

    ws_list = sh.worksheet(TAB_LISTING)
    ws_loc = sh.worksheet(TAB_LOC)

    df_list = pd.DataFrame(ws_list.get_all_records())
    df_loc = pd.DataFrame(ws_loc.get_all_records())

    try:
        ws_trade = sh.worksheet(TAB_TRADE)
        df_trade = pd.DataFrame(ws_trade.get_all_records())
    except Exception:
        df_trade = pd.DataFrame()

    return clean_columns(df_list), clean_columns(df_loc), clean_columns(df_trade), sa.get("client_email", "")


def build_grouped(df_active: pd.DataFrame) -> pd.DataFrame:
    g = (
        df_active.groupby(["단지명", "동_key"], dropna=False)
        .agg(
            구역=("구역", "first"),
            위도=("위도", "first"),
            경도=("경도", "first"),
            활성건수=("동_key", "size"),
        )
        .reset_index()
    )
    return g


def summarize_area_by_size(df_active: pd.DataFrame, area_value: str) -> pd.DataFrame:
    if not area_value:
        return pd.DataFrame()

    target = norm_area(area_value)
    df_area = df_active.copy()
    df_area["_area_norm"] = df_area["구역"].astype(str).map(norm_area)
    df_area = df_area[df_area["_area_norm"] == target].copy()
    if df_area.empty:
        return pd.DataFrame()

    size_key = "평형대" if "평형대" in df_area.columns else ("평형" if "평형" in df_area.columns else None)
    if not size_key:
        return pd.DataFrame()

    df_area["가격_num"] = pd.to_numeric(df_area["가격"], errors="coerce")
    s = (
        df_area.groupby(size_key, dropna=False)
        .agg(
            매물건수=("가격_num", "size"),
            최저가격=("가격_num", "min"),
            최고가격=("가격_num", "max"),
        )
        .reset_index()
        .rename(columns={size_key: "평형"})
    )

    s["평형_sort"] = s["평형"].astype(str)
    s = s.sort_values(by="평형_sort").drop(columns=["평형_sort"]).reset_index(drop=True)

    for c in ["최저가격", "최고가격"]:
        s[c] = s[c].round(0)

    s["가격대(최저~최고)"] = (
        s["최저가격"].fillna(0).astype(int).astype(str) + " ~ " + s["최고가격"].fillna(0).astype(int).astype(str)
    )
    return s


def recent_trades(df_trade: pd.DataFrame, area: str, complex_name: str, pyeong_value: str) -> pd.DataFrame:
    if df_trade is None or df_trade.empty:
        return pd.DataFrame()

    col_area = pick_first_existing_column(df_trade, ["구역"])
    col_complex = pick_first_existing_column(df_trade, ["단지", "단지명", "단지명(단지)"])
    col_size = pick_first_existing_column(df_trade, ["평형", "평형대"])
    col_date = pick_first_existing_column(df_trade, ["날짜", "거래일", "계약일", "일자", "거래일자"])
    if not (col_area and col_complex and col_size and col_date):
        return pd.DataFrame()

    t = df_trade.copy()
    t["_area_norm"] = t[col_area].astype(str).map(norm_area)
    t["_complex_norm"] = t[col_complex].astype(str).map(norm_text)
    t["_size_norm"] = t[col_size].astype(str).map(norm_size)

    area_norm = norm_area(area)
    complex_norm = norm_text(complex_name)
    size_norm = norm_size(pyeong_value)

    t = t[(t["_area_norm"] == area_norm) & (t["_complex_norm"] == complex_norm) & (t["_size_norm"] == size_norm)].copy()
    if t.empty:
        return pd.DataFrame()

    t["_dt"] = pd.to_datetime(t[col_date], errors="coerce", format="%y.%m.%d")
    t = t.dropna(subset=["_dt"]).sort_values("_dt", ascending=False).head(5).copy()

    price_col = pick_first_existing_column(t, ["가격", "거래가격", "거래가", "실거래가", "금액", "거래금액"])
    if price_col:
        t["가격(억)"] = t[price_col].map(to_eok_display)

    preferred = [col_date, col_area, col_complex, col_size]
    if "가격(억)" in t.columns:
        preferred.append("가격(억)")

    for extra in ["동", "호", "비고"]:
        if extra in t.columns and extra not in preferred:
            preferred.append(extra)

    out = t[preferred].copy()
    out[col_area] = out[col_area].astype(str).map(lambda v: f"{norm_area(v)}구역" if norm_area(v) else str(v).strip())
    return out


def resolve_clicked_meta(clicked_lat, clicked_lng, marker_rows):
    """가장 가까운 마커로 매칭(미세 좌표 차이/히트박스 문제 완화)"""
    if clicked_lat is None or clicked_lng is None:
        return None
    clat = float(clicked_lat)
    clng = float(clicked_lng)

    best_meta = None
    best_d = None
    for lat, lng, meta in marker_rows:
        d = (float(lat) - clat) ** 2 + (float(lng) - clng) ** 2
        if best_d is None or d < best_d:
            best_d = d
            best_meta = meta
    return best_meta


# =================== UI ===================
st.set_page_config(layout="wide")
st.title("압구정 매물 지도 MVP (상태=활성, 수동 갱신)")

# 행번호(인덱스) 영역 강제 숨김 (Streamlit 버전 차이 대응)
st.markdown(
    """
<style>
/* st.dataframe / st.data_editor 공통: row header(행번호) 숨김 */
div[data-testid="stDataFrame"] div[role="rowheader"],
div[data-testid="stDataFrame"] div[role="rowheader"] * {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    min-width: 0 !important;
}
/* 좌상단 빈 코너(인덱스 헤더) 숨김 */
div[data-testid="stDataFrame"] .blank,
div[data-testid="stDataFrame"] .row_heading {
    display: none !important;
}
/* 일부 버전에서 남는 여백 최소화 */
div[data-testid="stDataFrame"] div[role="grid"] {
    padding-left: 0 !important;
}
</style>
    """,
    unsafe_allow_html=True,
)

st.subheader("필터")
only_active = st.checkbox("상태=활성만 표시", value=True)

col_a, col_b = st.columns(2)
with col_a:
    if st.button("데이터 새로고침"):
        load_data.clear()
        st.rerun()
with col_b:
    if st.button("캐시만 비우기"):
        load_data.clear()
        st.success("캐시를 비웠습니다. (다음 실행 때 새로 로드)")

st.caption("지도는 클릭 이벤트만 수신(드래그/줌 시 자동 새로고침 없음).")

# ====== Load ======
df_list, df_loc, df_trade, client_email = load_data()
if client_email:
    st.caption(f"서비스계정: {client_email}")

# 층/호 보정
if "층/호" not in df_list.columns and "층수" in df_list.columns:
    df_list = df_list.copy()
    df_list["층/호"] = df_list["층수"]

# --- 요약내용 컬럼명 표준화/보장 ---
df_list = df_list.copy()
rename_map = {}
for c in df_list.columns:
    c0 = str(c)
    c_norm = re.sub(r"\s+", "", c0)  # 공백 제거
    if c_norm == "요약내용" or ("요약" in c_norm and "내용" in c_norm):
        rename_map[c] = "요약내용"

if rename_map:
    df_list.rename(columns=rename_map, inplace=True)

summary_src = pick_first_existing_column(df_list, ["요약내용", "요약 내용", "요약", "설명", "비고", "메모"])
if "요약내용" not in df_list.columns:
    df_list["요약내용"] = df_list[summary_src] if summary_src else ""
elif summary_src and summary_src != "요약내용":
    left = df_list["요약내용"].astype(str).str.strip()
    df_list.loc[left.eq(""), "요약내용"] = df_list.loc[left.eq(""), summary_src]
# -------------------------------

need_cols = ["평형대", "구역", "단지명", "평형", "대지지분", "동", "층/호", "가격", "부동산", "상태"]
missing = [c for c in need_cols if c not in df_list.columns]
if missing:
    st.error(f"'매매물건 목록' 탭에서 다음 컬럼이 필요합니다: {missing}")
    st.stop()

# 좌표 컬럼 확보
for c in ["위도", "경도"]:
    if c not in df_list.columns:
        df_list[c] = None

df_list["동_key"] = df_list["동"].apply(dong_key)

df_loc = df_loc.copy()
if "동" in df_loc.columns:
    df_loc["동_key"] = df_loc["동"].apply(dong_key)

# 활성 필터
df_view = df_list
if only_active:
    df_view = df_view[df_view["상태"].astype(str).str.strip() == "활성"].copy()

# 좌표 숫자화
df_view["위도"] = df_view["위도"].apply(to_float)
df_view["경도"] = df_view["경도"].apply(to_float)

# 위치정보 탭으로 좌표 보강
if all(c in df_loc.columns for c in ["단지명", "동_key", "위도", "경도"]):
    df_loc["위도"] = df_loc["위도"].apply(to_float)
    df_loc["경도"] = df_loc["경도"].apply(to_float)

    df_view = df_view.merge(
        df_loc[["단지명", "동_key", "위도", "경도"]].rename(columns={"위도": "위도_loc", "경도": "경도_loc"}),
        on=["단지명", "동_key"],
        how="left",
    )
    df_view["위도"] = df_view["위도"].fillna(df_view["위도_loc"])
    df_view["경도"] = df_view["경도"].fillna(df_view["경도_loc"])
    df_view.drop(columns=["위도_loc", "경도_loc"], inplace=True)

df_view = df_view.dropna(subset=["위도", "경도"]).copy()
if df_view.empty:
    st.warning("현재 표시할 활성 매물이 없거나 좌표가 없습니다.")
    st.stop()

# 평형대/가격 정규화 컬럼
df_view = df_view.copy()
df_view["가격_num"] = pd.to_numeric(df_view["가격"], errors="coerce")
df_view["평형대_num"] = df_view["평형대"].map(parse_pyeong_num)
df_view["평형대_bucket"] = df_view["평형대_num"].apply(pyeong_bucket_10)
df_view["가격(억)표시"] = df_view["가격_num"].map(lambda v: fmt_decimal(v, 2))

# 그룹(동 단위 포인트)
gdf = build_grouped(df_view)

# 구역별 색상
palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
areas = sorted([a for a in gdf["구역"].dropna().astype(str).unique()])
area_color = {a: palette[i % len(palette)] for i, a in enumerate(areas)}
default_color = "#333333"

DEFAULT_ZOOM = 16

# 상태 변수
if "map_center" not in st.session_state:
    st.session_state["map_center"] = [float(gdf["위도"].mean()), float(gdf["경도"].mean())]
if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = DEFAULT_ZOOM
if "selected_meta" not in st.session_state:
    st.session_state["selected_meta"] = None
if "last_click_sig" not in st.session_state:
    st.session_state["last_click_sig"] = ""
if "last_table_sel_sig" not in st.session_state:
    st.session_state["last_table_sel_sig"] = ""

if "quick_filter_mode" not in st.session_state:
    st.session_state["quick_filter_mode"] = "size"
if "quick_filter_bucket" not in st.session_state:
    st.session_state["quick_filter_bucket"] = 30


# ====== 지도 생성 ======
m = folium.Map(
    location=st.session_state["map_center"],
    zoom_start=int(st.session_state["map_zoom"]),
    tiles="CartoDB positron",
)

marker_rows = []
for _, r in gdf.iterrows():
    marker_rows.append(
        (
            r["위도"],
            r["경도"],
            {"단지명": r["단지명"], "동_key": r["동_key"], "구역": r["구역"], "위도": r["위도"], "경도": r["경도"]},
        )
    )

for _, r in gdf.iterrows():
    area_raw = str(r["구역"]) if pd.notna(r["구역"]) else ""
    bg = area_color.get(area_raw, default_color)
    dong_label = str(r["동_key"])
    area_display = f"{norm_area(area_raw)}구역" if norm_area(area_raw) else ""
    tooltip = f"{area_display} | {r['단지명']} {dong_label}동 | 활성 {int(r['활성건수'])}건"

    folium.CircleMarker(
        location=[r["위도"], r["경도"]],
        radius=18,
        weight=0,
        opacity=0,
        fill=True,
        fill_opacity=0,
        tooltip=tooltip,
    ).add_to(m)

    folium.Marker(
        location=[r["위도"], r["경도"]],
        icon=folium.DivIcon(html=make_circle_label_html(dong_label, bg)),
        tooltip=tooltip,
    ).add_to(m)

# 지도 영역 확장 (상하 배치로 변경)
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

st.subheader("선택한 동의 활성 매물")

meta = st.session_state.get("selected_meta", None)
if not meta:
    st.info("지도에서 마커를 클릭하면 아래에 상세가 표시됩니다.")
    st.stop()

complex_name = meta["단지명"]
dong = meta["동_key"]
area_value = str(meta["구역"]) if pd.notna(meta["구역"]) else ""
area_norm = norm_area(area_value)

df_pick = df_view[(df_view["단지명"] == complex_name) & (df_view["동_key"] == dong)].copy()

# ===== 요청 반영 (정확 적용) =====
# - '평형대','구역' 제거
# - '부동산'을 '요약내용' 앞에 배치
show_cols = ["단지명", "평형", "대지지분", "동", "층/호", "가격", "부동산", "요약내용", "상태"]
show_cols = [c for c in show_cols if c in df_pick.columns]
view_pick = df_pick[show_cols].reset_index(drop=True)

# 폭: 요약내용을 최대(large), 다른 컬럼은 최대한 small로 눌러 체감상 3배 확보
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
    st_df(
        view_pick,
        use_container_width=True,
        height=dataframe_height(view_pick, max_height=650),
        column_config=col_cfg,
    )
except TypeError:
    st_df(
        view_pick,
        use_container_width=True,
        height=dataframe_height(view_pick, max_height=650),
    )
# ================================

st.divider()

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

    trades = recent_trades(df_trade, area_value, complex_name, sel_pyeong)
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
            height=dataframe_height(trades2, max_height=240),
        )

st.divider()

st.divider()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("선택 구역 평형별 요약 (활성 매물)")
    if not area_value:
        st.info("선택한 마커의 구역 정보가 없습니다.")
    else:
        summary = summarize_area_by_size(df_view, area_value)
        if summary.empty:
            st.info("해당 구역에서 요약할 데이터가 없습니다.")
        else:
            summary_view = summary[["평형", "매물건수", "가격대(최저~최고)", "최저가격", "최고가격"]].reset_index(drop=True)
            st_df(
                summary_view,
                use_container_width=True,
                height=dataframe_height(summary_view, max_height=380),
            )

with col_right:
    st.subheader("빠른 필터 (활성 매물)")

    # -----------------------------
    # (A) 평형대 필터: 상단
    # -----------------------------
    c0, c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1, 1, 1, 1, 1, 1, 1.2])

    buckets = [20, 30, 40, 50, 60, 70, 80]
    cols = [c0, c1, c2, c3, c4, c5, c6]
    for col, b in zip(cols, buckets):
        if col.button(f"{b}평대", use_container_width=True, key=f"qsize_{b}"):
            st.session_state["quick_filter_mode"] = "size"
            st.session_state["quick_filter_bucket"] = b
            st.rerun()

    if c7.button("가격순", use_container_width=True, key="qprice"):
        st.session_state["quick_filter_mode"] = "price"
        st.rerun()

    mode = st.session_state["quick_filter_mode"]

    # -----------------------------
    # (B) 구역 필터: 하단(2단 버튼)
    # - 본 시트의 '구역' 컬럼을 읽어 동적으로 구성
    # - 기본값: 지도에서 클릭한 마커의 구역
    # -----------------------------
    df_area = df_view[["구역"]].copy()
    df_area["_area_norm"] = df_area["구역"].astype(str).map(norm_area)
    df_area = df_area[df_area["_area_norm"].astype(str).str.strip() != ""].copy()

    area_norms = sorted(df_area["_area_norm"].dropna().astype(str).unique().tolist(),
                        key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))

    area_labels = {}
    for an in area_norms:
        if str(an).isdigit():
            area_labels[an] = f"{int(an)}구역"
        else:
            area_labels[an] = str(an)

    # 세션 초기화/보정
    if "quick_filter_area_norm" not in st.session_state:
        st.session_state["quick_filter_area_norm"] = area_norm if area_norm else "__ALL__"

    # 현재 마커 구역이 바뀌었으면(새 클릭) 선택값이 비어있을 때만 따라가도록
    if st.session_state.get("quick_filter_area_norm") in [None, "", "__NO_SUCH_AREA__"]:
        st.session_state["quick_filter_area_norm"] = area_norm if area_norm else "__ALL__"

    # 선택값이 실제 목록에 없으면, 마커 구역 -> 첫 구역 -> 전체 순으로 폴백
    sel_area_norm = st.session_state.get("quick_filter_area_norm", "__ALL__")
    if sel_area_norm != "__ALL__" and sel_area_norm not in area_labels:
        if area_norm and area_norm in area_labels:
            sel_area_norm = area_norm
        elif len(area_norms) > 0:
            sel_area_norm = area_norms[0]
        else:
            sel_area_norm = "__ALL__"
        st.session_state["quick_filter_area_norm"] = sel_area_norm

    st.markdown("**구역 선택**")
    aleft, aright = st.columns(2)

    # '전체' 버튼 포함
    all_label = "[선택] 전체" if sel_area_norm == "__ALL__" else "전체"
    if aleft.button(all_label, use_container_width=True, key="qarea_all"):
        st.session_state["quick_filter_area_norm"] = "__ALL__"
        st.rerun()

    # 구역 버튼(2단)
    for i, an in enumerate(area_norms):
        label = area_labels.get(an, str(an))
        btn_label = f"[선택] {label}" if an == sel_area_norm else label
        target_col = aleft if (i % 2 == 0) else aright
        if target_col.button(btn_label, use_container_width=True, key=f"qarea_{an}"):
            st.session_state["quick_filter_area_norm"] = an
            st.rerun()

    # -----------------------------
    # (C) 결과 표시
    # - 선택 구역 내(또는 전체)에서 가격 오름차순
    # -----------------------------
    if sel_area_norm == "__ALL__":
        area_display = "전체"
    else:
        area_display = area_labels.get(sel_area_norm, f"{sel_area_norm}구역")

    if mode == "size":
        st.caption(f"현재: {area_display} / {st.session_state['quick_filter_bucket']}평대 (가격 낮은 순)")
    else:
        st.caption(f"현재: {area_display} / 전체 평형 (가격 낮은 순)")

    dfq = df_view.copy()
    dfq = dfq[dfq["가격_num"].notna()].copy()
    dfq["_area_norm"] = dfq["구역"].astype(str).map(norm_area)

    if sel_area_norm != "__ALL__":
        dfq = dfq[dfq["_area_norm"] == sel_area_norm].copy()

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

        event = st_df(
            df_table,
            height=dataframe_height(df_table, max_height=650),
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
