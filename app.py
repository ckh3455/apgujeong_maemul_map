import re
import pandas as pd
import streamlit as st

import folium
from streamlit_folium import st_folium
from streamlit_gsheets import GSheetsConnection
from streamlit_autorefresh import st_autorefresh


SHEET_URL = "https://docs.google.com/spreadsheets/d/1QP56lm5kPBdsUhrgcgY2U-JdmukXIkKCSxefd1QExKE/edit?gid=733341548#gid=733341548"

TAB_LISTING = "매매물건 목록"
TAB_LOC = "압구정 위치정보"

# 시트 컬럼명에 줄바꿈이 들어간 케이스 방지용
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\n", "").strip() for c in df.columns]
    return df

def dong_key(x) -> str:
    """'3', '3동', '03동' 등 어떤 형태든 숫자만 추출해서 동 키로 사용"""
    if pd.isna(x):
        return ""
    s = str(x)
    m = re.findall(r"\d+", s)
    return m[0].lstrip("0") if m else s.strip()

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

@st.cache_data(ttl=10)  # 10초마다 새로 읽어 활성 변경 반영
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    df_list = conn.read(worksheet=TAB_LISTING, ttl=0)
    df_loc = conn.read(worksheet=TAB_LOC, ttl=0)

    df_list = clean_columns(df_list)
    df_loc = clean_columns(df_loc)

    return df_list, df_loc

def build_grouped(df_active: pd.DataFrame) -> pd.DataFrame:
    # 그룹의 대표 좌표는 첫 행 좌표 사용(동별 고정 좌표라는 전제)
    g = (
        df_active
        .groupby(["단지명", "동_key"], dropna=False)
        .agg(
            위도=("위도", "first"),
            경도=("경도", "first"),
            활성건수=("동_key", "size"),
            최저가격=("가격", lambda s: pd.to_numeric(s, errors="coerce").min()),
            최고가격=("가격", lambda s: pd.to_numeric(s, errors="coerce").max()),
        )
        .reset_index()
    )
    return g

st.set_page_config(layout="wide")

st.title("압구정 매물 지도 MVP (상태=활성)")

# 10초 자동 새로고침(폴링)
st_autorefresh(interval=10_000, key="auto_refresh")  # interval ms 단위 :contentReference[oaicite:6]{index=6}

with st.sidebar:
    st.subheader("필터")
    only_active = st.checkbox("상태=활성만 표시", value=True)
    st.caption("10초마다 자동 갱신됩니다.")

df_list, df_loc = load_data()

# 필수 컬럼 체크(없으면 바로 안내)
need_cols = ["평형대","구역","단지명","평형","대지지분","동","층수","가격","부동산","상태"]
missing = [c for c in need_cols if c not in df_list.columns]
if missing:
    st.error(f"'매매물건 목록' 탭에서 다음 컬럼이 필요합니다: {missing}")
    st.stop()

# 동 키 생성
df_list = df_list.copy()
df_list["동_key"] = df_list["동"].apply(dong_key)

df_loc = df_loc.copy()
if "동" in df_loc.columns:
    df_loc["동_key"] = df_loc["동"].apply(dong_key)

# 좌표 컬럼이 없으면 만들어 둠
for c in ["위도", "경도"]:
    if c not in df_list.columns:
        df_list[c] = None

# 활성만
df_view = df_list
if only_active:
    df_view = df_view[df_view["상태"].astype(str).str.strip() == "활성"].copy()

# 좌표 숫자화
df_view["위도"] = df_view["위도"].apply(to_float)
df_view["경도"] = df_view["경도"].apply(to_float)

# 좌표가 비어있는 행은 위치정보 탭으로 보강(단지명+동_key 기준)
if all(c in df_loc.columns for c in ["단지명","동_key","위도","경도"]):
    df_loc["위도"] = df_loc["위도"].apply(to_float)
    df_loc["경도"] = df_loc["경도"].apply(to_float)

    df_view = df_view.merge(
        df_loc[["단지명","동_key","위도","경도"]].rename(columns={"위도":"위도_loc","경도":"경도_loc"}),
        on=["단지명","동_key"],
        how="left"
    )
    df_view["위도"] = df_view["위도"].fillna(df_view["위도_loc"])
    df_view["경도"] = df_view["경도"].fillna(df_view["경도_loc"])
    df_view.drop(columns=["위도_loc","경도_loc"], inplace=True)

# 좌표 없는 건 제외
df_view = df_view.dropna(subset=["위도","경도"]).copy()
if df_view.empty:
    st.warning("현재 표시할 활성 매물이 없거나(상태=활성), 좌표가 모두 비어있습니다.")
    st.stop()

# 그룹(동 단위 포인트)
gdf = build_grouped(df_view)

# 지도 중심
center_lat = float(gdf["위도"].mean())
center_lng = float(gdf["경도"].mean())

m = folium.Map(location=[center_lat, center_lng], zoom_start=15, tiles="CartoDB positron")

# 클릭 좌표로 그룹을 찾기 위한 인덱스(부동소수점 오차 방지: 반올림 문자열)
def key_latlng(lat, lng, nd=6):
    return f"{round(float(lat), nd)}|{round(float(lng), nd)}"

group_index = {}
for _, r in gdf.iterrows():
    k = key_latlng(r["위도"], r["경도"])
    group_index[k] = (r["단지명"], r["동_key"])

# 마커 추가
for _, r in gdf.iterrows():
    tooltip = f"{r['단지명']} {r['동_key']}동 | 활성 {int(r['활성건수'])}건"
    popup_html = f"""
    <div style="font-size:13px;">
      <b>{r['단지명']} {r['동_key']}동</b><br/>
      활성: {int(r['활성건수'])}건<br/>
      가격범위: {r['최저가격']} ~ {r['최고가격']}
    </div>
    """
    folium.Marker(
        location=[r["위도"], r["경도"]],
        tooltip=tooltip,
        popup=folium.Popup(popup_html, max_width=320),
    ).add_to(m)

col_map, col_table = st.columns([1.1, 1])

with col_map:
    st.subheader("지도")
    out = st_folium(m, height=650, width=None, returned_objects=["last_object_clicked"])

with col_table:
    st.subheader("선택한 동의 활성 매물")
    clicked = (out or {}).get("last_object_clicked", None)

    if clicked:
        lat = clicked.get("lat")
        lng = clicked.get("lng")
        k = key_latlng(lat, lng)
        picked = group_index.get(k)

        if picked:
            complex_name, dong = picked
            df_pick = df_view[(df_view["단지명"] == complex_name) & (df_view["동_key"] == dong)].copy()

            show_cols = ["평형대","구역","단지명","평형","대지지분","동","층수","가격","부동산","상태","위도","경도"]
            show_cols = [c for c in show_cols if c in df_pick.columns]
            st.dataframe(df_pick[show_cols], use_container_width=True)
        else:
            st.info("마커를 클릭하면 해당 동의 활성 매물이 표로 표시됩니다.")
    else:
        st.info("마커를 클릭하면 해당 동의 활성 매물이 표로 표시됩니다.")
