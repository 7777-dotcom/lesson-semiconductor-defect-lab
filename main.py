import json
from pathlib import Path

import pandas as pd
import streamlit as st

TITLE = "반도체 불량 탐지 실습실"
ICON = "🔬"

st.set_page_config(page_title=TITLE, page_icon=ICON, layout="wide")

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "uci-secom.csv"
RECORD_DIR = BASE_DIR / "내_기록"
RECORD_PATH = RECORD_DIR / "데이터_출처.json"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, encoding="utf-8")


@st.cache_data
def make_sensor_summary(df: pd.DataFrame) -> pd.DataFrame:
    sensors = df.drop(columns=["Time", "Pass/Fail"])
    summary = pd.DataFrame(
        {
            "센서 열": sensors.columns,
            "결측률(%)": (sensors.isna().mean() * 100).values,
            "평균": sensors.mean().values,
            "표준편차": sensors.std().values,
            "값 종류 수": sensors.nunique(dropna=True).values,
        }
    )
    summary["구분"] = summary["값 종류 수"].apply(lambda n: "상수 열" if n <= 1 else "")
    return summary


def load_record() -> str:
    try:
        return json.loads(RECORD_PATH.read_text(encoding="utf-8")).get("출처", "")
    except (FileNotFoundError, json.JSONDecodeError):
        return ""


def save_record():
    try:
        RECORD_DIR.mkdir(exist_ok=True)
        RECORD_PATH.write_text(
            json.dumps({"출처": st.session_state["source_text"]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        st.session_state["save_msg"] = "저장했어요."
    except OSError:
        st.session_state["save_msg"] = "저장하지 못했어요. 이 환경에서는 파일 저장이 막혀 있을 수 있어요."


st.title(f"{ICON} {TITLE}")
st.caption("반도체 제조 공정 센서 데이터(SECOM)로 제품 불량을 얼마나 가려낼 수 있는지 확인하는 실습 앱")

try:
    df = load_data()
except FileNotFoundError:
    st.error("uci-secom.csv 파일을 찾을 수 없어요. main.py와 같은 폴더에 있는지 확인해 주세요.")
    st.stop()

sensor_cols = [c for c in df.columns if c not in ("Time", "Pass/Fail")]
n_rows = len(df)
n_fail = int((df["Pass/Fail"] == 1).sum())
n_pass = int((df["Pass/Fail"] == -1).sum())

if n_fail + n_pass != n_rows:
    st.warning("Pass/Fail 열에 -1과 1이 아닌 값이 있어요. 아래 숫자를 확인해 주세요.")

# 1) 큰 숫자 카드 4개
c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 행 수", f"{n_rows:,}")
c2.metric("센서 열 수", f"{len(sensor_cols):,}")
c3.metric("불량(1) 수", f"{n_fail:,}")
c4.metric("불량 비율", f"{n_fail / n_rows * 100:.2f}%")

# 2) 열 설명표
st.subheader("열 설명")
time_desc = "측정 일시 (행 순서가 시간순이 아니고 날짜 형식이 일정하지 않을 수 있어, 이 앱에서는 분석에 쓰지 않음)"

col_desc = pd.DataFrame(
    {
        "열 이름": ["Time", f"{sensor_cols[0]} ~ {sensor_cols[-1]}", "Pass/Fail"],
        "설명": [
            time_desc,
            f"공정 중 수집된 수치형 센서값 {len(sensor_cols)}개 (측정 대상 비공개)",
            f"최종 검사 결과: -1 = 정상({n_pass:,}건), 1 = 불량({n_fail:,}건)",
        ],
    }
)
st.dataframe(col_desc, hide_index=True, use_container_width=True)

# 3) 센서 요약표
st.subheader("센서 요약")
summary = make_sensor_summary(df)
n_const = int((summary["구분"] == "상수 열").sum())
st.write(f"값이 하나뿐이거나 비어 있는 상수 열은 센서 {len(sensor_cols)}개 중 **{n_const}개**예요.")
st.dataframe(
    summary,
    hide_index=True,
    use_container_width=True,
    column_config={
        "결측률(%)": st.column_config.NumberColumn(format="%.1f"),
        "평균": st.column_config.NumberColumn(format="%.4g"),
        "표준편차": st.column_config.NumberColumn(format="%.4g"),
    },
)

# 4) 처음 다섯 줄
st.subheader("처음 다섯 줄")
st.dataframe(df.head(5), use_container_width=True)

# 5) 데이터 출처 입력란 (내_기록 폴더에 json 저장)
st.subheader("데이터 출처")
if "source_text" not in st.session_state:
    st.session_state["source_text"] = load_record()

st.text_input(
    "데이터를 어디에서 가져왔는지 적어 주세요.",
    key="source_text",
    on_change=save_record,
    placeholder="예: Kaggle, UCI SECOM (uci-secom.csv)",
)
if "save_msg" in st.session_state:
    st.caption(st.session_state.pop("save_msg"))
