from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

TITLE = "반도체 불량 탐지 실습실"
ICON = "🔁"

DATA_PATH = Path(__file__).parent.parent / "uci-secom.csv"
MISSING_LIMIT = 0.4  # 2_분류_모델과 같은 전처리 (로지스틱·의사결정트리용)
FIRST_SEED = 42      # 첫 번째 반복은 다른 페이지와 같은 분할

LOGISTIC = "로지스틱 회귀"
TREE = "의사결정트리 (깊이 3)"
BOOSTING = "그래디언트 부스팅"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, encoding="utf-8")


class ColumnDropper(BaseEstimator, TransformerMixin):
    """학습 데이터에서만 상수 열과 결측률이 높은 열을 찾아 제거한다."""

    def __init__(self, missing_limit=MISSING_LIMIT):
        self.missing_limit = missing_limit

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        is_varied = X.nunique(dropna=True) > 1
        is_complete = X.isna().mean() < self.missing_limit
        self.keep_cols_ = list(X.columns[is_varied & is_complete])
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.keep_cols_]


def make_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("drop", ColumnDropper()),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def make_model(name: str):
    if name == LOGISTIC:
        return make_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000))
    if name == TREE:
        return make_pipeline(
            DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
        )
    return HistGradientBoostingClassifier(
        class_weight="balanced",
        max_depth=3,
        learning_rate=0.05,
        max_iter=150,
        random_state=42,
    )


def get_xy(df: pd.DataFrame):
    X = df.drop(columns=["Time", "Pass/Fail"])
    y = (df["Pass/Fail"] == 1).astype(int)  # 불량(1)을 양성으로
    return X, y


@st.cache_data
def repeat_score(df: pd.DataFrame, model_name: str, seed: int) -> dict:
    """난수 seed로 층화 5겹을 한 번 돌린 평균 재현율·정밀도·F1."""
    X, y = get_xy(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    recalls, precisions, f1s = [], [], []
    for train_idx, test_idx in cv.split(X, y):
        model = make_model(model_name)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        y_test = y.iloc[test_idx]
        recalls.append(recall_score(y_test, pred, zero_division=0))
        precisions.append(precision_score(y_test, pred, zero_division=0))
        f1s.append(f1_score(y_test, pred, zero_division=0))
    return {
        "recall": sum(recalls) / len(recalls),
        "precision": sum(precisions) / len(precisions),
        "f1": sum(f1s) / len(f1s),
    }


def reset_run():
    st.session_state["run_repeat"] = False


st.title(f"{ICON} 반복 평가")
st.caption("같은 데이터를 다르게 나누면 점수가 얼마나 달라지는지, 모델 순위가 그대로인지 확인해요.")

try:
    df = load_data()
except FileNotFoundError:
    st.error("uci-secom.csv 파일을 찾을 수 없어요. 저장소 맨 위 폴더에 있는지 확인해 주세요.")
    st.stop()

_, y = get_xy(df)
n_fail = int((y == 1).sum())

st.write("난수를 바꿔 층화 5겹 교차검증을 여러 번 반복하고, 반복마다 나온 점수를 모아서 비교해요.")
st.write(
    f"첫 번째 반복은 다른 페이지와 같은 분할(난수 {FIRST_SEED})이라 그 페이지의 5겹 평균과 같은 값이에요."
)

# ───────────── 입력 ─────────────
col_n, col_boost = st.columns(2)
with col_n:
    n_repeats = st.selectbox("반복 횟수", [3, 5, 10], on_change=reset_run)
with col_boost:
    include_boosting = st.checkbox(
        "그래디언트 부스팅도 포함 (오래 걸려요)", value=False, on_change=reset_run
    )

if st.button("반복 평가 시작"):
    st.session_state["run_repeat"] = True

if not st.session_state.get("run_repeat", False):
    st.info("위에서 설정을 고르고 '반복 평가 시작'을 눌러 주세요. 부스팅을 포함하면 몇 분 걸릴 수 있어요.")
    st.stop()

# ───────────── 반복 평가 ─────────────
model_names = [LOGISTIC, TREE] + ([BOOSTING] if include_boosting else [])
seeds = [FIRST_SEED + i for i in range(n_repeats)]

scores = {}
with st.spinner("반복해서 학습하는 중이에요. 처음 한 번만 오래 걸려요..."):
    for name in model_names:
        scores[name] = pd.DataFrame([repeat_score(df, name, seed) for seed in seeds])

# 반복마다 F1이 가장 높은 모델
f1_table = pd.DataFrame({name: scores[name]["f1"] for name in model_names})
winners = f1_table.idxmax(axis=1)
win_counts = winners.value_counts()

# ───────────── 요약 표 ─────────────
st.subheader(f"반복 {n_repeats}번 결과 요약")
summary = pd.DataFrame(
    [
        {
            "모델": name,
            "재현율 평균": scores[name]["recall"].mean(),
            "정밀도 평균": scores[name]["precision"].mean(),
            "F1 평균": scores[name]["f1"].mean(),
            "F1 최솟값": scores[name]["f1"].min(),
            "F1 최댓값": scores[name]["f1"].max(),
            "F1 1등 횟수": int(win_counts.get(name, 0)),
        }
        for name in model_names
    ]
)
st.dataframe(
    summary,
    hide_index=True,
    use_container_width=True,
    column_config={
        "재현율 평균": st.column_config.NumberColumn(format="%.3f"),
        "정밀도 평균": st.column_config.NumberColumn(format="%.3f"),
        "F1 평균": st.column_config.NumberColumn(format="%.3f"),
        "F1 최솟값": st.column_config.NumberColumn(format="%.3f"),
        "F1 최댓값": st.column_config.NumberColumn(format="%.3f"),
    },
)

if len(win_counts) > 1:
    order_text = "반복마다 F1이 가장 높은 모델이 바뀌었어요."
else:
    order_text = f"반복마다 F1이 가장 높은 모델이 {win_counts.index[0]}으로 같았어요."
wins_text = ", ".join(f"{name} {int(count)}번" for name, count in win_counts.items())
st.write(f"F1 1등은 반복 {n_repeats}번 중 {wins_text}이에요. {order_text}")

# ───────────── 그래프 ─────────────
fig = go.Figure()
for name in model_names:
    fig.add_trace(
        go.Box(
            y=scores[name]["f1"],
            name=name,
            boxpoints="all",
            jitter=0.4,
            pointpos=0,
        )
    )
fig.update_layout(
    title="반복마다 나온 F1 점수의 퍼짐",
    yaxis_title="F1 (5겹 평균)",
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"점 하나가 반복 한 번의 F1이에요. 불량이 {n_fail:,}개뿐이라 나누는 방식에 따라 점수가 달라져요. "
    "같은 데이터를 다르게 나눈 것이라, 새 제품에서의 실제 흔들림은 이보다 클 수 있어요."
)
