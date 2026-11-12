from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

TITLE = "반도체 불량 탐지 실습실"
ICON = "🌳"

DATA_PATH = Path(__file__).parent.parent / "uci-secom.csv"
MISSING_LIMIT = 0.4  # 2_분류_모델과 같은 전처리 (로지스틱·의사결정트리용)

BASELINE = "기준 모델 (전부 정상으로 예측)"
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


def make_models() -> dict:
    return {
        BASELINE: DummyClassifier(strategy="constant", constant=0),
        LOGISTIC: make_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000)),
        TREE: make_pipeline(
            DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
        ),
        # 부스팅은 빈 값을 채우지 않고 그대로 넣는다 (스스로 처리함)
        BOOSTING: HistGradientBoostingClassifier(
            class_weight="balanced",
            max_depth=3,
            learning_rate=0.05,
            max_iter=150,
            random_state=42,
        ),
    }


def get_xy(df: pd.DataFrame):
    X = df.drop(columns=["Time", "Pass/Fail"])
    y = (df["Pass/Fail"] == 1).astype(int)  # 불량(1)을 양성으로
    return X, y


@st.cache_data
def evaluate_all(df: pd.DataFrame) -> dict:
    """층화 5겹 교차검증(2_분류_모델과 같은 분할). 모델마다 5겹 평균 지표와 합산 혼동행렬."""
    X, y = get_xy(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for name, model in make_models().items():
        recalls, precisions, f1s = [], [], []
        preds = pd.Series(0, index=y.index)
        for train_idx, test_idx in cv.split(X, y):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_test = y.iloc[test_idx]
            pred = model.predict(X.iloc[test_idx])
            preds.iloc[test_idx] = pred
            recalls.append(recall_score(y_test, pred, zero_division=0))
            precisions.append(precision_score(y_test, pred, zero_division=0))
            f1s.append(f1_score(y_test, pred, zero_division=0))
        results[name] = {
            "recall": sum(recalls) / len(recalls),
            "precision": sum(precisions) / len(precisions),
            "f1": sum(f1s) / len(f1s),
            "tp": int(((y == 1) & (preds == 1)).sum()),
            "fp": int(((y == 0) & (preds == 1)).sum()),
            "fn": int(((y == 1) & (preds == 0)).sum()),
            "tn": int(((y == 0) & (preds == 0)).sum()),
        }
    return results


st.title(f"{ICON} 부스팅 모델")
st.caption("수업에서 배운 모델에 더해, 빈 값을 그대로 다룰 수 있는 그래디언트 부스팅을 비교해요.")

try:
    df = load_data()
except FileNotFoundError:
    st.error("uci-secom.csv 파일을 찾을 수 없어요. 저장소 맨 위 폴더에 있는지 확인해 주세요.")
    st.stop()

_, y = get_xy(df)
n_fail = int((y == 1).sum())

# ───────────── 설명 ─────────────
st.subheader("그래디언트 부스팅이란?")
st.write(
    "작은 의사결정트리를 여러 개 차례로 이어 붙이는 방법이에요. "
    "앞 트리가 틀린 제품에 다음 트리가 더 집중해서, 실수를 조금씩 고쳐 가요."
)
st.write(
    "이 데이터에는 센서 값이 빈 곳이 많은데, 이 모델은 빈 값을 채우지 않고 그대로 넣어도 스스로 처리해요. "
    "그래서 이 페이지의 부스팅은 중앙값 채우기와 표준화 없이 원래 값을 그대로 썼어요."
)

# ───────────── 학습 ─────────────
with st.spinner("처음 한 번만 학습해요. 부스팅이 있어서 시간이 걸릴 수 있어요..."):
    results = evaluate_all(df)

# ───────────── 모델 비교 표 ─────────────
st.subheader("모델 비교 (층화 5겹 교차검증 평균)")
table = pd.DataFrame(
    [
        {
            "모델": name,
            "재현율": r["recall"],
            "정밀도": r["precision"],
            "F1": r["f1"],
        }
        for name, r in results.items()
    ]
)
st.dataframe(
    table,
    hide_index=True,
    use_container_width=True,
    column_config={
        "재현율": st.column_config.NumberColumn(format="%.3f"),
        "정밀도": st.column_config.NumberColumn(format="%.3f"),
        "F1": st.column_config.NumberColumn(format="%.3f"),
    },
)

fig = go.Figure()
for metric, label, color in [
    ("recall", "재현율", "#F58518"),
    ("precision", "정밀도", "#54A24B"),
    ("f1", "F1", "#4C78A8"),
]:
    fig.add_trace(
        go.Bar(
            name=label,
            x=list(results.keys()),
            y=[r[metric] for r in results.values()],
            marker_color=color,
        )
    )
fig.update_layout(barmode="group", title="모델별 재현율·정밀도·F1", yaxis_title="점수")
st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"모든 모델이 같은 5겹 분할을 썼어요. 실제 불량이 {n_fail:,}개뿐이라서 "
    "모델 사이의 작은 차이는 우연일 수 있어요."
)

# ───────────── 부스팅 혼동행렬 ─────────────
st.subheader("부스팅 혼동행렬 (행: 실제, 열: 예측)")
b = results[BOOSTING]
matrix = pd.DataFrame(
    [[f"TN {b['tn']:,}", f"FP {b['fp']:,}"], [f"FN {b['fn']:,}", f"TP {b['tp']:,}"]],
    index=["실제 정상(-1)", "실제 불량(1)"],
    columns=["예측 정상", "예측 불량"],
)
st.table(matrix)

warned = b["tp"] + b["fp"]
if warned > 0:
    warn_text = f"불량이라고 경고한 {warned:,}개 중 실제 불량은 {b['tp']:,}개였어요."
else:
    warn_text = "불량이라고 경고한 제품이 하나도 없었어요."
st.write(f"불량 {n_fail:,}개 중 {b['tp']:,}개를 찾았고, {b['fn']:,}개를 놓쳤어요. {warn_text}")

lr, tree = results[LOGISTIC], results[TREE]
st.write(
    f"찾은 불량(TP) / 오경보(FP): 로지스틱 회귀 {lr['tp']:,} / {lr['fp']:,}, "
    f"의사결정트리 {tree['tp']:,} / {tree['fp']:,}, 부스팅 {b['tp']:,} / {b['fp']:,}"
)
st.caption(
    "혼동행렬은 5겹 시험 결과를 합친 값이고(3_채점표와 같음), "
    "위 표의 5겹 평균과는 소수점 셋째 자리가 조금 다를 수 있어요."
)
