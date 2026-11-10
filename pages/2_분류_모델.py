from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz

TITLE = "반도체 불량 탐지 실습실"
ICON = "🤖"

DATA_PATH = Path(__file__).parent.parent / "uci-secom.csv"
MISSING_LIMIT = 0.4  # 결측률이 이 값 이상인 열은 제거


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
        "기준 모델 (전부 정상으로 예측)": DummyClassifier(strategy="constant", constant=0),
        "로지스틱 회귀": make_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=2000)
        ),
        "의사결정트리 (깊이 3)": make_pipeline(
            DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
        ),
    }


def get_xy(df: pd.DataFrame):
    X = df.drop(columns=["Time", "Pass/Fail"])
    y = (df["Pass/Fail"] == 1).astype(int)  # 불량(1)을 양성으로
    return X, y


@st.cache_data
def evaluate_models(df: pd.DataFrame) -> pd.DataFrame:
    """층화 5겹 교차검증으로 재현율·정밀도·F1 평균을 계산한다."""
    X, y = get_xy(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    for name, model in make_models().items():
        recalls, precisions, f1s = [], [], []
        for train_idx, test_idx in cv.split(X, y):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            y_test = y.iloc[test_idx]
            recalls.append(recall_score(y_test, pred, zero_division=0))
            precisions.append(precision_score(y_test, pred, zero_division=0))
            f1s.append(f1_score(y_test, pred, zero_division=0))
        rows.append(
            {
                "모델": name,
                "재현율": sum(recalls) / len(recalls),
                "정밀도": sum(precisions) / len(precisions),
                "F1": sum(f1s) / len(f1s),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data
def make_tree_dot(df: pd.DataFrame):
    """전체 데이터로 의사결정트리를 한 번 학습해 DOT 문자열을 만든다."""
    X, y = get_xy(df)
    pipe = make_pipeline(
        DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
    )
    pipe.fit(X, y)
    kept_cols = pipe.named_steps["drop"].keep_cols_
    dot = export_graphviz(
        pipe.named_steps["model"],
        out_file=None,
        feature_names=[f"센서 {c}" for c in kept_cols],
        class_names=["정상", "불량"],
        filled=True,
        rounded=True,
    )
    return dot, len(kept_cols), X.shape[1]


st.title(f"{ICON} 분류 모델")
st.caption("센서 수치로 불량(1)을 얼마나 가려내는지, 모델 세 개를 같은 조건에서 비교해요.")

try:
    df = load_data()
except FileNotFoundError:
    st.error("uci-secom.csv 파일을 찾을 수 없어요. 저장소 맨 위 폴더에 있는지 확인해 주세요.")
    st.stop()

n_total = len(df)
n_fail = int((df["Pass/Fail"] == 1).sum())
baseline_acc = (n_total - n_fail) / n_total * 100

# ───────────── 정확도를 쓰지 않는 이유 (두 줄) ─────────────
st.subheader("정확도는 쓰지 않는 이유")
st.write(
    f"전부 정상으로 예측하는 기준 모델도 정확도가 {baseline_acc:.1f}%나 나와서, "
    "정확도만으로는 불량을 잡아내는지 알 수 없어요."
)
st.write(
    "그래서 불량을 얼마나 찾았는지(재현율)와 불량이라고 한 것 중 실제 불량의 비율(정밀도)로 평가해요."
)

# ───────────── 교차검증 결과 표 ─────────────
st.subheader("모델 비교 (층화 5겹 교차검증 평균)")
with st.spinner("처음 한 번만 학습해요. 잠시 기다려 주세요..."):
    result = evaluate_models(df)

st.dataframe(
    result,
    hide_index=True,
    use_container_width=True,
    column_config={
        "재현율": st.column_config.NumberColumn(format="%.3f"),
        "정밀도": st.column_config.NumberColumn(format="%.3f"),
        "F1": st.column_config.NumberColumn(format="%.3f"),
    },
)
st.caption(
    "불량(1)을 양성으로 봤어요. 재현율 = 실제 불량 중 찾은 비율, "
    "정밀도 = 불량이라고 예측한 것 중 실제 불량 비율, F1 = 둘의 조화평균이에요. "
    "예측한 불량이 0명이면 정밀도와 F1은 0으로 표시해요."
)
st.caption(
    "상수 열 제거, 결측 40% 이상 열 제거, 중앙값 채우기, 표준화는 "
    "각 겹의 학습 데이터에서만 계산했어요."
)

# ───────────── 의사결정트리 가지 그림 ─────────────
st.subheader("의사결정트리 가지 그림")
dot, n_kept, n_all = make_tree_dot(df)
st.graphviz_chart(dot, use_container_width=True)
st.caption(
    f"전체 데이터로 한 번 학습한 그림이에요 (전처리 후 센서 {n_kept}개 사용, 원래 {n_all}개). "
    "기준값은 표준화된 값이라 센서의 원래 단위가 아니에요. "
    "samples는 행 수이고, value는 클래스 가중치가 반영돼 있어요."
)
