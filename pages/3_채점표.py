from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

TITLE = "반도체 불량 탐지 실습실"
ICON = "📝"

DATA_PATH = Path(__file__).parent.parent / "uci-secom.csv"
MISSING_LIMIT = 0.4  # 결측률이 이 값 이상인 열은 제거 (2_분류_모델과 같음)


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


def make_model(name: str) -> Pipeline:
    if name == "로지스틱 회귀":
        return make_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000))
    return make_pipeline(
        DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
    )


def get_xy(df: pd.DataFrame):
    X = df.drop(columns=["Time", "Pass/Fail"])
    y = (df["Pass/Fail"] == 1).astype(int)  # 불량(1)을 양성으로
    return X, y


@st.cache_data
def cross_predictions(df: pd.DataFrame, model_name: str) -> pd.Series:
    """층화 5겹 교차검증에서 각 행이 시험 데이터였을 때의 예측을 모아 돌려준다."""
    X, y = get_xy(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = pd.Series(0, index=y.index)
    for train_idx, test_idx in cv.split(X, y):
        model = make_model(model_name)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds.iloc[test_idx] = model.predict(X.iloc[test_idx])
    return preds


def count_cells(y: pd.Series, pred: pd.Series):
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return tp, fp, fn, tn


def divide(a: int, b: int) -> float:
    return a / b if b > 0 else 0.0  # 분모가 0이면(예측한 불량이 0명 등) 0으로 표시


def scores(tp: int, fp: int, fn: int):
    recall = divide(tp, tp + fn)
    precision = divide(tp, tp + fp)
    f1 = divide(2 * precision * recall, precision + recall)
    return recall, precision, f1


st.title(f"{ICON} 채점표")
st.caption("모델이 불량을 얼마나 찾았고 얼마나 놓쳤는지, 5겹 교차검증 결과를 합쳐서 채점해요.")

try:
    df = load_data()
except FileNotFoundError:
    st.error("uci-secom.csv 파일을 찾을 수 없어요. 저장소 맨 위 폴더에 있는지 확인해 주세요.")
    st.stop()

_, y = get_xy(df)
n_total = len(y)
n_fail = int((y == 1).sum())

model_name = st.selectbox("모델 선택", ["로지스틱 회귀", "의사결정트리 (깊이 3)"])

with st.spinner("처음 한 번만 학습해요. 잠시 기다려 주세요..."):
    pred = cross_predictions(df, model_name)

tp, fp, fn, tn = count_cells(y, pred)
recall, precision, f1 = scores(tp, fp, fn)

# ───────────── 혼동행렬 ─────────────
st.subheader("혼동행렬 (행: 실제, 열: 예측)")
matrix = pd.DataFrame(
    [[f"TN {tn:,}", f"FP {fp:,}"], [f"FN {fn:,}", f"TP {tp:,}"]],
    index=["실제 정상(-1)", "실제 불량(1)"],
    columns=["예측 정상", "예측 불량"],
)
st.table(matrix)
st.caption(
    "TP = 불량을 불량으로 찾음, FN = 불량을 놓침, "
    "FP = 정상을 불량이라고 잘못 경고함, TN = 정상을 정상으로 맞힘. "
    f"5개 겹의 시험 결과를 모두 합쳐서 {n_total:,}개 제품이 한 번씩 채점돼요."
)

# ───────────── 큰 숫자 ─────────────
st.subheader("성적")
c1, c2, c3 = st.columns(3)
c1.metric("재현율", f"{recall:.3f}", help="실제 불량 중 찾아낸 비율 = TP ÷ (TP + FN)")
c2.metric("정밀도", f"{precision:.3f}", help="불량이라고 경고한 것 중 실제 불량 비율 = TP ÷ (TP + FP)")
c3.metric("F1", f"{f1:.3f}", help="재현율과 정밀도의 조화평균")

# 기준 모델: 전부 정상으로 예측 → TP = FP = 0
b_recall, b_precision, b_f1 = scores(0, 0, n_fail)
st.write(
    f"기준 모델(전부 정상으로 예측): 재현율 {b_recall:.3f}, 정밀도 {b_precision:.3f}, "
    f"F1 {b_f1:.3f} → 불량을 하나도 찾지 못해요. "
    f"선택한 모델은 불량을 {tp:,}개 찾았어요."
)

# ───────────── 문장 설명 ─────────────
st.subheader("한 문장으로 보면")
warned = tp + fp
if warned > 0:
    warn_text = f"불량이라고 경고한 {warned:,}개 중 실제 불량은 {tp:,}개였어요."
else:
    warn_text = "불량이라고 경고한 제품이 하나도 없었어요."
st.write(f"불량 {n_fail:,}개 중 {tp:,}개를 찾았고, {fn:,}개를 놓쳤어요. {warn_text}")

st.caption(
    "여기 숫자는 합친 혼동행렬에서 한 번에 계산한 값이라, "
    "2_분류_모델의 5겹 평균과 소수점 셋째 자리가 조금 다를 수 있어요."
)
