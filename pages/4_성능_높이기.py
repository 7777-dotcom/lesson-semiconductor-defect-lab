from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

TITLE = "반도체 불량 탐지 실습실"
ICON = "🚀"

DATA_PATH = Path(__file__).parent.parent / "uci-secom.csv"

MODEL_OPTIONS = ["로지스틱 회귀", "의사결정트리 (깊이 3)"]
MISSING_OPTIONS = {"20%": 0.2, "40%": 0.4, "50%": 0.5, "제외 안 함": 1.01}
IMBALANCE_OPTIONS = [
    "처리 없음",
    "클래스 가중치",
    "언더샘플링 (정상:불량=3:1)",
    "랜덤 오버샘플링 (1:1)",
]

# 선택 전(기본 설정) = 2_분류_모델과 같은 설정
BASE_MISSING = "40%"
BASE_IMBALANCE = "클래스 가중치"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, encoding="utf-8")


class ColumnDropper(BaseEstimator, TransformerMixin):
    """학습 데이터에서만 상수 열과 결측률이 높은 열을 찾아 제거한다."""

    def __init__(self, missing_limit=0.4):
        self.missing_limit = missing_limit

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        is_varied = X.nunique(dropna=True) > 1
        is_complete = X.isna().mean() < self.missing_limit
        self.keep_cols_ = list(X.columns[is_varied & is_complete])
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.keep_cols_]


def make_model(model_name: str, missing_limit: float, class_weight) -> Pipeline:
    if model_name == "로지스틱 회귀":
        classifier = LogisticRegression(class_weight=class_weight, max_iter=2000)
    else:
        classifier = DecisionTreeClassifier(
            max_depth=3, class_weight=class_weight, random_state=42
        )
    return Pipeline(
        [
            ("drop", ColumnDropper(missing_limit)),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", classifier),
        ]
    )


def get_xy(df: pd.DataFrame):
    X = df.drop(columns=["Time", "Pass/Fail"])
    y = (df["Pass/Fail"] == 1).astype(int)  # 불량(1)을 양성으로
    return X, y


def resample_train(X_train: pd.DataFrame, y_train: pd.Series, method: str):
    """학습 데이터만 다시 뽑는다. 시험 데이터는 이 함수를 거치지 않는다."""
    pos_idx = y_train.index[y_train == 1]
    neg_idx = y_train.index[y_train == 0]

    if method.startswith("언더샘플링"):
        n_keep = min(len(neg_idx), 3 * len(pos_idx))
        kept_neg = pd.Series(neg_idx).sample(n=n_keep, random_state=42)
        new_index = list(pos_idx) + list(kept_neg)
    else:  # 랜덤 오버샘플링: 불량을 복원 추출해서 정상과 같은 수로
        extra_pos = pd.Series(pos_idx).sample(
            n=len(neg_idx) - len(pos_idx), replace=True, random_state=42
        )
        new_index = list(y_train.index) + list(extra_pos)
    return X_train.loc[new_index], y_train.loc[new_index]


@st.cache_data
def run_experiment(df: pd.DataFrame, model_name: str, missing_label: str, imbalance: str) -> dict:
    """층화 5겹 교차검증(2_분류_모델과 같은 분할). 5겹 평균 지표와 합산 혼동행렬을 돌려준다."""
    X, y = get_xy(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    class_weight = "balanced" if imbalance == "클래스 가중치" else None
    use_sampling = imbalance.startswith(("언더샘플링", "랜덤 오버샘플링"))

    recalls, precisions, f1s = [], [], []
    preds = pd.Series(0, index=y.index)
    for train_idx, test_idx in cv.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        if use_sampling:
            X_train, y_train = resample_train(X_train, y_train, imbalance)

        model = make_model(model_name, MISSING_OPTIONS[missing_label], class_weight)
        model.fit(X_train, y_train)

        y_test = y.iloc[test_idx]
        pred = model.predict(X.iloc[test_idx])
        preds.iloc[test_idx] = pred
        recalls.append(recall_score(y_test, pred, zero_division=0))
        precisions.append(precision_score(y_test, pred, zero_division=0))
        f1s.append(f1_score(y_test, pred, zero_division=0))

    return {
        "recall": sum(recalls) / len(recalls),
        "precision": sum(precisions) / len(precisions),
        "f1": sum(f1s) / len(f1s),
        "tp": int(((y == 1) & (preds == 1)).sum()),
        "fp": int(((y == 0) & (preds == 1)).sum()),
        "fn": int(((y == 1) & (preds == 0)).sum()),
        "tn": int(((y == 0) & (preds == 0)).sum()),
    }


def matrix_table(r: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [[f"TN {r['tn']:,}", f"FP {r['fp']:,}"], [f"FN {r['fn']:,}", f"TP {r['tp']:,}"]],
        index=["실제 정상(-1)", "실제 불량(1)"],
        columns=["예측 정상", "예측 불량"],
    )


def trade_off_text(d_tp: int, d_fp: int) -> str:
    if d_tp > 0 and d_fp > 0:
        return "불량을 더 찾는 대신 오경보도 늘었어요."
    if d_tp > 0 and d_fp <= 0:
        return "불량을 더 찾았고, 오경보는 늘지 않았어요."
    if d_tp < 0 and d_fp < 0:
        return "오경보는 줄었지만, 불량을 덜 찾아요."
    if d_tp < 0 and d_fp >= 0:
        return "불량을 덜 찾았고, 오경보는 줄지 않았어요."
    if d_tp == 0 and d_fp > 0:
        return "찾은 불량은 같은데 오경보만 늘었어요."
    if d_tp == 0 and d_fp < 0:
        return "찾은 불량은 같은데 오경보가 줄었어요."
    return "찾은 불량과 오경보 모두 같아요."


st.title(f"{ICON} 성능 높이기")
st.caption("결측 처리와 불균형 처리를 바꾸면 불량을 더 잘 찾는지, 같은 분할로 전·후를 비교해요.")

try:
    df = load_data()
except FileNotFoundError:
    st.error("uci-secom.csv 파일을 찾을 수 없어요. 저장소 맨 위 폴더에 있는지 확인해 주세요.")
    st.stop()

# ───────────── 입력 ─────────────
col_model, col_imb = st.columns(2)
with col_model:
    model_name = st.selectbox("모델 선택", MODEL_OPTIONS)
with col_imb:
    imbalance = st.selectbox("불균형 처리", IMBALANCE_OPTIONS, index=IMBALANCE_OPTIONS.index(BASE_IMBALANCE))

missing_label = st.select_slider(
    "결측 제외 기준 (결측률이 이 값 이상인 센서 열을 제거)",
    options=list(MISSING_OPTIONS.keys()),
    value=BASE_MISSING,
)
st.caption(
    f"선택 전(기본 설정)은 결측 {BASE_MISSING} 이상 제외 + {BASE_IMBALANCE}로, 2_분류_모델과 같아요. "
    "'제외 안 함'이어도 값이 하나뿐인 열은 제거해요."
)

# ───────────── 실험 ─────────────
with st.spinner("처음 한 번씩만 학습해요. 잠시 기다려 주세요..."):
    before = run_experiment(df, model_name, BASE_MISSING, BASE_IMBALANCE)
    after = run_experiment(df, model_name, missing_label, imbalance)

# ───────────── 지표 표 ─────────────
st.subheader("선택 전과 후")
result = pd.DataFrame(
    {
        "지표": ["재현율", "정밀도", "F1"],
        "선택 전(기본 설정)": [before["recall"], before["precision"], before["f1"]],
        "선택 후": [after["recall"], after["precision"], after["f1"]],
        "변화": [
            after["recall"] - before["recall"],
            after["precision"] - before["precision"],
            after["f1"] - before["f1"],
        ],
    }
)
st.dataframe(
    result,
    hide_index=True,
    use_container_width=True,
    column_config={
        "선택 전(기본 설정)": st.column_config.NumberColumn(format="%.3f"),
        "선택 후": st.column_config.NumberColumn(format="%.3f"),
        "변화": st.column_config.NumberColumn(format="%+.3f"),
    },
)
st.write(
    f"선택 전보다 재현율 {after['recall'] - before['recall']:+.3f}, "
    f"정밀도 {after['precision'] - before['precision']:+.3f}, "
    f"F1 {after['f1'] - before['f1']:+.3f} 바뀌었어요."
)
st.caption(
    "표는 5겹 평균이에요(2_분류_모델과 같음). "
    "언더·오버샘플링은 각 겹의 학습 데이터에만 적용하고, 시험 데이터는 그대로 둬요."
)

# ───────────── 혼동행렬 전·후 ─────────────
st.subheader("혼동행렬 전·후 (행: 실제, 열: 예측)")
left, right = st.columns(2)
with left:
    st.write("선택 전(기본 설정)")
    st.table(matrix_table(before))
with right:
    st.write("선택 후")
    st.table(matrix_table(after))

d_tp = after["tp"] - before["tp"]
d_fp = after["fp"] - before["fp"]
st.write(
    f"찾은 불량(TP) {before['tp']:,}개 → {after['tp']:,}개 ({d_tp:+,}), "
    f"오경보(FP) {before['fp']:,}개 → {after['fp']:,}개 ({d_fp:+,}). "
    f"{trade_off_text(d_tp, d_fp)}"
)
st.caption("혼동행렬은 5겹 시험 결과를 합친 값이에요(3_채점표와 같음).")
