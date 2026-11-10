from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

TITLE = "반도체 불량 탐지 실습실"
ICON = "🔬"

DATA_PATH = Path(__file__).parent.parent / "uci-secom.csv"
COLOR_PASS = "#4C78A8"
COLOR_FAIL = "#E45756"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, encoding="utf-8")


@st.cache_data
def mean_diff_table(df: pd.DataFrame) -> pd.DataFrame:
    """정상·불량 평균 차이를 합동 표준편차로 나눈 값(표준화 평균차). 값이 있는 데이터만 사용."""
    sensors = df.drop(columns=["Time", "Pass/Fail"])
    fail = sensors[df["Pass/Fail"] == 1]
    ok = sensors[df["Pass/Fail"] == -1]

    n1, n0 = fail.count(), ok.count()
    m1, m0 = fail.mean(), ok.mean()
    v1, v0 = fail.var(), ok.var()

    pooled_sd = (((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2)) ** 0.5
    d = (m1 - m0) / pooled_sd

    table = pd.DataFrame(
        {
            "센서": sensors.columns,
            "표준화 평균차": d.values,
            "불량 평균": m1.values,
            "정상 평균": m0.values,
            "값 있는 불량 수": n1.values,
            "값 있는 정상 수": n0.values,
        }
    )
    valid = (table["값 있는 불량 수"] >= 2) & (table["값 있는 정상 수"] >= 2)
    valid &= pooled_sd.values > 0
    table = table[valid & table["표준화 평균차"].notna()].copy()
    table["절댓값"] = table["표준화 평균차"].abs()
    return table.sort_values("절댓값", ascending=False).reset_index(drop=True)


@st.cache_data
def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """불량(1) 여부와 센서의 상관계수. 상수 열은 제외, 값이 있는 쌍만 사용."""
    sensors = df.drop(columns=["Time", "Pass/Fail"])
    y = (df["Pass/Fail"] == 1).astype(int)

    sensors = sensors.loc[:, sensors.nunique(dropna=True) > 1]
    r = sensors.corrwith(y)
    table = pd.DataFrame(
        {
            "센서": sensors.columns,
            "상관계수": r.values,
            "결측률(%)": (sensors.isna().mean() * 100).values,
            "값 있는 불량 수": sensors[y == 1].count().values,
        }
    )
    table = table[table["상관계수"].notna()].copy()
    table["절댓값"] = table["상관계수"].abs()
    table["신뢰"] = (table["값 있는 불량 수"] < 20).map({True: "신뢰 낮음", False: ""})
    return table.sort_values("절댓값", ascending=False).reset_index(drop=True)


st.title(f"{ICON} 탐색")
st.caption("모델을 만들기 전에 데이터의 불균형, 결측, 분포, 상관을 확인해요.")

try:
    df = load_data()
except FileNotFoundError:
    st.error("uci-secom.csv 파일을 찾을 수 없어요. 저장소 맨 위 폴더에 있는지 확인해 주세요.")
    st.stop()

sensor_cols = [c for c in df.columns if c not in ("Time", "Pass/Fail")]

tab1, tab2, tab3, tab4 = st.tabs(["불균형", "결측", "분포", "상관"])

# ───────────── 탭1 불균형 (①) ─────────────
with tab1:
    n_total = len(df)
    n_pass = int((df["Pass/Fail"] == -1).sum())
    n_fail = int((df["Pass/Fail"] == 1).sum())

    fig = go.Figure(
        go.Bar(
            x=["정상(-1)", "불량(1)"],
            y=[n_pass, n_fail],
            marker_color=[COLOR_PASS, COLOR_FAIL],
            text=[f"{n_pass:,}개", f"{n_fail:,}개"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="정상·불량 개수",
        yaxis_title="개수",
        yaxis_range=[0, max(n_pass, n_fail) * 1.15],
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write(
        f"정상 **{n_pass:,}개** ({n_pass / n_total * 100:.1f}%), "
        f"불량 **{n_fail:,}개** ({n_fail / n_total * 100:.1f}%)"
    )
    if n_fail > 0:
        st.write(f"정상 대 불량은 약 **{n_pass / n_fail:.0f} : 1**이에요.")
    else:
        st.write("불량 데이터가 없어서 비율을 계산할 수 없어요.")

# ───────────── 탭2 결측 (②) ─────────────
with tab2:
    missing_pct = df[sensor_cols].isna().mean() * 100

    threshold = st.slider("결측 기준(%)", min_value=0, max_value=100, value=50, step=1)
    n_over = int((missing_pct >= threshold).sum())

    fig = go.Figure(
        go.Histogram(
            x=missing_pct.values,
            xbins=dict(start=0, end=100, size=5),
            marker_color="#72B7B2",
        )
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="black")
    fig.update_layout(
        title="센서 열별 결측률 분포",
        xaxis_title="결측률(%)",
        yaxis_title="센서 열 수",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write(
        f"결측률이 **{threshold}% 이상**인 센서 열은 {len(sensor_cols)}개 중 **{n_over}개**예요."
    )
    st.caption("막대 하나는 결측률 5%p 구간이에요. 점선이 슬라이더로 고른 기준이에요.")

# ───────────── 탭3 분포 (③) ─────────────
with tab3:
    diff_table = mean_diff_table(df)

    options = [str(c) for c in sensor_cols]
    default_index = 0
    if len(diff_table) > 0 and str(diff_table.loc[0, "센서"]) in options:
        default_index = options.index(str(diff_table.loc[0, "센서"]))

    sensor = st.selectbox("센서 선택 (측정 대상 비공개)", options, index=default_index)

    values = df[sensor]
    valid_values = values.dropna()
    if valid_values.nunique() < 2:
        st.info("이 센서는 값이 하나뿐이거나 비어 있어서 분포를 그릴 수 없어요.")
    else:
        lo, hi = valid_values.quantile([0.01, 0.99])
        in_range = values.between(lo, hi)
        pass_vals = values[(df["Pass/Fail"] == -1) & in_range]
        fail_vals = values[(df["Pass/Fail"] == 1) & in_range]

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=pass_vals, name="정상", histnorm="probability density",
                opacity=0.6, marker_color=COLOR_PASS, nbinsx=30,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=fail_vals, name="불량", histnorm="probability density",
                opacity=0.6, marker_color=COLOR_FAIL, nbinsx=30,
            )
        )
        fig.update_layout(
            barmode="overlay",
            title=f"센서 {sensor}의 정상·불량 분포",
            xaxis_title=f"센서 {sensor} 값 (1~99 백분위 구간)",
            yaxis_title="밀도",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"정상 {len(pass_vals):,}개, 불량 {len(fail_vals):,}개의 값을 그렸어요. "
            "정상과 불량의 개수 차이가 커서 막대 높이는 개수가 아니라 밀도로 맞췄어요."
        )

    st.subheader("정상·불량 평균 차이 상위 10개")
    top10 = diff_table.head(10)[
        ["센서", "표준화 평균차", "불량 평균", "정상 평균", "값 있는 불량 수", "값 있는 정상 수"]
    ]
    st.dataframe(
        top10,
        hide_index=True,
        use_container_width=True,
        column_config={
            "표준화 평균차": st.column_config.NumberColumn(format="%.3f"),
            "불량 평균": st.column_config.NumberColumn(format="%.4g"),
            "정상 평균": st.column_config.NumberColumn(format="%.4g"),
        },
    )
    st.caption(
        "표준화 평균차 = (불량 평균 − 정상 평균) ÷ 합동 표준편차, 값이 있는 데이터만 사용했어요. "
        "절댓값이 클수록 두 그룹의 평균이 멀지만, 값이 있는 불량 수가 적으면 우연일 수 있어요."
    )

# ───────────── 탭4 상관 (④) ─────────────
with tab4:
    corr_table = correlation_table(df)
    top5 = corr_table.head(5)

    labels = [
        f"센서 {s}" + (" (신뢰 낮음)" if flag else "")
        for s, flag in zip(top5["센서"], top5["신뢰"])
    ]
    fig = go.Figure(
        go.Bar(
            x=top5["상관계수"][::-1],
            y=labels[::-1],
            orientation="h",
            marker_color=[COLOR_FAIL if r > 0 else COLOR_PASS for r in top5["상관계수"][::-1]],
            text=[f"{r:+.3f}" for r in top5["상관계수"][::-1]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="불량(1) 여부와 상관계수 절댓값 상위 5개 센서",
        xaxis_title="상관계수",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        top5[["센서", "상관계수", "결측률(%)", "값 있는 불량 수", "신뢰"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "상관계수": st.column_config.NumberColumn(format="%+.3f"),
            "결측률(%)": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption(
        "상수 열은 제외했고, 값이 있는 행끼리만 상관을 계산했어요. "
        "값이 있는 불량이 20개 미만이면 \"신뢰 낮음\"으로 표시해요. "
        "전체 데이터로 고른 순위라서, 모델에 쓸 때는 교차검증 안에서 다시 골라야 해요."
    )
