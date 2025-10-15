import os
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

DATA_FILE = "vehicle_traffic_data (1).csv"


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure correct dtypes
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df["Time"] = df["Time"].astype(str)

    # Parse start time as HH:MM from "HH:MM-HH:MM"
    start_times = df["Time"].str.split("-", n=1, expand=True)[0]
    df["start_time"] = start_times
    # Build a precise datetime combining date and start_time
    df["datetime"] = pd.to_datetime(df["Date"].dt.strftime("%Y-%m-%d") + " " + df["start_time"]) 
    # Extract hour for aggregations
    df["hour"] = df["datetime"].dt.hour

    # Identify metric columns (all numeric gate counters)
    non_metrics = {"Date", "Time", "start_time", "datetime", "hour"}
    metric_cols = [c for c in df.columns if c not in non_metrics]

    # Coerce metrics to numeric
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, metric_cols


def melt_long(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    return df.melt(
        id_vars=["Date", "Time", "datetime", "hour"],
        value_vars=metric_cols,
        var_name="Gate",
        value_name="Count",
    )


def kpis(df: pd.DataFrame, metric_cols: list[str]):
    # Peak interval (unchanged: considers all gates)
    peak_row = df.loc[df[metric_cols].sum(axis=1).idxmax()]
    peak_dt = peak_row["datetime"]
    peak_val = int(peak_row[metric_cols].sum())

    # Determine the two specific gates (Arch IN and Vallamai IN)
    preferred_cols: list[str] = []
    if "Arch_Gate_IN_D30" in metric_cols:
        preferred_cols.append("Arch_Gate_IN_D30")
    # Assume D13 represents Vallamai IN when available; fallback to D16
    if "Vallamai_Main_Gate_D13" in metric_cols:
        preferred_cols.append("Vallamai_Main_Gate_D13")
    elif "Vallamai_Main_Gate_D16" in metric_cols:
        preferred_cols.append("Vallamai_Main_Gate_D16")

    # Fallback: try to infer sensible columns with pattern matching
    selected_cols = preferred_cols if len(preferred_cols) == 2 else []
    if not selected_cols:
        arch_candidates = [
            c for c in metric_cols
            if "Arch" in c and "IN" in c and "IN_OUT" not in c and "Outside" not in c
        ]
        vall_candidates = [
            c for c in metric_cols
            if "Vallamai" in c and ("IN" in c or "Gate_D13" in c or "Gate_D16" in c)
        ]
        if arch_candidates:
            selected_cols.append(arch_candidates[0])
        if vall_candidates:
            selected_cols.append(vall_candidates[0])

    # If we still can't identify, fall back to all gates with a warning
    if len(selected_cols) != 2:
        st.warning(
            "Could not identify Arch IN and Vallamai IN precisely; falling back to all gates for totals and average.",
            icon="⚠️",
        )
        selected_cols = metric_cols

    # Compute totals using ONLY the selected two gates when identified
    total_selected = int(df[selected_cols].sum().sum())

    # Compute Avg Daily for the selected gates
    daily_subset = df.groupby(df["Date"].dt.date)[selected_cols].sum()
    avg_daily = int(daily_subset.sum(axis=1).mean())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Vehicles (Arch IN + Vallamai IN)", f"{total_selected:,}")
    c2.metric("Peak Interval", peak_dt.strftime("%Y-%m-%d %H:%M"), delta=f"{peak_val:,}")
    c3.metric("Avg Daily Volume", f"{avg_daily:,}")


def sidebar_filters(df: pd.DataFrame, metric_cols: list[str]):
    st.sidebar.header("Filters")

    # Date range
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    # Time slots
    time_options = df["Time"].unique().tolist()
    selected_times = st.sidebar.multiselect(
        "Time slots",
        options=time_options,
        default=time_options,
    )

    # Metric selection
    selected_metrics = st.sidebar.multiselect(
        "Gate metrics",
        options=metric_cols,
        default=metric_cols[: min(3, len(metric_cols))] or metric_cols,
    )

    # Aggregation grain
    grain = st.sidebar.radio("Aggregate by", ["Interval", "Date", "Hour"], index=0)

    return {
        "start_date": start_date,
        "end_date": end_date,
        "selected_times": selected_times,
        "selected_metrics": selected_metrics,
        "grain": grain,
    }


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    mask = (
        (df["Date"].dt.date >= f["start_date"]) &
        (df["Date"].dt.date <= f["end_date"]) &
        (df["Time"].isin(f["selected_times"]))
    )
    return df.loc[mask].copy()


def chart_time_series(df_long: pd.DataFrame, selected_metrics: list[str]):
    ts = df_long[df_long["Gate"].isin(selected_metrics)]
    chart = (
        alt.Chart(ts)
        .mark_line(point=True)
        .encode(
            x=alt.X("datetime:T", title="Time"),
            y=alt.Y("Count:Q", title="Vehicles"),
            color=alt.Color("Gate:N", title="Gate"),
            tooltip=["Date:T", "Time:N", "Gate:N", "Count:Q"],
        )
        .properties(height=300)
    )
    st.subheader("Time Series by Gate")
    st.altair_chart(chart, use_container_width=True)


def chart_daily_totals_with_average(df_long: pd.DataFrame, df_full: pd.DataFrame, metric_cols: list[str]):
    """Bar chart of daily totals using ONLY Arch IN and Vallamai IN, plus average.

    - Bars: computed from the current filtered dataset (df_long) but restricted
      to the Arch IN and Vallamai IN gates only.
    - Average line: computed across ALL available days from the full dataset
      (df_full) for those same gates, regardless of current filters.
    """

    # Identify the exact gate metric names
    arch_gate = "Arch_Gate_IN_D30" if "Arch_Gate_IN_D30" in metric_cols else None
    vall_gate = None
    if "Vallamai_Main_Gate_D13" in metric_cols:
        vall_gate = "Vallamai_Main_Gate_D13"
    elif "Vallamai_Main_Gate_D16" in metric_cols:
        vall_gate = "Vallamai_Main_Gate_D16"

    # Fallback detection if explicit names are missing
    if arch_gate is None:
        arch_candidates = [
            c for c in metric_cols
            if "Arch" in c and "IN" in c and "IN_OUT" not in c and "Outside" not in c
        ]
        arch_gate = arch_candidates[0] if arch_candidates else None
    if vall_gate is None:
        vall_candidates = [
            c for c in metric_cols
            if "Vallamai" in c and ("IN" in c or "Gate_D13" in c or "Gate_D16" in c)
        ]
        vall_gate = vall_candidates[0] if vall_candidates else None

    selected_two = [g for g in [arch_gate, vall_gate] if g]
    if not selected_two:
        st.warning("Could not find Arch IN and Vallamai IN metrics.", icon="⚠️")
        return

    # Bars: use filtered long data restricted to those gates
    subset = df_long[df_long["Gate"].isin(selected_two)]
    daily = (
        subset.groupby(subset["Date"].dt.date)["Count"].sum().reset_index()
        .rename(columns={"Date": "Day", "Count": "Vehicles"})
    )
    # Use a string-formatted date for a discrete X axis with exactly one bar per day
    daily["Day_str"] = pd.to_datetime(daily["Day"]).dt.strftime("%Y-%m-%d")

    # Average line: compute using ALL days in the full dataset (wide form)
    daily_full = df_full.groupby(df_full["Date"].dt.date)[selected_two].sum().sum(axis=1)
    avg_all_days = float(daily_full.mean()) if len(daily_full) else 0.0

    bars = (
        alt.Chart(daily)
        .mark_bar()
        .encode(
            x=alt.X("Day_str:N", title="Date"),
            y=alt.Y("Vehicles:Q", title="Vehicles"),
            tooltip=[alt.Tooltip("Day_str:N", title="Date"), "Vehicles:Q"],
        )
        .properties(height=260)
    )

    avg_rule = (
        alt.Chart(pd.DataFrame({"avg": [avg_all_days]}))
        .mark_rule(color="#d62728", strokeDash=[6, 4])
        .encode(y="avg:Q")
    )

    st.subheader("Daily Vehicles (Bar) with Average")
    st.altair_chart(bars + avg_rule, use_container_width=True)
    st.caption(f"Daily average (Arch IN + Vallamai IN): {int(round(avg_all_days)):,}")


def chart_bar_totals(df_long: pd.DataFrame):
    totals = df_long.groupby("Gate")["Count"].sum().reset_index()
    chart = (
        alt.Chart(totals)
        .mark_bar()
        .encode(
            x=alt.X("Gate:N", title="Gate"),
            y=alt.Y("Count:Q", title="Total Vehicles"),
            color="Gate:N",
            tooltip=["Gate:N", "Count:Q"],
        )
        .properties(height=280)
    )
    st.subheader("Totals by Camera")
    st.altair_chart(chart, use_container_width=True)


def chart_stacked_by_day(df_long: pd.DataFrame):
    daily = df_long.groupby([df_long["Date"].dt.date, "Gate"])['Count'].sum().reset_index().rename(columns={"Date": "Day"})
    # Use ISO date string for a discrete axis (exactly one bar per day, no time)
    daily["Day_str"] = pd.to_datetime(daily["Day"]).dt.strftime("%Y-%m-%d")
    chart = (
        alt.Chart(daily)
        .mark_bar()
        .encode(
            x=alt.X("Day_str:N", title="Date"),
            y=alt.Y("Count:Q", stack=True, title="Vehicles"),
            color=alt.Color("Gate:N", title="Gate"),
            tooltip=[alt.Tooltip("Day_str:N", title="Date"), "Gate:N", "Count:Q"],
        )
        .properties(height=280)
    )
    st.subheader("Stacked Daily Volumes by Gate")
    st.altair_chart(chart, use_container_width=True)


def chart_heatmap(df_long: pd.DataFrame):
    heat = df_long.groupby([df_long["Date"].dt.date, "Time"])['Count'].sum().reset_index().rename(columns={"Date": "Day"})
    chart = (
        alt.Chart(heat)
        .mark_rect()
        .encode(
            x=alt.X("Time:N", title="Time Interval"),
            y=alt.Y("Day:T", title="Day"),
            color=alt.Color("Count:Q", title="Vehicles", scale=alt.Scale(scheme="goldred")),
            tooltip=["Day:T", "Time:N", "Count:Q"],
        )
        .properties(height=300)
    )
    st.subheader("Heatmap: Vehicles by Day and Time")
    st.altair_chart(chart, use_container_width=True)


def chart_pie(df_long: pd.DataFrame):
    totals = df_long.groupby("Gate")["Count"].sum().reset_index()
    chart = (
        alt.Chart(totals)
        .mark_arc(innerRadius=40)
        .encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color("Gate:N"),
            tooltip=["Gate:N", "Count:Q"],
        )
        .properties(height=280)
    )
    st.subheader("Share of Traffic by Gate")
    st.altair_chart(chart, use_container_width=True)


def chart_boxplot(df_long: pd.DataFrame):
    st.subheader("Distribution by Gate (Boxplot)")
    chart = (
        alt.Chart(df_long)
        .mark_boxplot()
        .encode(
            x=alt.X("Gate:N", title="Gate"),
            y=alt.Y("Count:Q", title="Vehicles"),
            color="Gate:N",
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)


def chart_correlation(df: pd.DataFrame, metric_cols: list[str]):
    st.subheader("Correlation Between Gates")
    corr = df[metric_cols].corr()
    corr_df = corr.stack().reset_index()
    corr_df.columns = ["Gate_X", "Gate_Y", "Correlation"]
    chart = (
        alt.Chart(corr_df)
        .mark_rect()
        .encode(
            x=alt.X("Gate_X:N", title="Gate X"),
            y=alt.Y("Gate_Y:N", title="Gate Y"),
            color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="blueorange", domain=[-1, 1])),
            tooltip=["Gate_X:N", "Gate_Y:N", alt.Tooltip("Correlation:Q", format=".2f")],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def main():
    st.set_page_config(page_title="Campus Traffic Dashboard", layout="wide")
    st.title("Campus Traffic Dashboard")
    st.caption("Interactive visualizations for campus gate traffic using the provided CSV data.")

    data_path = os.path.join(os.getcwd(), DATA_FILE)
    if not os.path.exists(data_path):
        st.error(f"Data file not found: {DATA_FILE}")
        return

    df, metric_cols = load_data(data_path)

    # KPIs
    kpis(df, metric_cols)

    # Filters
    filters = sidebar_filters(df, metric_cols)
    df_f = apply_filters(df, filters)
    df_long = melt_long(df_f, metric_cols)

    # Charts
    chart_time_series(df_long, filters["selected_metrics"])
    chart_daily_totals_with_average(df_long, df, metric_cols)
    c1, c2 = st.columns(2)
    with c1:
        chart_bar_totals(df_long)
    with c2:
        chart_pie(df_long)

    chart_stacked_by_day(df_long)
    chart_heatmap(df_long)
    chart_boxplot(df_long)
    chart_correlation(df_f, metric_cols)

    # Data table and download
    st.subheader("Filtered Data Preview")
    st.dataframe(df_f)
    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", csv, file_name="filtered_vehicle_traffic.csv", mime="text/csv")


if __name__ == "__main__":
    main()