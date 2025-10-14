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
    total = int(df[metric_cols].sum().sum())
    peak_row = df.loc[df[metric_cols].sum(axis=1).idxmax()]
    peak_dt = peak_row["datetime"]
    peak_val = int(peak_row[metric_cols].sum())
    daily_totals = df.groupby(df["Date"].dt.date)[metric_cols].sum().sum(axis=1)
    avg_daily = int(daily_totals.mean())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Vehicles (all gates)", f"{total:,}")
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
    st.subheader("Totals by Gate")
    st.altair_chart(chart, use_container_width=True)


def chart_stacked_by_day(df_long: pd.DataFrame):
    daily = df_long.groupby([df_long["Date"].dt.date, "Gate"])['Count'].sum().reset_index().rename(columns={"Date": "Day"})
    chart = (
        alt.Chart(daily)
        .mark_bar()
        .encode(
            x=alt.X("Day:T", title="Day"),
            y=alt.Y("Count:Q", stack=True, title="Vehicles"),
            color=alt.Color("Gate:N", title="Gate"),
            tooltip=["Day:T", "Gate:N", "Count:Q"],
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