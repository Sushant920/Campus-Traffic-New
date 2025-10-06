import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io

# Page configuration
st.set_page_config(
    page_title="Campus Traffic Simulation & Analytics",
    page_icon="🚶‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Location data with aggregate clips
LOCATION_DATA = {
    "Arch Gate IN-D30": 322,
    "Arch Gate OUT-D32": 279,
    "Chola Statue OUT-D-26": 629,
    "Arch Gate Outside IN D36": 31,
    "Arch Gate IN-OUT D35": 472,
    "Chola Statue IN D22": 322,
    "Vallamai Main Gate D13": 339,
    "Vallamai Main Gate D16": 55
}

# Peak hour definitions
PEAK_HOURS = {
    "morning": {"start": 7.5, "end": 8.5, "multiplier": 4.0},  # 7:30-8:30 AM
    "afternoon": {"start": 12.5, "end": 13.5, "multiplier": 3.5},  # 12:30-1:30 PM
    "evening": {"start": 16.5, "end": 17.5, "multiplier": 4.5}  # 4:30-5:30 PM
}

# Location-specific peak preferences
LOCATION_PEAKS = {
    "Arch Gate IN-D30": ["morning", "evening"],
    "Arch Gate OUT-D32": ["afternoon", "evening"],
    "Chola Statue OUT-D-26": ["evening"],
    "Arch Gate Outside IN D36": ["morning"],
    "Arch Gate IN-OUT D35": ["morning", "afternoon", "evening"],
    "Chola Statue IN D22": ["morning"],
    "Vallamai Main Gate D13": ["morning", "afternoon"],
    "Vallamai Main Gate D16": ["afternoon"]
}

# Daily variation factors to create different congestion levels
DAILY_FACTORS = {
    "2025-09-02": {"name": "Tuesday", "factor": 1.0, "description": "Normal Traffic"},      # Baseline
    "2025-09-03": {"name": "Wednesday", "factor": 1.15, "description": "High Traffic"},    # Busiest day
    "2025-09-04": {"name": "Thursday", "factor": 0.85, "description": "Light Traffic"},    # Lighter day
    "2025-09-05": {"name": "Friday", "factor": 1.05, "description": "Moderate Traffic"}    # End of week
}

@st.cache_data
def simulate_traffic_data(holiday_days=None):
    """
    Simulate traffic data based on aggregate counts and peak hour analysis
    with daily variation factors for different congestion levels
    """
    if holiday_days is None:
        holiday_days = []
    
    # Date range: Sept 2-5, 2025 (Tuesday to Friday)
    start_date = datetime(2025, 9, 2)
    dates = [start_date + timedelta(days=i) for i in range(4)]
    
    # Create 30-minute intervals for each day
    intervals = []
    for hour in range(24):
        for minute in [0, 30]:
            time_str = f"{hour:02d}:{minute:02d}"
            intervals.append((hour + minute/60, time_str))
    
    data = []
    
    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        is_holiday = date_str in holiday_days
        
        # Get daily variation factor
        daily_info = DAILY_FACTORS.get(date_str, {"factor": 1.0, "description": "Normal Traffic"})
        daily_factor = daily_info["factor"]
        
        for location, total_clips in LOCATION_DATA.items():
            # Calculate daily average with daily variation
            daily_avg = (total_clips / 4) * daily_factor
            
            # Holiday reduction
            if is_holiday:
                daily_avg *= 0.15  # Reduce to 15% for holidays
            
            # Distribute across intervals
            interval_counts = []
            
            for hour_decimal, time_str in intervals:
                # Base count (evenly distributed)
                base_count = daily_avg / 48
                
                # Apply peak multipliers with enhanced time-based patterns
                multiplier = 1.0
                
                # Enhanced time-based traffic patterns
                if 6 <= hour_decimal < 9:  # Morning rush (6-9 AM)
                    if hour_decimal >= 7.5 and hour_decimal <= 8.5:  # Peak morning
                        multiplier = 5.0
                    else:
                        multiplier = 3.0
                elif 11 <= hour_decimal < 14:  # Lunch period (11 AM - 2 PM)
                    if hour_decimal >= 12.5 and hour_decimal <= 13.5:  # Peak lunch
                        multiplier = 4.0
                    else:
                        multiplier = 2.5
                elif 16 <= hour_decimal < 19:  # Evening rush (4-7 PM)
                    if hour_decimal >= 16.5 and hour_decimal <= 17.5:  # Peak evening
                        multiplier = 5.5
                    else:
                        multiplier = 3.5
                elif 9 <= hour_decimal < 11 or 14 <= hour_decimal < 16:  # Mid-morning/afternoon
                    multiplier = 1.8
                elif hour_decimal >= 19 and hour_decimal < 22:  # Evening (7-10 PM)
                    multiplier = 1.2
                else:  # Night hours (10 PM - 6 AM)
                    multiplier = 0.05
                
                # Location-specific adjustments
                location_peaks = LOCATION_PEAKS.get(location, [])
                location_multiplier = 1.0
                
                for peak_name, peak_info in PEAK_HOURS.items():
                    if (peak_info["start"] <= hour_decimal <= peak_info["end"] and 
                        peak_name in location_peaks):
                        location_multiplier = max(location_multiplier, 1.3)  # Boost for location-specific peaks
                
                multiplier *= location_multiplier
                
                # Calculate count with enhanced variability
                count = base_count * multiplier
                
                # Add daily-specific noise patterns
                if daily_factor > 1.1:  # High traffic days - more variability
                    noise = np.random.uniform(0.7, 1.4)
                elif daily_factor < 0.9:  # Light traffic days - less variability
                    noise = np.random.uniform(0.9, 1.1)
                else:  # Normal days
                    noise = np.random.uniform(0.8, 1.2)
                
                count *= noise
                
                # Ensure non-negative and round
                count = max(0, round(count))
                
                interval_counts.append(count)
            
            # Normalize to match aggregate total (approximately)
            current_total = sum(interval_counts)
            if current_total > 0:
                scaling_factor = daily_avg / current_total
                interval_counts = [max(0, round(count * scaling_factor)) for count in interval_counts]
            
            # Add to data
            for i, (hour_decimal, time_str) in enumerate(intervals):
                end_hour = hour_decimal + 0.5
                end_time_str = f"{int(end_hour):02d}:{int((end_hour % 1) * 60):02d}"
                
                # Classify traffic intensity
                count = interval_counts[i]
                avg_for_location = daily_avg / 48
                
                if count >= avg_for_location * 3:
                    intensity = "Very High"
                elif count >= avg_for_location * 2:
                    intensity = "High"
                elif count >= avg_for_location * 1.2:
                    intensity = "Moderate"
                elif count >= avg_for_location * 0.5:
                    intensity = "Low"
                else:
                    intensity = "Very Low"
                
                data.append({
                    "Date": date_str,
                    "Location": location,
                    "Time_Interval": f"{time_str}-{end_time_str}",
                    "Hour_Decimal": hour_decimal,
                    "Hour_Display": f"{int(hour_decimal):02d}:{int((hour_decimal % 1) * 60):02d}",
                    "Simulated_Clip_Count": count,
                    "Day_Name": date.strftime("%A"),
                    "Daily_Factor": daily_factor,
                    "Daily_Description": daily_info["description"],
                    "Traffic_Intensity": intensity
                })
    
    return pd.DataFrame(data)

def calculate_analytics(df):
    """Calculate various analytics metrics"""
    analytics = {}
    
    # Total clips
    analytics["total_clips"] = df["Simulated_Clip_Count"].sum()
    
    # Average daily clips per location
    daily_totals = df.groupby(["Date", "Location"])["Simulated_Clip_Count"].sum().reset_index()
    analytics["avg_daily_per_location"] = daily_totals["Simulated_Clip_Count"].mean()
    
    # Peak detection (intervals with >2x average)
    avg_count = df["Simulated_Clip_Count"].mean()
    peak_threshold = avg_count * 2
    peaks = df[df["Simulated_Clip_Count"] > peak_threshold].copy()
    analytics["peak_intervals"] = len(peaks)
    
    # Top 10 busiest intervals
    analytics["top_intervals"] = df.nlargest(10, "Simulated_Clip_Count")[
        ["Date", "Location", "Time_Interval", "Simulated_Clip_Count"]
    ]
    
    # Location totals
    analytics["location_totals"] = df.groupby("Location")["Simulated_Clip_Count"].sum().sort_values(ascending=False)
    
    # Daily totals
    analytics["daily_totals"] = df.groupby("Date")["Simulated_Clip_Count"].sum()
    
    # Congestion index (% of intervals above threshold)
    total_intervals = len(df)
    congested_intervals = len(df[df["Simulated_Clip_Count"] > avg_count])
    analytics["congestion_index"] = (congested_intervals / total_intervals) * 100
    
    return analytics

def create_hourly_traffic_overview(df):
    """Create comprehensive hourly traffic overview"""
    # Aggregate traffic by hour across all locations and days
    hourly_data = df.groupby('Hour_Decimal')['Simulated_Clip_Count'].agg(['sum', 'mean']).reset_index()
    hourly_data['Hour_Display'] = hourly_data['Hour_Decimal'].apply(
        lambda x: f"{int(x):02d}:{int((x % 1) * 60):02d}"
    )
    
    # Create the main hourly overview chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Total Campus Traffic by Hour', 'Average Traffic Intensity by Hour'),
        vertical_spacing=0.12
    )
    
    # Total traffic line
    fig.add_trace(
        go.Scatter(
            x=hourly_data['Hour_Display'],
            y=hourly_data['sum'],
            mode='lines+markers',
            name='Total Traffic',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Time:</b> %{x}<br><b>Total Traffic:</b> %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Average traffic line
    fig.add_trace(
        go.Scatter(
            x=hourly_data['Hour_Display'],
            y=hourly_data['mean'],
            mode='lines+markers',
            name='Average Traffic',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Time:</b> %{x}<br><b>Average Traffic:</b> %{y:.1f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add peak hour shading
    peak_colors = {'morning': 'rgba(255, 0, 0, 0.1)', 'afternoon': 'rgba(0, 255, 0, 0.1)', 'evening': 'rgba(0, 0, 255, 0.1)'}
    
    for peak_name, peak_info in PEAK_HOURS.items():
        start_idx = int(peak_info["start"] * 2)  # Convert to 30-min intervals
        end_idx = int(peak_info["end"] * 2)
        
        for row in [1, 2]:
            fig.add_vrect(
                x0=hourly_data.iloc[start_idx]['Hour_Display'],
                x1=hourly_data.iloc[end_idx]['Hour_Display'],
                fillcolor=peak_colors.get(peak_name, 'rgba(128, 128, 128, 0.1)'),
                opacity=0.3,
                layer="below",
                line_width=0,
                row=row, col=1
            )
    
    fig.update_layout(
        height=600,
        title_text="Campus Traffic Patterns Throughout the Day",
        showlegend=True
    )
    
    # Update x-axes to show fewer labels
    for row in [1, 2]:
        fig.update_xaxes(
            tickmode='array',
            tickvals=[hourly_data.iloc[i]['Hour_Display'] for i in range(0, len(hourly_data), 4)],
            ticktext=[hourly_data.iloc[i]['Hour_Display'] for i in range(0, len(hourly_data), 4)],
            row=row, col=1
        )
    
    return fig

def create_daily_comparison_chart(df):
    """Create daily comparison showing different congestion levels"""
    daily_hourly = df.groupby(['Date', 'Hour_Decimal', 'Daily_Description'])['Simulated_Clip_Count'].sum().reset_index()
    daily_hourly['Hour_Display'] = daily_hourly['Hour_Decimal'].apply(
        lambda x: f"{int(x):02d}:{int((x % 1) * 60):02d}"
    )
    
    fig = px.line(
        daily_hourly,
        x='Hour_Display',
        y='Simulated_Clip_Count',
        color='Daily_Description',
        title='Daily Traffic Comparison - Different Congestion Levels',
        labels={'Simulated_Clip_Count': 'Total Traffic', 'Hour_Display': 'Time of Day'},
        line_shape='spline'
    )
    
    # Customize colors for different traffic levels
    color_map = {
        'Normal Traffic': '#1f77b4',
        'High Traffic': '#d62728',
        'Light Traffic': '#2ca02c',
        'Moderate Traffic': '#ff7f0e'
    }
    
    fig.for_each_trace(lambda trace: trace.update(line_color=color_map.get(trace.name, trace.line.color)))
    
    # Add peak hour annotations
    for peak_name, peak_info in PEAK_HOURS.items():
        fig.add_vrect(
            x0=f"{int(peak_info['start']):02d}:{int((peak_info['start'] % 1) * 60):02d}",
            x1=f"{int(peak_info['end']):02d}:{int((peak_info['end'] % 1) * 60):02d}",
            fillcolor="red", opacity=0.1,
            annotation_text=f"{peak_name.title()}",
            annotation_position="top left"
        )
    
    # Update x-axis to show fewer labels
    fig.update_xaxes(
        tickmode='array',
        tickvals=[f"{i:02d}:00" for i in range(0, 24, 3)],
        ticktext=[f"{i:02d}:00" for i in range(0, 24, 3)]
    )
    
    fig.update_layout(height=500)
    return fig

def create_traffic_intensity_heatmap(df):
    """Create heatmap showing traffic intensity throughout the day"""
    # Create intensity mapping
    intensity_map = {'Very Low': 1, 'Low': 2, 'Moderate': 3, 'High': 4, 'Very High': 5}
    df_intensity = df.copy()
    df_intensity['Intensity_Value'] = df_intensity['Traffic_Intensity'].map(intensity_map)
    
    # Pivot for heatmap
    heatmap_data = df_intensity.pivot_table(
        values='Intensity_Value',
        index='Hour_Display',
        columns='Daily_Description',
        aggfunc='mean'
    )
    
    fig = px.imshow(
        heatmap_data.T,
        title='Traffic Intensity Heatmap - Time vs Daily Patterns',
        labels={'x': 'Time of Day', 'y': 'Daily Pattern', 'color': 'Intensity Level'},
        color_continuous_scale='RdYlBu_r',
        aspect='auto'
    )
    
    # Update colorbar
    fig.update_coloraxes(
        colorbar_title="Traffic Intensity",
        colorbar_tickmode="array",
        colorbar_tickvals=[1, 2, 3, 4, 5],
        colorbar_ticktext=["Very Low", "Low", "Moderate", "High", "Very High"]
    )
    
    # Show only every 4th time label
    fig.update_xaxes(
        tickmode='array',
        tickvals=list(range(0, len(heatmap_data.index), 4)),
        ticktext=[heatmap_data.index[i] for i in range(0, len(heatmap_data.index), 4)]
    )
    
    return fig

def create_time_series_chart(df, selected_locations, selected_dates):
    """Create interactive time-series line chart"""
    filtered_df = df[
        (df["Location"].isin(selected_locations)) & 
        (df["Date"].isin(selected_dates))
    ].copy()
    
    if filtered_df.empty:
        return go.Figure().add_annotation(text="No data available for selected filters", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    fig = px.line(
        filtered_df, 
        x="Hour_Decimal", 
        y="Simulated_Clip_Count",
        color="Location",
        facet_col="Date",
        title="Traffic Patterns Over Time (30-minute intervals)",
        labels={"Hour_Decimal": "Hour of Day", "Simulated_Clip_Count": "Clip Count"}
    )
    
    # Add peak hour annotations
    for peak_name, peak_info in PEAK_HOURS.items():
        fig.add_vrect(
            x0=peak_info["start"], x1=peak_info["end"],
            fillcolor="red", opacity=0.1,
            annotation_text=f"{peak_name.title()} Peak",
            annotation_position="top left"
        )
    
    fig.update_layout(height=500)
    return fig

def create_location_bar_chart(analytics):
    """Create bar chart of total clips per location"""
    fig = px.bar(
        x=analytics["location_totals"].index,
        y=analytics["location_totals"].values,
        title="Total Clips by Location",
        labels={"x": "Location", "y": "Total Clips"}
    )
    fig.update_xaxes(tickangle=45)
    return fig

def create_heatmap(df, selected_locations):
    """Create heatmap of activity by time and location"""
    filtered_df = df[df["Location"].isin(selected_locations)].copy()
    
    if filtered_df.empty:
        return go.Figure().add_annotation(text="No data available for selected locations", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    # Aggregate by time interval and location
    heatmap_data = filtered_df.groupby(["Time_Interval", "Location"])["Simulated_Clip_Count"].sum().unstack(fill_value=0)
    
    fig = px.imshow(
        heatmap_data.T,
        title="Activity Heatmap (Location vs Time)",
        labels={"x": "Time Interval", "y": "Location", "color": "Clip Count"},
        aspect="auto"
    )
    
    # Show only every 4th time label to avoid crowding
    fig.update_xaxes(tickmode='array', tickvals=list(range(0, len(heatmap_data.index), 4)),
                     ticktext=[heatmap_data.index[i] for i in range(0, len(heatmap_data.index), 4)])
    
    return fig

def create_peak_distribution_chart(df):
    """Create stacked bar chart showing peak vs off-peak distribution"""
    df_copy = df.copy()
    
    # Classify intervals as peak or off-peak
    def classify_interval(hour):
        for peak_name, peak_info in PEAK_HOURS.items():
            if peak_info["start"] <= hour <= peak_info["end"]:
                return f"{peak_name.title()} Peak"
        return "Off-Peak"
    
    df_copy["Period_Type"] = df_copy["Hour_Decimal"].apply(classify_interval)
    
    # Aggregate by location and period type
    period_data = df_copy.groupby(["Location", "Period_Type"])["Simulated_Clip_Count"].sum().unstack(fill_value=0)
    
    fig = go.Figure()
    
    for period in period_data.columns:
        fig.add_trace(go.Bar(
            name=period,
            x=period_data.index,
            y=period_data[period]
        ))
    
    fig.update_layout(
        title="Traffic Distribution by Peak Periods",
        xaxis_title="Location",
        yaxis_title="Total Clips",
        barmode='stack',
        xaxis_tickangle=45
    )
    
    return fig

def create_pie_chart(analytics):
    """Create pie chart of clip distribution across locations"""
    fig = px.pie(
        values=analytics["location_totals"].values,
        names=analytics["location_totals"].index,
        title="Distribution of Total Clips Across Locations"
    )
    return fig

# Main Streamlit App
def main():
    st.title("🚶‍♂️ Campus Traffic Simulation & Analytics")
    st.markdown("*Simulated traffic data based on CCTV footage analysis (Sept 2-5, 2025)*")
    
    # Sidebar filters
    st.sidebar.header("📊 Filters & Controls")
    
    # Location filter
    all_locations = list(LOCATION_DATA.keys())
    selected_locations = st.sidebar.multiselect(
        "Select Locations",
        all_locations,
        default=all_locations
    )
    
    # Date filter
    all_dates = ["2025-09-02", "2025-09-03", "2025-09-04", "2025-09-05"]
    selected_dates = st.sidebar.multiselect(
        "Select Dates",
        all_dates,
        default=all_dates
    )
    
    # Holiday simulation
    st.sidebar.subheader("🏖️ Holiday Simulation")
    holiday_days = []
    for date in all_dates:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        if st.sidebar.checkbox(f"{date_obj.strftime('%A, %b %d')}", key=f"holiday_{date}"):
            holiday_days.append(date)
    
    # Time range filter
    time_range = st.sidebar.slider(
        "Time Range (Hours)",
        min_value=0.0,
        max_value=24.0,
        value=(0.0, 24.0),
        step=0.5
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Simulation"):
        st.cache_data.clear()
    
    # Generate data
    with st.spinner("Generating simulation data..."):
        df = simulate_traffic_data(holiday_days)
    
    # Apply time filter
    df_filtered = df[
        (df["Hour_Decimal"] >= time_range[0]) & 
        (df["Hour_Decimal"] <= time_range[1]) &
        (df["Location"].isin(selected_locations)) &
        (df["Date"].isin(selected_dates))
    ].copy()
    
    # Calculate analytics
    analytics = calculate_analytics(df_filtered)
    
    # Dashboard Overview
    st.header("📈 Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Clips", f"{analytics['total_clips']:,}")
    
    with col2:
        st.metric("Avg Daily/Location", f"{analytics['avg_daily_per_location']:.1f}")
    
    with col3:
        st.metric("Peak Intervals", analytics['peak_intervals'])
    
    with col4:
        st.metric("Congestion Index", f"{analytics['congestion_index']:.1f}%")
    
    # Daily Traffic Overview Section
    st.header("🕐 Time-Based Traffic Analysis")
    st.markdown("*Focus on traffic patterns throughout the day with daily variations*")
    
    # Show daily factors info
    st.subheader("📅 Daily Congestion Levels")
    daily_info_cols = st.columns(4)
    for i, (date, info) in enumerate(DAILY_FACTORS.items()):
        with daily_info_cols[i]:
            color = {"Normal Traffic": "🟢", "High Traffic": "🔴", "Light Traffic": "🟡", "Moderate Traffic": "🟠"}
            st.metric(
                f"{info['name']} ({date[-5:]})",
                info['description'],
                delta=f"{(info['factor']-1)*100:+.0f}%" if info['factor'] != 1.0 else "Baseline"
            )
    
    # Main hourly traffic overview
    st.subheader("🚶‍♂️ Campus Traffic Throughout the Day")
    hourly_overview_fig = create_hourly_traffic_overview(df_filtered)
    st.plotly_chart(hourly_overview_fig, use_container_width=True)
    
    # Daily comparison
    st.subheader("📊 Daily Traffic Comparison")
    daily_comparison_fig = create_daily_comparison_chart(df_filtered)
    st.plotly_chart(daily_comparison_fig, use_container_width=True)
    
    # Traffic intensity heatmap
    st.subheader("🌡️ Traffic Intensity Heatmap")
    intensity_heatmap_fig = create_traffic_intensity_heatmap(df_filtered)
    st.plotly_chart(intensity_heatmap_fig, use_container_width=True)
    
    # Additional Visualizations
    st.header("📊 Additional Analysis")
    
    # Time Series Chart
    st.subheader("Location-Specific Time Analysis")
    if selected_locations and selected_dates:
        time_series_fig = create_time_series_chart(df, selected_locations, selected_dates)
        st.plotly_chart(time_series_fig, use_container_width=True)
    else:
        st.warning("Please select at least one location and date.")
    
    # Two column layout for other charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Location Totals")
        if not analytics["location_totals"].empty:
            location_bar_fig = create_location_bar_chart(analytics)
            st.plotly_chart(location_bar_fig, use_container_width=True)
    
    with col2:
        st.subheader("Distribution by Location")
        if not analytics["location_totals"].empty:
            pie_fig = create_pie_chart(analytics)
            st.plotly_chart(pie_fig, use_container_width=True)
    
    # Heatmap
    st.subheader("Activity Heatmap")
    if selected_locations:
        heatmap_fig = create_heatmap(df, selected_locations)
        st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Peak Distribution
    st.subheader("Peak Period Analysis")
    peak_dist_fig = create_peak_distribution_chart(df_filtered)
    st.plotly_chart(peak_dist_fig, use_container_width=True)
    
    # Top Intervals Table
    st.subheader("🔥 Top 10 Busiest Intervals")
    st.dataframe(analytics["top_intervals"], use_container_width=True)
    
    # Daily Trends
    st.subheader("📅 Daily Trends")
    daily_fig = px.line(
        x=analytics["daily_totals"].index,
        y=analytics["daily_totals"].values,
        title="Daily Total Clips Trend",
        labels={"x": "Date", "y": "Total Clips"}
    )
    st.plotly_chart(daily_fig, use_container_width=True)
    
    # Data Export
    st.header("💾 Data Export")
    
    if st.button("📥 Download Simulated Data as CSV"):
        csv_buffer = io.StringIO()
        df_filtered.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"campus_traffic_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("*This simulation is based on aggregate CCTV footage counts and peak hour analysis. No actual video processing was performed.*")

if __name__ == "__main__":
    main()