# Product Requirements Document (PRD): Campus Traffic Simulation and Analytics App

## 1. Introduction

### 1.1 Purpose
This PRD outlines the requirements for a Python-based Streamlit application that simulates campus traffic data based on provided aggregate footage counts from CCTV cameras and peak hour analysis. The app will generate simulated time-series data to mimic processed video analytics without actually processing any videos. It will focus on analytics through interactive visualizations and metrics, emphasizing graphs for data exploration rather than static reports. The simulation will use the peak hour insights to create realistic traffic patterns, making it appear as if the data has been derived from video processing.

### 1.2 Background
- **Data Source**: Aggregate footage (video clip counts) from 8 CCTV cameras over the period September 2, 2025 (Tuesday) to September 5, 2025 (Friday). These are all weekdays, with no weekends or assumed holidays in the range.
- **Peak Hour Analysis**:
  - Morning (7:30–8:30 AM): Heavy load at Arch Gate IN and Chola Statue (simulating students rushing for classes).
  - Afternoon (12:30–1:30 PM): Traffic spikes at Clock Tower and Arch Gate OUT (simulating lunch breaks and mid-day shifts).
  - Evening (4:30–5:30 PM): Heavy exit congestion at Chola Statue OUT and Arch Gate OUT (simulating class dispersal).
- **Key Constraint**: No real video processing or real-time data integration. All data is simulated using the aggregate totals and peak hour patterns to infer activity levels (e.g., higher clip counts during peaks to represent congestion).
- **User Goal**: Provide analytics (e.g., trends, comparisons, peaks) via graphs, not narrative reports.

### 1.3 Target Users
- Campus administrators or analysts interested in traffic patterns.
- Developers or stakeholders testing simulated analytics prototypes.

### 1.4 Assumptions
- Footage counts represent activity levels (e.g., motion-triggered clips indicating foot traffic or congestion).
- Simulation will match the aggregate totals per camera over the 4-day period.
- No weekends/holidays in the date range, so all days simulate typical weekday traffic. However, the app will include an optional toggle to simulate "holiday mode" on selected days (reducing traffic to minimal levels, e.g., 10-20% of normal) to demonstrate variability.
- Granularity: 30-minute intervals for time-series simulation.

## 2. Objectives
- Simulate multi-day traffic data across locations, with peaks aligned to the provided analysis.
- Enable interactive exploration of analytics, including totals, averages, peaks, and trends.
- Use graphs for visualization to highlight patterns (e.g., time-based spikes, location comparisons).
- Ensure the app is user-friendly, performant, and deployable via Streamlit.

## 3. Scope

### 3.1 In Scope
- Data simulation logic based on peak hours and aggregates.
- Interactive dashboard with filters and multiple graph types.
- Analytics metrics derived from simulated data (e.g., daily averages, peak detection).
- Export options for simulated data (e.g., CSV download).

### 3.2 Out of Scope
- Integration with actual video files or real-time CCTV feeds.
- Advanced ML-based predictions (unless simple rule-based, e.g., extrapolating peaks).
- Campus maps or location descriptions.
- User authentication or multi-user features.
- Mobile optimization (focus on desktop web).

## 4. Functional Requirements

### 4.1 Data Simulation
- **Input Data**:
  - Locations and Aggregates:

| Location | Aggregate Clips (Sept 2-5, 2025) |
|----------|----------------------------------|
| Arch Gate IN-D30 | 322 |
| Arch Gate OUT-D32 | 279 |
| Chola Statue OUT-D-26 | 629 |
| Arch Gate Outside IN D36 | 31 |
| Arch Gate IN-OUT D35 | 472 |
| Chola Statue IN D22 | 322 |
| Vallamai Main Gate D13 | 339 |
| Vallamai Main Gate D16 | 55 |

  - Peak Hours: Used to weight simulation (e.g., 3-5x higher counts during peaks vs. off-peak).
- **Simulation Logic**:
  - Generate a Pandas DataFrame with columns: Date, Location, Time Interval (e.g., "07:00-07:30"), Simulated Clip Count.
  - For each day (Sept 2-5, 2025) and location:
    - Divide 24 hours into 48 x 30-minute intervals.
    - Assign base counts: Evenly distribute aggregate clips across days (e.g., average daily = aggregate / 4), then adjust per interval.
    - Apply multipliers:
      - Peak periods: High counts (e.g., morning peaks at Arch Gate IN and Chola Statue; afternoon at Clock Tower and Arch Gate OUT; evening at Chola Statue OUT and Arch Gate OUT).
      - Off-peak: Low counts (e.g., 20-50% of peak).
      - Night hours (e.g., 8 PM-7 AM): Minimal (near zero).
    - Introduce variability: Random noise (±10-20%) per interval to simulate realism.
    - Holiday/Weekend Toggle: User-selectable per day; reduces overall counts to 10-20% if enabled.
  - Ensure totals match aggregates when summed over the period per location.
- **Output**: In-memory DataFrame for analytics and graphing.

### 4.2 Core Features
- **Dashboard Overview**:
  - Display key metrics: Total clips across all locations/days, average daily clips per location, detected peaks (auto-identified based on thresholds, e.g., >2x average).
  - Use Streamlit metrics widgets for quick insights.
- **Interactive Filters** (Sidebar):
  - Multi-select locations (default: all).
  - Date range selector (default: Sept 2-5, 2025).
  - Time range slider (e.g., filter to morning/afternoon).
  - Holiday simulation toggle (per day checkboxes).
  - Refresh button to re-simulate data with changes.
- **Analytics Calculations**:
  - Totals: Sum of clips by location/day/period.
  - Averages: Mean clips per interval/day/location.
  - Peaks: Identify and list top intervals with highest counts, aligned to peak hour analysis.
  - Comparisons: Variance between days/locations, percentage distribution.
  - Trends: Daily totals, growth/decline over the period.
  - Congestion Index: Simple metric (e.g., clips above threshold as % of total).
- **Visualizations** (Main Area):
  - **Time-Series Line Chart**: Clips over time (30-min intervals) for selected day(s)/location(s). Use Plotly for interactivity (zoom, hover).
  - **Bar Chart**: Total clips per location or per day.
  - **Heatmap**: Activity by time interval (rows) and location/day (columns), color-coded by count.
  - **Stacked Bar Chart**: Breakdown of clips by peak periods (morning/afternoon/evening/off-peak).
  - **Pie Chart**: Distribution of total clips across locations.
  - All graphs interactive, with tooltips showing exact values and peak hour context (e.g., "Morning Rush").
- **Data Export**:
  - Button to download simulated DataFrame as CSV.

### 4.3 Non-Functional Requirements
- **Performance**: App loads in <5 seconds; simulations run quickly (e.g., pre-compute on load if possible).
- **Usability**: Clean Streamlit layout; responsive graphs; tooltips for explanations.
- **Error Handling**: Graceful handling if no data selected (e.g., "Select a location").
- **Accessibility**: Basic (e.g., alt text for graphs).

## 5. Technical Requirements
- **Framework**: Streamlit (for UI and deployment).
- **Languages/Libraries**:
  - Python 3.x.
  - Data Handling: Pandas (for simulation and analysis).
  - Visualization: Plotly (interactive graphs) or Matplotlib/Seaborn (fallback).
  - Others: NumPy (for random variability), Datetime (for intervals).
- **Deployment**: Local run via `streamlit run app.py`; optional sharing via Streamlit Cloud.
- **Code Structure**:
  - `simulate_data()`: Function to generate DataFrame based on inputs.
  - Main script: Load data, apply filters, compute analytics, render graphs.
- **Testing**: Unit tests for simulation logic (e.g., totals match aggregates); manual UI testing.

## 6. Risks and Mitigations
- **Risk**: Simulated data feels unrealistic. **Mitigation**: Tune multipliers based on peak analysis; add user-adjustable parameters.
- **Risk**: Overly complex UI. **Mitigation**: Start with core graphs; iterate based on feedback.
- **Risk**: Date range is short/no weekends. **Mitigation**: Include holiday toggle to demonstrate minimal traffic scenarios.

## 7. Timeline and Milestones (High-Level)
- Week 1: Data simulation logic and basic dashboard.
- Week 2: Add filters, graphs, and analytics.
- Week 3: Testing, refinements, and documentation.