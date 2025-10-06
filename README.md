# Campus Traffic Simulation & Analytics App

A Python-based Streamlit application that simulates campus traffic data based on CCTV footage analysis and provides interactive analytics visualizations.

## Features

- **Data Simulation**: Generates realistic traffic patterns based on peak hour analysis
- **Interactive Dashboard**: Overview metrics with filters for locations, dates, and time ranges
- **Multiple Visualizations**: Time-series charts, bar charts, heatmaps, and pie charts
- **Analytics**: Peak detection, congestion index, and trend analysis
- **Holiday Simulation**: Toggle to simulate reduced traffic on selected days
- **Data Export**: Download simulated data as CSV

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

3. Use the sidebar filters to:
   - Select specific locations to analyze
   - Choose date ranges (Sept 2-5, 2025)
   - Set time ranges for analysis
   - Enable holiday simulation for specific days

4. Explore the interactive visualizations and analytics

## Data Overview

The simulation is based on aggregate CCTV footage counts from 8 campus locations over 4 weekdays (September 2-5, 2025):

- **Arch Gate IN-D30**: 322 clips
- **Arch Gate OUT-D32**: 279 clips  
- **Chola Statue OUT-D-26**: 629 clips
- **Arch Gate Outside IN D36**: 31 clips
- **Arch Gate IN-OUT D35**: 472 clips
- **Chola Statue IN D22**: 322 clips
- **Vallamai Main Gate D13**: 339 clips
- **Vallamai Main Gate D16**: 55 clips

## Peak Hours

The simulation incorporates realistic traffic patterns with these peak periods:
- **Morning Rush** (7:30-8:30 AM): Heavy at entrance gates
- **Lunch Break** (12:30-1:30 PM): Activity at central locations
- **Evening Dispersal** (4:30-5:30 PM): Heavy at exit points

## Technical Details

- **Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly
- **Time Granularity**: 30-minute intervals
- **Simulation Logic**: Peak multipliers with random variability

*Note: This application generates simulated data for demonstration purposes. No actual video processing is performed.*