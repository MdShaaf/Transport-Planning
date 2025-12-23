import pandas as pd
import streamlit as st
import duckdb
import numpy as np
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="NYC Taxi Route Optimizer", layout="wide", page_icon="🚕")

# === STEP 1: LOAD REAL DATA ===
@st.cache_data
def load_and_prepare_data():
    """Load real NYC taxi data"""
    con = duckdb.connect(database=':memory:', read_only=False)
    path = r"C:\Users\Shaaf\Desktop\Data Science\Practice Projects\Transport Planning\Sampled_Data\combined_sampled_data.parquet"
    df = con.execute(f"SELECT * FROM '{path}'").df()
    
    # Clean data
    df = df[df['total_amount'] > 0]
    df = df[df['trip_distance'] <= 100]
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['Date'] = pd.to_datetime(df['tpep_pickup_datetime'].dt.date)
    df = df[df['Date'].dt.year == 2025]
    
    return df

# === STEP 2: CALCULATE PROFIT FROM REAL DATA ===
def calculate_profit_by_zone_hour(df):
    """
    Simple approach: For each zone and hour, calculate average profit
    Profit = Revenue (fare + tip) - Costs (fuel based on distance)
    """
    FUEL_COST_PER_MILE = 0.60
    
    # Calculate profit for each trip
    df['revenue'] = df['fare_amount'] + df.get('tip_amount', 0).fillna(0)
    df['cost'] = df['trip_distance'] * FUEL_COST_PER_MILE
    df['profit'] = df['revenue'] - df['cost']
    
    # Group by pickup zone and hour to find average profit
    profit_summary = df.groupby(['PULocationID', 'pickup_hour']).agg({
        'profit': 'mean',
        'PULocationID': 'count'  # Count trips
    }).rename(columns={'PULocationID': 'trip_count'})
    
    return profit_summary

# === STEP 3: CREATE SIMPLE RECOMMENDATION TABLE ===
def create_recommendation_table(profit_summary):
    """
    Convert profit data into a simple lookup table
    For each zone and hour, we know the average profit
    """
    # Create a table: rows=zones, columns=hours
    zones = range(266)
    hours = range(24)
    
    recommendation_table = np.zeros((266, 24))
    
    for zone in zones:
        for hour in hours:
            try:
                # Get average profit for this zone-hour combination
                profit = profit_summary.loc[(zone, hour), 'profit']
                recommendation_table[zone, hour] = profit
            except:
                # If no data exists, set to 0
                recommendation_table[zone, hour] = 0
    
    return recommendation_table

# === STEP 4: FIND BEST NEARBY ZONES ===
def get_nearby_zones(current_zone):
    """
    Get nearby zones (simple adjacency based on zone numbers)
    In reality, you'd use geographic coordinates
    """
    nearby = []
    for offset in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
        neighbor = current_zone + offset
        if 0 <= neighbor < 266:
            nearby.append(neighbor)
    return nearby

# Load data
with st.spinner("Loading real NYC taxi data..."):
    data = load_and_prepare_data()
    profit_summary = calculate_profit_by_zone_hour(data)
    recommendation_table = create_recommendation_table(profit_summary)

# Store in session state
if 'rec_table' not in st.session_state:
    st.session_state.rec_table = recommendation_table

# === INTERFACE ===
st.title("🚕 NYC Taxi Profit Analyzer")
st.markdown("""
**Simple Concept**: This tool analyzes real taxi trip data to show you which zones are most profitable at different times.
It calculates: **Profit = (Fare + Tips) - (Fuel Costs)**
""")

# Show data stats
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Total Trips Analyzed", f"{len(data):,}")
with col_info2:
    avg_profit = data['profit'].mean()
    st.metric("Average Profit per Trip", f"${avg_profit:.2f}")
with col_info3:
    st.metric("Data Year", "2025")

st.markdown("---")

# Zone names
ZONE_NAMES = {
    161: "Midtown Center", 162: "Midtown East", 163: "Midtown North",
    164: "Midtown South", 230: "Times Square", 237: "Upper East Side",
    238: "Upper West Side", 142: "Lincoln Square", 170: "Murray Hill",
    186: "Penn Station", 79: "East Village", 100: "Garment District",
    113: "Greenwich Village", 132: "JFK Airport", 138: "LaGuardia Airport",
}

# === USER INPUT ===
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📍 Select Location & Time")
    
    # Zone selection
    zone_options = {f"{k} - {v}": k for k, v in ZONE_NAMES.items()}
    selected_zone_name = st.selectbox(
        "Where are you now?",
        options=list(zone_options.keys()),
        index=0
    )
    selected_zone = zone_options[selected_zone_name]
    
    # Time selection
    selected_hour = st.slider("What time is it?", 0, 23, 14, 1)
    time_label = f"{selected_hour:02d}:00"
    st.info(f"🕐 **{time_label}**")

with col2:
    st.subheader("💰 Expected Profit")
    
    # Get profit for this zone and hour
    expected_profit = st.session_state.rec_table[selected_zone, selected_hour]
    
    if expected_profit > 0:
        st.success(f"### ${expected_profit:.2f}")
        st.caption("Average profit per trip based on historical data")
        
        # Show historical trip count
        try:
            trip_count = profit_summary.loc[(selected_zone, selected_hour), 'trip_count']
            st.info(f"Based on **{int(trip_count)}** historical trips")
        except:
            st.info("Limited historical data for this zone/hour")
    else:
        st.warning("⚠️ No historical data for this zone/hour")

# === RECOMMENDATION: WHERE TO GO NEXT ===
st.markdown("---")
st.subheader("🎯 Recommendation: Where Should You Go Next?")

# Find nearby zones and their profits
nearby_zones = get_nearby_zones(selected_zone)
next_hour = (selected_hour + 1) % 24  # Next hour (wraps to 0 after 23)

nearby_recommendations = []
for zone in nearby_zones:
    profit = st.session_state.rec_table[zone, next_hour]
    zone_name = ZONE_NAMES.get(zone, f"Zone {zone}")
    nearby_recommendations.append({
        'Zone ID': zone,
        'Zone Name': zone_name,
        'Expected Profit ($)': profit
    })

# Sort by profit (highest first)
nearby_df = pd.DataFrame(nearby_recommendations).sort_values('Expected Profit ($)', ascending=False)

# Show top 3 recommendations
col_r1, col_r2, col_r3 = st.columns(3)

if len(nearby_df) > 0 and nearby_df.iloc[0]['Expected Profit ($)'] > 0:
    # Best recommendation
    with col_r1:
        best = nearby_df.iloc[0]
        st.success("**🥇 Best Choice**")
        st.metric(
            best['Zone Name'],
            f"${best['Expected Profit ($)']:.2f}",
            delta="Highest Profit"
        )
        st.caption(f"Zone {best['Zone ID']} at {next_hour:02d}:00")
    
    # Second best
    with col_r2:
        if len(nearby_df) > 1:
            second = nearby_df.iloc[1]
            st.info("**🥈 Alternative**")
            st.metric(
                second['Zone Name'],
                f"${second['Expected Profit ($)']:.2f}"
            )
            st.caption(f"Zone {second['Zone ID']} at {next_hour:02d}:00")
    
    # Third best
    with col_r3:
        if len(nearby_df) > 2:
            third = nearby_df.iloc[2]
            st.info("**🥉 Backup Option**")
            st.metric(
                third['Zone Name'],
                f"${third['Expected Profit ($)']:.2f}"
            )
            st.caption(f"Zone {third['Zone ID']} at {next_hour:02d}:00")
    
    # Decision helper
    current_zone_next_hour = st.session_state.rec_table[selected_zone, next_hour]
    best_nearby_profit = nearby_df.iloc[0]['Expected Profit ($)']
    
    if best_nearby_profit > current_zone_next_hour * 1.2:  # 20% better
        st.success(f"💡 **Recommendation: Move to {nearby_df.iloc[0]['Zone Name']}** - It's {((best_nearby_profit/current_zone_next_hour - 1)*100):.0f}% more profitable!")
    else:
        st.info(f"💡 **Recommendation: Stay in current zone** - Nearby zones aren't significantly better")
else:
    st.warning("⚠️ Limited data for nearby zones at the next hour")

# === COMPARISON CHARTS ===
st.markdown("---")
st.subheader(f"📊 Profit Analysis for Zone {selected_zone}")

# Chart 1: Profit throughout the day
hourly_profits = []
for hour in range(24):
    profit = st.session_state.rec_table[selected_zone, hour]
    hourly_profits.append({
        'Hour': f"{hour:02d}:00",
        'Hour_num': hour,
        'Expected Profit ($)': profit
    })

hourly_df = pd.DataFrame(hourly_profits)

fig_hourly = px.line(
    hourly_df,
    x='Hour',
    y='Expected Profit ($)',
    title=f'Expected Profit Throughout the Day - {ZONE_NAMES.get(selected_zone, f"Zone {selected_zone}")}',
    markers=True
)
fig_hourly.add_vline(x=selected_hour, line_dash="dash", line_color="red", 
                     annotation_text="Current Time")
fig_hourly.update_layout(height=400)
st.plotly_chart(fig_hourly, use_container_width=True)

# Chart 2: Compare different zones at current time
st.subheader(f"🗺️ Compare Zones at {time_label}")

comparison_data = []
for zone_id, zone_name in ZONE_NAMES.items():
    profit = st.session_state.rec_table[zone_id, selected_hour]
    comparison_data.append({
        'Zone': zone_name,
        'Expected Profit ($)': profit
    })

comp_df = pd.DataFrame(comparison_data).sort_values('Expected Profit ($)', ascending=False)

fig_zones = px.bar(
    comp_df,
    x='Zone',
    y='Expected Profit ($)',
    title=f'Most Profitable Zones at {time_label}',
    color='Expected Profit ($)',
    color_continuous_scale='Viridis'
)
fig_zones.update_layout(height=400, xaxis_tickangle=-45)
st.plotly_chart(fig_zones, use_container_width=True)

# === INSIGHTS SECTION ===
st.markdown("---")
st.subheader("💡 Key Insights")

# Find best zone and hour overall
best_zone_idx, best_hour_idx = np.unravel_index(
    st.session_state.rec_table.argmax(), 
    st.session_state.rec_table.shape
)
best_profit = st.session_state.rec_table[best_zone_idx, best_hour_idx]

col_i1, col_i2, col_i3 = st.columns(3)

with col_i1:
    st.metric(
        "Most Profitable Zone", 
        ZONE_NAMES.get(best_zone_idx, f"Zone {best_zone_idx}"),
        delta=f"${best_profit:.2f}"
    )

with col_i2:
    # Best hour for current zone
    best_hour_current = np.argmax(st.session_state.rec_table[selected_zone, :])
    best_profit_current = st.session_state.rec_table[selected_zone, best_hour_current]
    st.metric(
        f"Best Time for {ZONE_NAMES.get(selected_zone, 'Current Zone')}", 
        f"{best_hour_current:02d}:00",
        delta=f"${best_profit_current:.2f}"
    )

with col_i3:
    # Current vs best comparison
    current_profit = st.session_state.rec_table[selected_zone, selected_hour]
    if best_profit_current > 0:
        efficiency = (current_profit / best_profit_current) * 100
        st.metric(
            "Current Time Efficiency",
            f"{efficiency:.0f}%",
            delta=f"${current_profit - best_profit_current:.2f}"
        )

# === EXPLANATION ===
with st.expander("ℹ️ How This Works (For Interview)"):
    st.markdown("""
    ### Simple Explanation:
    
    1. **Data Collection**: Load real NYC taxi trip data (fares, distances, tips)
    
    2. **Profit Calculation**: 
       - Revenue = Fare + Tips
       - Cost = Distance × $0.60 (fuel)
       - **Profit = Revenue - Cost**
    
    3. **Pattern Analysis**: Group trips by zone and hour, calculate average profit
    
    4. **Recommendation Logic**: 
       - Look at nearby zones (±5 zone IDs)
       - Check their profit in the next hour
       - Recommend the most profitable option
       - Compare: Is moving worth it vs staying?
    
    ### Why It's Useful:
    - Drivers can see which areas make the most money
    - They get actionable recommendations: "Go to Times Square" or "Stay here"
    - Better than random driving or guesswork
    - Real-time decision support based on historical patterns
    
    ### No Complex AI Needed:
    - This is just **statistical analysis** of historical data
    - Average profit per zone/hour is calculated directly from past trips
    - Simple comparison logic: find nearby zones with higher profit
    - Simple, interpretable, and effective!
    
    ### Interview Talking Points:
    - "Built a profit optimizer using real 2025 NYC taxi data"
    - "Calculates profit = revenue - fuel costs for each zone/hour"
    - "Recommends next destination by comparing nearby zones"
    - "Helps drivers make data-driven routing decisions"
    - "Uses simple statistics, not black-box ML"
    """)

# Footer
st.markdown("---")
st.caption("📊 Based on Real NYC Taxi Data (2025) | Profit = (Fare + Tips) - Fuel Costs")