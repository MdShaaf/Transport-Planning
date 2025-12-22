import pandas as pd
import streamlit as st
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
from sklearn.preprocessing import MinMaxScaler
st.title("🚗 Data Insights" )
@st.cache_data
def load_data():
    con = duckdb.connect(database=':memory:', read_only=False)
    path=r"C:\Users\Shaaf\Desktop\Data Science\Practice Projects\Transport Planning\Sampled_Data\combined_sampled_data.parquet"
    df = con.execute(f"SELECT * FROM '{path}'").df()
    return df
data = load_data()
# Data Cleaning
data=data[data['total_amount']>0]
data=data[data['trip_distance']<=100]
# Load geojson
# with open(r"C:\Users\Shaaf\Desktop\Data Science\Practice Projects\Transport Planning\Sampled_Data\NYC Taxi Zones.geojson") as f:
#     taxi_zones = json.load(f)
#Famous Cab Companies
st.subheader("🚕 Famous Cab Companies")
company_counts = pd.DataFrame(data['VendorID'].value_counts()).rename_axis("VendorID").reset_index()
company_counts.sort_values(by="count",inplace=True,ascending=False)

fig = px.bar(
    company_counts,
    x="VendorID",
    y="count",
    text="count",
    title="Number of Rides by Cab Company", category_orders={"VendorID": company_counts["VendorID"].tolist()}
)

fig.update_traces(textposition="outside")

st.plotly_chart(fig, use_container_width=True)

st.subheader("🕒 Hourly Ride Distribution")
data['Hour'] = pd.to_datetime(data['tpep_pickup_datetime']).dt.hour
hourly_counts = pd.DataFrame(data['Hour'].value_counts()).rename_axis("Hour").reset_index().sort_values(by="Hour")
fig3 = px.bar(hourly_counts,
    x="Hour",
    y="count",
    text="count",
    title="Number of Rides per Hour", category_orders={"Hour": hourly_counts["Hour"].tolist()}
)
fig3.update_traces(textposition="outside")
st.plotly_chart(fig3, use_container_width=True)

#Time Series Analysis - Daily Ride Trends
data=data[data['tpep_pickup_datetime'].dt.year==2025]
data['Date'] = pd.to_datetime(data['tpep_pickup_datetime'])
df_resampled=data.set_index('Date')
df_resample=pd.DataFrame(df_resampled.resample('W').size(),columns=['passenger_count'])


st.subheader("📈 Weekly Ride Trends")
st.line_chart(df_resample)  

data['Month'] = pd.to_datetime(data['tpep_pickup_datetime']).dt.month
monthly_counts = data['Month'].value_counts().sort_index()
monthly_counts_df = pd.DataFrame({'Month': monthly_counts.index, 'Ride Count': monthly_counts.values})
st.subheader("📊 Monthly Ride Distribution")
fig5 = px.bar(
    monthly_counts_df,
    x="Month",
    y="Ride Count",
    text="Ride Count",
    color="Ride Count",
    width=800,
    title="Number of Rides per Month", category_orders={"Month": monthly_counts_df["Month"].tolist()}
)
fig5.update_layout(bargap=0.3)
fig5.update_traces(texttemplate='%{y}', textposition='outside')
st.plotly_chart(fig5, use_container_width=True)

#Plotting weekend vs weekday demand
data['is_weekend'] = data['tpep_pickup_datetime'].dt.dayofweek >= 5
hourly = data.groupby(['Hour', 'is_weekend']).size().reset_index(name='trips')
fig0=px.line(
    hourly,
    x='Hour',
    y='trips',
    color='is_weekend',
    markers=True,
    title="Hourly Travel Demand: Weekday vs Weekend",
    color_discrete_map={True: 'orange', False: 'blue'}
)
fig0.update_layout(legend_title_text='Is Weekend', legend=dict(
    itemsizing='constant',
    title_font_family="Arial",
    title_font_size=14,
    font=dict(family="Arial", size=12)
))
st.plotly_chart(fig0,use_container_width=True)

##Plotting Demand Handling vs Surcharge
st.subheader("Demand vs Surcharge")
data['Total Congestion Surcharge']=(data['congestion_surcharge']+data['cbd_congestion_fee'])
data['Total Congestion Surcharge']=data['Total Congestion Surcharge'].fillna(0)
demand_handling = data.groupby('Hour')['Total Congestion Surcharge'].mean().reset_index()
demand_handling['Trip Count']=data.groupby('Hour').size().values
scaler = MinMaxScaler()
demand_handling[['Total Congestion Surcharge', 'Trip Count']] = scaler.fit_transform(demand_handling[['Total Congestion Surcharge', 'Trip Count']])
df_long = demand_handling.melt(
    id_vars='Hour',
    value_vars=['Trip Count', 'Total Congestion Surcharge'],
    var_name='Metric',
    value_name='Value'
)
fig6 = px.line(
    df_long,
    x='Hour',
    y='Value',
    color='Metric',
    color_discrete_map={
        'Trip_Count_scaled': '#1f77b4',          # blue
        'Congestion Surcharge': '#d62728'  # red
    }
)

st.plotly_chart(fig6,use_container_width=False)

#plotting famous pickup points
st.subheader("📍 Most Popular Pickup Boroughs")
locations_data = pd.read_csv(r"C:\Users\Shaaf\Desktop\Data Science\Practice Projects\Transport Planning\Sampled_Data\taxi_zone_lookup.csv")
locations_name = locations_data[['LocationID', 'Zone']]
famous_trips = data.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='trip_count')
famous_trips = famous_trips.sort_values(by='trip_count', ascending=False)
famous_trips['Pick up Borough'] = famous_trips['PULocationID'].map(locations_data.set_index('LocationID')['Borough'])
famous_trips['Drop off Borough'] = famous_trips['DOLocationID'].map(locations_data.set_index('LocationID')['Borough'])

#most pickup zones
hotspots=famous_trips['Pick up Borough'].value_counts().head(5)
fig9=px.bar(hotspots, x=hotspots.index, y=hotspots.values, title='Most Popular Pickup Boroughs')
fig9.update_layout(xaxis_title='Borough', yaxis_title='Number of Pickups')
fig9.update_traces(marker_color=hotspots.values)
fig9.update_layout(width=700, height=500)
fig9.update_traces(texttemplate='%{y}', textposition='outside')
fig9.update_layout(title_text='Most Popular Pickup Boroughs', title_x=0.5)
#show the plot
st.plotly_chart(fig9,use_container_width=True)


# Ensure ID datatype matches GeoJSON

zone_stats = (
    data
    .groupby('PULocationID')
    .size()
    .reset_index(name='Trip_Count')
).sort_values(by='Trip_Count', ascending=False).head(30)
zone_stats['PULocationID'] = zone_stats['PULocationID'].astype(str)

with open(r"C:\Users\Shaaf\Desktop\Data Science\Practice Projects\Transport Planning\Sampled_Data\NYC Taxi Zones.geojson") as f:
    taxi_zones_geo = json.load(f)

fig10 = px.choropleth_mapbox(
    zone_stats,
    geojson=taxi_zones_geo,
    locations='PULocationID',
    featureidkey="properties.location_id",  # Ensure this matches the GeoJSON property
    color='Trip_Count',
    color_continuous_scale='Viridis',
    mapbox_style='carto-positron',
    zoom=10,
    center={"lat": 40.7128, "lon": -74.0060},
    opacity=0.75
)

fig10.update_layout(
    title="🚕 Pickup Demand by Taxi Zone",
    margin={"r":0, "t":40, "l":0, "b":0}
)
fig10.update_coloraxes(colorbar_title="Number of Pickups")
st.plotly_chart(fig10,use_container_width=True)

##Distance and Bins plot
bins= [0, 5, 10, 20, 30, 50, 100]
labels = ['0-5km', '5-10km', '10-20km', '20-30km', '30-50km', '50-100km']
data['Distance_Bin'] = pd.cut(data['trip_distance'], bins=bins, labels=labels, right=False)
st.subheader("🚙 Trip Distance Distribution")
summary = data.groupby('Distance_Bin').agg(
    trips=('passenger_count', 'size'),
    total_passengers=('passenger_count', 'sum'),
    avg_passengers=('passenger_count', 'mean')
)
fig11 = px.bar(
    summary.reset_index(),
    x='Distance_Bin',
    y='trips',
    color='trips',
    text='trips',
    title="Trips by Distance Range"
)
fig11.update_traces(marker_color='indianred', texttemplate='%{y}', textposition='outside')
st.plotly_chart(fig11,use_container_width=True)

fig12 = px.line(
    summary.reset_index(),
    x='Distance_Bin',
    y='avg_passengers',
    markers=True,

    title="Average Passengers per Trip vs Distance"
)
fig12.update_traces(line_color='green', marker=dict(size=10))
st.plotly_chart(fig12,use_container_width=True)

st.markdown("---")