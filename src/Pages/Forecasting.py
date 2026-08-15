import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import holidays

project_dir = Path()
model_dir  = project_dir/"models"
artifacts_dir  = project_dir/"artifacts"
print(f"{project_dir}")
col1, col2, col3 = st.columns([1,1,1.3])

with col1:
    st.subheader("Borough")
    boroug_choice =st.selectbox(
        "Borough",['Manhattan','Queens','Brooklyn','Bronx'])

with col2:
    st.subheader("Model")
    # st.write("*CatBoost Recommended")
    model_choice = st.selectbox(
        "Model", 
        ["CatBoost", "XGBoost", "LightGBM", "ARIMA", "SARIMA"])
    if model_choice=="CatBoost":
        model = joblib.load(model_dir/f"{boroug_choice}_CatBoostRegressor.pkl")
        
with col3:
    st.subheader("Forecast Horizon")
    forecast_choice = st.selectbox(
        "Forecast Step", 
        ["Next Hour", "Next 2 Hours", "Next 6 Hours", "Next Day", "Next Week", "Next 2 Weeks", "Next Month"])
    

st.info(
    f"""
    Borough: {boroug_choice}  
    Model: {model_choice}  
    Forecast: {forecast_choice}
    """
)


forecast_options = {
    "Next Hour": 1,
    "Next 6 Hours": 6,
    "Next 12 Hours": 12,
    "Next Day": 24,
    "Next Week": 24 * 7,
    "Next 2 Weeks": 24 * 14,
    "Next Month": 24 * 30
}
expected_model_features = model.feature_names_
with st.spinner('Forecasting...'):
    steps = forecast_options[forecast_choice]
    def forecast(steps, borough, model):
        import holidays
        import pandas as pd
        import numpy as np
        
        us_holidays = holidays.UnitedStates()
        df = pd.read_csv(project_dir/"Data/combined_sampled_data.csv", parse_dates=['tpep_pickup_datetime'])
        df = df[df['Borough'] == borough]
        df = df[['tpep_pickup_datetime', 'Trips']]
        df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)

        features = ['Is_Holiday','Day of the week','Month','Is weekend','Hour of the Day','Lag1',
                    'Lag2','Lag3','Lag6','Lag12','Lag24','Lag48','Lag72','Lag168','Lag336']
        results = []  
        for step in range(steps):
            df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
            last_time = pd.Timestamp(df["tpep_pickup_datetime"].iloc[-1])

            next_time = last_time + pd.DateOffset(hours=1)

            step_df = pd.DataFrame({'tpep_pickup_datetime': [next_time], 'Trips': [np.nan]})  

            step_df["Is_Holiday"] = int(next_time.date() in us_holidays)
            step_df['Is weekend'] = int(next_time.dayofweek >= 5)
            step_df['Month'] = next_time.month
            step_df['Day of the week'] = next_time.dayofweek
            step_df['Hour of the Day'] = next_time.hour

            lags = [1, 2, 3, 6, 12, 48, 24, 72, 168, 336]
            for lag in lags:
                step_df[f'Lag{lag}'] = df['Trips'].iloc[-lag]

            X = step_df[features]
            prediction = model.predict(X)[0]
            step_df['Trips'] = prediction

            df = pd.concat([df, step_df[['tpep_pickup_datetime', 'Trips']]], ignore_index=True)
            results.append(step_df)

        return pd.concat(results, ignore_index=True)
    forcast = forecast(steps=steps,borough=boroug_choice,model=model)  
    with st.expander("View Forecast Data", expanded=False):
        st.write(forcast[['tpep_pickup_datetime', 'Trips']])
    # st.write(forcast.head())

    ##Plotting 
    import plotly.express as px
    fig = px.line(forcast, x='tpep_pickup_datetime', y='Trips', color_discrete_sequence=['blue'], title=f'Forecasted Trips for {boroug_choice} using {model_choice} for {forecast_choice}')
    #set x-axis title
    fig.update_layout(xaxis_title='Date and Time', yaxis_title='Number of Trips')
    st.plotly_chart(fig)

    ##Demand Pattern Heatmap
    df = pd.read_csv(project_dir/"Data/combined_sampled_data.csv", parse_dates=['tpep_pickup_datetime'])
    df = df[df['Borough'] == boroug_choice]
    df = df[['tpep_pickup_datetime', 'Trips']]
    df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)

    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day'] = df['tpep_pickup_datetime'].dt.day_name()
    heatmap_data = (
    df.groupby(
        ["day","hour"]
    )["Trips"]
    .mean()
    .reset_index())

    heatmap_table = heatmap_data.pivot(
        index="day",
        columns="hour",
        values="Trips")

    import plotly.express as px


    fig = px.imshow(
        heatmap_table,
        aspect="auto",
        color_continuous_scale="Viridis",
        labels=dict(
            x="Hour",
            y="Day",
            color="Trips"
        ),
        title="Taxi Demand Pattern"
    )

    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(24)),
            ticktext=[f"{i}:00" for i in range(24)]
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(7)),
            ticktext=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
    )
    st.plotly_chart(fig)

    ##let's also mention Error metric
error_metrics = pd.read_csv(
    artifacts_dir / f"Error_metrics_for_{boroug_choice}.csv"
)

st.subheader("Model Performance Metrics")

# Create 4 columns for clean KPI cards (excluding MSE for cleaner layout)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="MAE",
        value=f"{error_metrics['best_mae'].iloc[0]:,.2f}",
        help="Mean Absolute Error",
    )

with col2:
    st.metric(
        label="RMSE",
        value=f"{error_metrics['best_rmse'].iloc[0]:,.2f}",
        help="Root Mean Squared Error",
    )

with col3:
    st.metric(
        label="R² Score",
        value=f"{error_metrics['best_r2_score'].iloc[0]:.2%}",  # Displays 97.00%
        help="Coefficient of Determination",
    )

with col4:
    st.metric(
        label="Error Rate",
        value=f"{error_metrics['best_percentage_error'].iloc[0]:.2f}%",  # Displays 2.64%
        help="Mean Absolute Percentage Error",
    )