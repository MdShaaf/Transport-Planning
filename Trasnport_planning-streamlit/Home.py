import streamlit as st
# Configure the page
st.set_page_config(
    page_title="Transport Analytics Dashboard",
    page_icon="🚗",
    layout="wide"
)

# Main page content
st.title("🚗 Transport Analytics Dashboard")
st.markdown("---")

st.write("""
Welcome to the Transport Analytics Application! 

Use the sidebar to navigate between different analysis pages:
- **Data Insights**: Explore transport data visualizations
- **Demand Forecasting**: Predict future transport demand
- **Route Recommendation**: Recommend Zones for best profit
""")



st.sidebar.success("Select a page from the sidebar to get started.")

st.caption("Developed by Shafeeq")
