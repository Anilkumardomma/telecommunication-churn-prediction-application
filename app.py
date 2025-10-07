# Importing Libraries
# This block imports the required libraries for the application. These include Streamlit for the web framework,
# Pandas for data manipulation, Joblib for model loading, and Plotly and SHAP for visualization and model explainability.
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import shap

# Page Configuration
# This code configures the web page's metadata and default layout. It sets the browser tab title,
# centers the main content for better readability, and ensures the sidebar is collapsed on initial load.
st.set_page_config(
    page_title="Churn Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Loading and Caching ML Assets
# This function loads the pre-trained machine learning assets required for prediction. The '@st.cache_resource'
# decorator is used to optimize performance by loading these assets into memory only once when the app starts.
@st.cache_resource
def load_assets():
    """Loads all necessary machine learning assets."""
    imputer = joblib.load("imputer.joblib")
    scaler = joblib.load("scaler.joblib")
    model = joblib.load("xgb_churn_model.joblib")
    explainer = shap.TreeExplainer(model)
    return imputer, scaler, model, explainer

imputer, scaler, model, explainer = load_assets()

# App Header
# This section renders the main title and a descriptive subtitle for the application. It provides the user
# with an immediate understanding of the app's purpose and scope.
st.title("Customer Churn Predictor")
st.write("Predict churn with high accuracy by providing the complete customer profile.")

# The Input Form
# This block creates a user input form with 'st.form' to efficiently gather customer data. Using a form ensures
# the prediction logic only runs upon submission, and widgets are organized into columns for a clean UI.
with st.form("churn_prediction_form"):
    st.header("Enter Full Customer Details")

    # Plans & Account
    st.subheader("Account and Plans")
    col1, col2 = st.columns(2)
    with col1:
        international_plan = st.radio("Has International Plan?", ("No", "Yes"), horizontal=True)
    with col2:
        voice_mail_plan = st.radio("Has Voice Mail Plan?", ("No", "Yes"), horizontal=True)
    
    account_length = st.slider("Account Length (days)", 1, 240, 100)
    customer_service_calls = st.slider("Customer Service Calls", 0, 10, 1)

    # Usage Details (Minutes)
    st.subheader("Usage Details (in Minutes)")
    col3, col4, col5 = st.columns(3)
    with col3:
        day_mins = st.number_input("Day Mins", min_value=0.0, value=180.0)
    with col4:
        evening_mins = st.number_input("Evening Mins", min_value=0.0, value=200.0)
    with col5:
        night_mins = st.number_input("Night Mins", min_value=0.0, value=200.0)
    
    international_mins = st.number_input("International Mins", min_value=0.0, value=10.0)
    
    # Call Details (Number of Calls)
    st.subheader("Number of Calls")
    col6, col7, col8 = st.columns(3)
    with col6:
        day_calls = st.number_input("Day Calls", min_value=0, value=100)
    with col7:
        evening_calls = st.number_input("Evening Calls", min_value=0, value=100)
    with col8:
        night_calls = st.number_input("Night Calls", min_value=0, value=100)

    international_calls = st.number_input("International Calls", min_value=0, value=4)

    # Submit Button
    st.markdown("---") # Visual separator
    predict_button = st.form_submit_button(label="Predict Churn", use_container_width=True)


# Prediction Logic
# Upon form submission, this logic preprocesses the user input data through the imputer and scaler.
# It then generates a churn probability using the trained model and calculates SHAP values to explain the prediction.
if predict_button:
    # Convert radio button inputs to numerical values
    int_plan_val = 1 if international_plan == "Yes" else 0
    vm_plan_val = 1 if voice_mail_plan == "Yes" else 0

    # Create the DataFrame with ALL user inputs 
    feature_names = [
        'account_length', 'voice_mail_plan', 'day_mins', 'evening_mins',
        'night_mins', 'international_mins', 'customer_service_calls',
        'international_plan', 'day_calls', 'evening_calls',
        'night_calls', 'international_calls'
    ]
    input_data = pd.DataFrame([[
        account_length, vm_plan_val, day_mins, evening_mins,
        night_mins, international_mins, customer_service_calls, int_plan_val,
        day_calls, evening_calls, night_calls, international_calls
    ]], columns=feature_names)

    # Preprocess and predict
    input_imputed = imputer.transform(input_data)
    input_scaled = scaler.transform(input_imputed)
    probability = model.predict_proba(input_scaled)[0][1]
    shap_values = explainer.shap_values(input_scaled)

    # Displaying the Prediction Result
    # The prediction results are presented in a two-column layout for clarity. It displays a color-coded
    # status (High/Low Risk) for quick interpretation and a precise 'st.metric' for the numerical churn probability.
    st.markdown("---")
    st.header("Prediction Result")
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        if probability > 0.5:
            st.error("Status: HIGH RISK")
        else:
            st.success("Status: LOW RISK")
    with res_col2:
        st.metric(label="Churn Probability", value=f"{probability:.1%}")

    # Creating the Explanation Plot
    # This section enhances model transparency by visualizing the prediction's rationale with a SHAP bar chart. It shows
    # the top five features influencing the outcome, color-coded to indicate if they increased or decreased churn risk.
    st.subheader("Top 5 Reasons for this Prediction")
    
    # Creates a DataFrame for SHAP values and feature names.
    shap_df = pd.DataFrame({
        'feature': [f.replace('_', ' ').title() for f in input_data.columns],
        'shap_value': shap_values[0, :],
    })
    
    # Assign colors like red for features increasing churn risk, blue for those decreasing it.
    shap_df['color'] = np.where(shap_df['shap_value'] > 0, '#e53e3e', '#3182ce')
    
    # Sort by the absolute impact to find the top 5 features.
    shap_df['abs_shap'] = np.abs(shap_df['shap_value'])
    shap_df = shap_df.sort_values('abs_shap', ascending=False).head(5)

    # Creates and display the Plotly bar chart.
    fig = go.Figure(go.Bar(
        x=shap_df['shap_value'],
        y=shap_df['feature'],
        orientation='h',
        marker_color=shap_df['color']
    ))
    fig.update_layout(
        xaxis_title="Impact on Churn Risk (Red = Higher Risk)",
        yaxis=dict(autorange="reversed"),
        height=300,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})
