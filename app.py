import streamlit as st
import pandas as pd
from streamlit_lottie import st_lottie
import json
import sys
import os
import requests
from sklearn.model_selection import train_test_split

# Add backend folder to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from backend.waste_reduction_backend import (
    load_data, train_models,
    generate_recommendations, cost_benefit_analysis, generate_pdf_report
)

# ----------- Lottie Animation Loaders ----------- #

def load_lottie_file(filepath: str):
    try:
        with open(filepath, "r") as file:
            return json.load(file)
    except Exception as e:
        st.warning(f"⚠️ Could not load local Lottie file: {e}")
        return None

def load_lottie_url(url: str):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            st.warning(f"⚠️ Lottie URL failed with status: {response.status_code}")
            return None
    except Exception as e:
        st.warning(f"⚠️ Lottie URL exception: {e}")
        return None

# ----------- Streamlit UI ----------- #

st.markdown("<h1 style='font-size:40px;'>🌱 Waste Reduction in Manufacturing</h1>", unsafe_allow_html=True)
st.markdown("<h3>Predicting Waste Generation and Providing Recommendations</h3>", unsafe_allow_html=True)

# Lottie animation (Optional)
lottie_animation = load_lottie_file("Animation - 1743591615160.json")  # Place this file in the same folder

# Uncomment below if you want to display an animation (if Lottie file exists)
if lottie_animation:
    st_lottie(lottie_animation, height=300, key="waste_reduction")
else:
    st.info("No animation available (Lottie file missing or invalid).")

st.write("Upload your manufacturing data CSV to begin analysis.")

# ----------- File Upload ----------- #

uploaded_file = st.file_uploader('Upload your CSV file', type=['csv'])

if uploaded_file:
    @st.cache_data
    def cached_load_data(file):
        return load_data(file)

    df = cached_load_data(uploaded_file)
    st.write('### Data Preview')
    st.write(df.head())

    # Split data
    X = df.drop('Waste_Generation', axis=1)
    y = df['Waste_Generation']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with st.spinner("Training models..."):
        models, feature_importance = train_models(X_train, y_train, X_test)

    

    # Recommendations
    recommendations = generate_recommendations(feature_importance['Random Forest'])
    st.write('### Recommendations for Waste Reduction')
    for rec in recommendations:
        st.write(f'- {rec}')

    # Cost-Benefit Analysis
    st.write('### Cost-Benefit Analysis')
    feature = st.selectbox('Select Feature', X.columns)
    reduction_percentage = st.slider('Reduction Percentage', 0, 100, 10)
    if st.button('Analyze Cost-Benefit'):
        analysis = cost_benefit_analysis(df, feature, reduction_percentage)
        st.write(analysis)

    # PDF Report
    if st.button('Generate PDF Report'):
        with st.spinner("Generating PDF..."):
            model_name, mae, mse, r2 = best_model_metrics
            generate_pdf_report(mae, mse, r2, recommendations)
        st.success('📄 PDF Report Generated Successfully!')
        st.download_button('Download PDF', 'Waste_Reduction_Report.pdf', mime='application/pdf')
