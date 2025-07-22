import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    st.write("Columns after encoding:", df.columns)
    return df

# Train models and get predictions
def train_models(X_train, y_train, X_test):
    models = {}
    feature_importance = {}

    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf.predict(X_test)
    feature_importance['Random Forest'] = dict(zip(X_train.columns, rf.feature_importances_))

    # Support Vector Regressor
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(X_train, y_train)
    models['SVR'] = svr.predict(X_test)

    return models, feature_importance



# Generate waste reduction strategies
def generate_recommendations(feature_importance):
    recommendations = []
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features:
        if 'Energy' in feature:
            recommendations.append(f'🔋 Reduce energy consumption to lower waste generation.')
        elif 'Greenhouse' in feature:
            recommendations.append(f'🌍 Adopt cleaner technologies to minimize greenhouse gas emissions.')
        elif 'Water' in feature:
            recommendations.append(f'💧 Implement water-saving measures to reduce waste.')
        elif 'Pollutants' in feature:
            recommendations.append(f'🛑 Install filtration systems to reduce pollutant emissions.')
        else:
            recommendations.append(f'⚙️ Optimize processes related to {feature} to minimize waste.')
    return recommendations

# Cost-Benefit Analysis
def cost_benefit_analysis(df, feature, reduction_percentage):
    avg_waste = df['Waste_Generation'].mean()
    avg_revenue = df['Sales_Revenue'].mean()
    estimated_reduction = avg_waste * (reduction_percentage / 100)
    cost_saving = estimated_reduction * (avg_revenue / avg_waste)
    return f'Reducing {feature} by {reduction_percentage}% may save approximately ${cost_saving:.2f} in revenue and also reduce waste by {estimated_reduction:.2f} units.'

# Generate PDF Report
def generate_pdf_report(mae, mse, r2, recommendations):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Waste Reduction in Manufacturing - Report", ln=True, align='C')
    pdf.cell(200, 10, txt="Model Evaluation Metrics", ln=True)
    pdf.cell(200, 10, txt=f"Mean Absolute Error: {mae}".encode('latin-1', 'replace').decode('latin-1'), ln=True)
    pdf.cell(200, 10, txt=f"Mean Squared Error: {mse}".encode('latin-1', 'replace').decode('latin-1'), ln=True)
    pdf.cell(200, 10, txt=f"R-Squared Score: {r2}".encode('latin-1', 'replace').decode('latin-1'), ln=True)

    pdf.cell(200, 10, txt="Recommendations for Waste Reduction:", ln=True)
    for rec in recommendations:
        pdf.cell(200, 10, txt=rec.encode('latin-1', 'replace').decode('latin-1'), ln=True)

    pdf.output("Waste_Reduction_Report.pdf")
    print("PDF Report Generated Successfully!")
