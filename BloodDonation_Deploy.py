import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page Config
st.set_page_config(
    page_title="Blood Donation Eligibility Predictor",
    page_icon="🩸",
    layout="wide"
)

# --- 1. Load Data ---
@st.cache_data
def load_data():
    try:
        # Ensure the csv file is in the same directory
        df = pd.read_csv('blood_donation.csv')
        return df
    except FileNotFoundError:
        return None

# --- 2. Data Preprocessing ---
def preprocess_data(df):
    data = df.copy()
    
    # Date Conversion (as done in your notebook)
    if 'Last_Donation_Date' in data.columns:
        data['Last_Donation_Date'] = pd.to_datetime(data['Last_Donation_Date'], format='%d-%m-%Y', errors='coerce')
        data['Donation_Year'] = data['Last_Donation_Date'].dt.year

    # Select Features for Training
    features = ['Age', 'Gender', 'Weight_kg', 'Hemoglobin_g_dL', 'Total_Donations', 'Blood_Group']
    target = 'Eligible_for_Donation'
    
    # Keep a copy for visualization before encoding
    df_vis = data.copy()
    
    # Handling missing values
    data = data.dropna(subset=features)

    # Label Encoding
    encoders = {}
    label_cols = ['Gender', 'Blood_Group', 'Eligible_for_Donation']
    
    for col in label_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le
        
    return data, df_vis, encoders

# --- 3. Main App Interface ---

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2060/2060267.png", width=100)
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Page", ["Dashboard (Analysis)", "Prediction (AI Model)"])

# Load and Process
raw_df = load_data()

if raw_df is None:
    st.error("Error: 'blood_donation.csv' not found. Please place the dataset in the same directory as this script.")
else:
    df_processed, df_vis, encoders = preprocess_data(raw_df)

    # ================= Page 1: Dashboard =================
    if app_mode == "Dashboard (Analysis)":
        st.title("📊 Blood Donor Data Analysis")
        st.markdown("A comprehensive overview of donor distributions and health metrics.")

        # Show Data Sample
        with st.expander("View Raw Data"):
            st.dataframe(raw_df.head())
            st.write(f"Dataset Shape: {raw_df.shape}")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribution by Blood Group")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.countplot(data=df_vis, x='Blood_Group', palette='viridis', ax=ax1)
            plt.title('Number of Donors by Blood Group')
            plt.xlabel('Blood Group')
            plt.ylabel('Count')
            st.pyplot(fig1)

        with col2:
            st.subheader("Average Hemoglobin by Gender")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            mean_hb = df_vis.groupby('Gender')['Hemoglobin_g_dL'].mean()
            mean_hb.plot(kind='bar', color=['#2E8B57', '#FFA07A'], edgecolor='black', ax=ax2)
            plt.title('Average Hemoglobin by Gender')
            plt.ylabel('Hemoglobin (g/dL)')
            st.pyplot(fig2)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Donations Over Years")
            if 'Donation_Year' in df_vis.columns:
                fig3, ax3 = plt.subplots(figsize=(8, 5))
                donation_per_year = df_vis.groupby('Donation_Year')['Blood_Group'].count()
                donation_per_year.plot(kind='line', marker='o', color='skyblue', ax=ax3)
                plt.grid(True)
                st.pyplot(fig3)
            else:
                st.warning("Donation Year data not available.")

        with col4:
            st.subheader("Age Distribution")
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            ax4.hist(df_vis['Age'], bins=15, color='skyblue', edgecolor='black')
            plt.title('Age Distribution')
            st.pyplot(fig4)

        st.subheader("Weight vs. Hemoglobin Relationship")
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df_vis, x='Weight_kg', y='Hemoglobin_g_dL', hue='Gender', palette='Set1', ax=ax5)
        st.pyplot(fig5)

    # ================= Page 2: Prediction =================
    elif app_mode == "Prediction (AI Model)":
        st.title("🤖 AI Eligibility Predictor")
        st.markdown("Enter donor details to predict if they are eligible to donate blood using **Random Forest**.")

        # Train Model on the fly
        X = df_processed[['Age', 'Gender', 'Weight_kg', 'Hemoglobin_g_dL', 'Total_Donations', 'Blood_Group']]
        y = df_processed['Eligible_for_Donation']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and Train Random Forest (Best model from notebook)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Show Accuracy
        acc = accuracy_score(y_test, model.predict(X_test))
        st.success(f"Model Trained Successfully! Accuracy: **{acc*100:.2f}%**")

        st.markdown("---")
        
        # Input Form
        col_input1, col_input2, col_input3 = st.columns(3)

        with col_input1:
            input_gender = st.selectbox("Gender", encoders['Gender'].classes_)
            input_blood = st.selectbox("Blood Group", encoders['Blood_Group'].classes_)

        with col_input2:
            input_age = st.number_input("Age", min_value=18, max_value=65, value=30)
            input_weight = st.number_input("Weight (kg)", min_value=45.0, max_value=150.0, value=70.0)

        with col_input3:
            input_hb = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=13.5)
            input_donations = st.number_input("Previous Donations", min_value=0, max_value=50, value=0)

        # Predict Button
        if st.button("Check Eligibility", type="primary"):
            # Encode categorical inputs
            gender_encoded = encoders['Gender'].transform([input_gender])[0]
            blood_encoded = encoders['Blood_Group'].transform([input_blood])[0]
            
            # Prepare input array
            input_data = np.array([[input_age, gender_encoded, input_weight, input_hb, input_donations, blood_encoded]])
            
            # Make Prediction
            prediction_idx = model.predict(input_data)[0]
            prediction_text = encoders['Eligible_for_Donation'].inverse_transform([prediction_idx])[0]
            
            st.markdown("---")
            if prediction_text == 'Yes':
                st.balloons()
                st.success("✅ **Eligible for Donation**")
                st.info("The donor meets the medical criteria based on the AI model.")
            else:
                st.error("❌ **Not Eligible**")
                st.warning("The donor may have low hemoglobin or other contraindications based on historical patterns.")

# Footer
st.markdown("---")
st.markdown("Developed using Python & Streamlit based on provided Data Analysis Notebook.")
