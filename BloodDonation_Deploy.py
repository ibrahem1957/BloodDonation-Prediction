import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page Configuration
st.set_page_config(
    page_title="Blood Donation Analysis & AI Prediction",
    page_icon="🩸",
    layout="wide"
)

# --- 1. Load Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('blood_donation.csv')
        return df
    except FileNotFoundError:
        return None

# --- 2. Preprocessing & Model Training ---
def process_and_train(df):
    data = df.copy()
    
    # Date Processing
    if 'Last_Donation_Date' in data.columns:
        data['Last_Donation_Date'] = pd.to_datetime(data['Last_Donation_Date'], format='%d-%m-%Y', errors='coerce')
        data['Donation_Year'] = data['Last_Donation_Date'].dt.year

    # Encoding for ML
    encoders = {}
    label_cols = ['Gender', 'Blood_Group', 'Eligible_for_Donation']
    
    # Fit encoders on the whole dataset to ensure consistency
    for col in label_cols:
        le = LabelEncoder()
        data[f'{col}_Encoded'] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    # Features & Target
    feature_cols = ['Age', 'Gender_Encoded', 'Weight_kg', 'Hemoglobin_g_dL', 'Total_Donations', 'Blood_Group_Encoded']
    X = data[feature_cols]
    y = data['Eligible_for_Donation_Encoded']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Model (Best performing in notebook)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    
    return data, model, encoders, acc

# --- Main Application ---
st.title("🩸 Blood Donation Analysis & Prediction")

raw_df = load_data()

if raw_df is None:
    st.error("Error: 'blood_donation.csv' file not found. Please make sure it is in the same directory.")
else:
    # Process data and Train Model
    df_clean, model, encoders, accuracy = process_and_train(raw_df)

    # Tabs for better organization
    tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 AI Prediction"])

    # ================= TAB 1: DASHBOARD =================
    with tab1:
        st.header("Data Visualization")
        
        # Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Donors by Blood Group")
            fig1 = plt.figure(figsize=(8, 5))
            sns.countplot(data=raw_df, x='Blood_Group', palette='viridis')
            plt.title('Number of Donors by Blood Group')
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig1)

        with col2:
            st.subheader("2. Distribution by Gender")
            donor_counts = raw_df.groupby('Gender')['Blood_Group'].count()
            fig2 = plt.figure(figsize=(6, 6))
            plt.pie(donor_counts, labels=donor_counts.index, autopct='%1.1f%%', colors=['skyblue','lightcoral'])
            plt.title('Donor Distribution by Gender')
            st.pyplot(fig2)

        # Row 2
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("3. Donations per Year")
            if 'Donation_Year' in df_clean.columns:
                donation_per_year = df_clean.groupby('Donation_Year')['Blood_Group'].count()
                fig3 = plt.figure(figsize=(8, 5))
                donation_per_year.plot(kind='line', marker='o', color='skyblue')
                plt.title('Number of Donations per Year')
                plt.grid(True)
                st.pyplot(fig3)
            else:
                st.warning("Year data not available")

        with col4:
            st.subheader("4. Donations per Year by Gender")
            if 'Donation_Year' in df_clean.columns:
                gender_year = df_clean.groupby("Donation_Year")['Gender'].value_counts().unstack(fill_value=0)
                fig4 = plt.figure(figsize=(8, 5))
                gender_year.plot(kind='bar', stacked=False, ax=plt.gca())
                plt.title("Donations per Year by Gender")
                plt.grid(axis='y')
                st.pyplot(fig4)

        # Row 3
        col5, col6 = st.columns(2)

        with col5:
            st.subheader("5. Average Hemoglobin by Gender")
            mean_hb = raw_df.groupby('Gender')['Hemoglobin_g_dL'].mean()
            fig5, ax5 = plt.subplots(figsize=(8, 5))
            mean_hb.plot(kind='bar', color=['#2E8B57', '#FFA07A'], edgecolor='black', ax=ax5)
            plt.title('Average Hemoglobin by Gender')
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig5)

        with col6:
            st.subheader("6. Top 5 Cities")
            top_cities = raw_df.groupby("City")['Blood_Group'].count().sort_values(ascending=False).head()
            fig6, ax6 = plt.subplots(figsize=(8, 5))
            top_cities.plot(kind='bar', color=['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974'], edgecolor='black', ax=ax6)
            plt.title('Top 5 Cities by Number of Donors')
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig6)

        # Row 4
        col7, col8 = st.columns(2)

        with col7:
            st.subheader("7. Age Distribution")
            fig7 = plt.figure(figsize=(8, 5))
            plt.hist(raw_df['Age'], bins=10, color='skyblue', edgecolor='black')
            plt.title('Distribution of Donors Age')
            st.pyplot(fig7)

        with col8:
            st.subheader("8. Age by Blood Group")
            fig8 = plt.figure(figsize=(8, 5))
            sns.boxplot(data=raw_df, x='Blood_Group', y='Age', palette='Set2')
            plt.title('Age Distribution by Blood Group')
            st.pyplot(fig8)

        # Row 5
        col9, col10 = st.columns(2)

        with col9:
            st.subheader("9. Weight vs Hemoglobin")
            fig9 = plt.figure(figsize=(8, 5))
            sns.scatterplot(data=raw_df, x='Weight_kg', y='Hemoglobin_g_dL', hue='Gender', palette='Set1')
            plt.title('Weight vs Hemoglobin by Gender')
            st.pyplot(fig9)

        with col10:
            st.subheader("10. Correlation Heatmap")
            fig10 = plt.figure(figsize=(10, 8))
            # Calculate correlation only on numeric encoded columns
            numeric_df = df_clean.select_dtypes(include=[np.number])
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
            plt.title('Feature Correlation')
            st.pyplot(fig10)

    # ================= TAB 2: ML PREDICTION =================
    with tab2:
        st.header("🤖 Check Donation Eligibility")
        st.markdown(f"**Model Used:** Random Forest Classifier | **Accuracy:** {accuracy*100:.2f}%")
        
        st.write("### Enter Donor Details:")
        
        # Input Form in Columns
        c1, c2, c3 = st.columns(3)
        
        with c1:
            in_gender = st.selectbox("Gender", encoders['Gender'].classes_)
            in_blood = st.selectbox("Blood Group", encoders['Blood_Group'].classes_)
        
        with c2:
            in_age = st.number_input("Age", min_value=18, max_value=65, value=30)
            in_weight = st.number_input("Weight (kg)", min_value=40.0, max_value=150.0, value=70.0)
            
        with c3:
            in_hb = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=14.0)
            in_donations = st.number_input("Total Previous Donations", min_value=0, max_value=50, value=0)

        # Prediction Logic
        if st.button("Predict Eligibility", type="primary"):
            # Encode inputs
            gen_code = encoders['Gender'].transform([in_gender])[0]
            blood_code = encoders['Blood_Group'].transform([in_blood])[0]
            
            # Create feature array [Age, Gender, Weight, Hemoglobin, Total_Donations, Blood_Group]
            input_vector = np.array([[in_age, gen_code, in_weight, in_hb, in_donations, blood_code]])
            
            # Predict
            pred = model.predict(input_vector)[0]
            result = encoders['Eligible_for_Donation'].inverse_transform([pred])[0]
            
            st.markdown("---")
            if result == 'Yes':
                st.success(f"✅ **Eligible**: This donor is likely eligible to donate.")
                st.balloons()
            else:
                st.error(f"❌ **Not Eligible**: This donor is mostly NOT eligible to donate based on the model.")
                st.info("Note: This could be due to low hemoglobin, weight, or donation history.")

# Footer
st.markdown("---")
st.markdown("Created with Streamlit based on Random Forest Analysis")
