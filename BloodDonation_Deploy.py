import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# --- Page Configuration ---
st.set_page_config(page_title="Blood Donation Analysis", layout="wide")

# --- Main Title ---
st.title("🩸 Blood Donation Eligibility Analysis & Prediction System")

# --- Data Loading Function ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('blood_donation.csv')
        return df
    except FileNotFoundError:
        return None

# Load the raw data
df = load_data()

# Handle file not found
if df is None:
    st.warning("The file 'blood_donation.csv' was not found. Please upload it below.")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

# =========================================================
# === Data Cleaning & Processing ===
# =========================================================

# 1. Drop unnecessary columns
cols_to_drop = ['Full_Name', 'Contact_Number', 'Email', 'Country', 'Donor_ID']
existing_drop = [c for c in cols_to_drop if c in df.columns]
if existing_drop:
    df_clean = df.drop(columns=existing_drop)
else:
    df_clean = df.copy()

# 2. Remove 'Other' from Gender
if 'Gender' in df_clean.columns:
    df_clean = df_clean[df_clean['Gender'] != 'Other']

# 3. Date Processing
if 'Last_Donation_Date' in df_clean.columns:
    df_clean['Last_Donation_Date'] = pd.to_datetime(df_clean['Last_Donation_Date'], format='%d-%m-%Y', errors='coerce')
    df_clean['Donation_Year'] = df_clean['Last_Donation_Date'].dt.year

# Save to session state for use across pages
st.session_state['df_clean'] = df_clean
df_viz = df_clean.copy()  # Copy for visualization

# ---------------------------------------------------------
# --- Sidebar Navigation (Separate Pages) ---
# ---------------------------------------------------------
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

# Clear English Page Names
pages = [
    "1. Project Overview & Data",
    "2. Blood Group Distribution",
    "3. Gender Demographics",
    "4. Yearly Donation Trends",
    "5. Average Hemoglobin Levels",
    "6. Weight vs. Hemoglobin",
    "7. Geographic Analysis (Cities)",
    "8. Age Distribution",
    "9. Train Prediction Model",
    "10. Predict Donor Eligibility"
]

selection = st.sidebar.radio("Go to:", pages)

# =========================================================
# === Page Content ===
# =========================================================

# --- Page 1: Overview ---
if selection == "1. Project Overview & Data":
    st.header("📋 Data Overview & Cleaning")
    st.markdown("""
    **Steps performed:**
    1. Dropped ID and Contact columns.
    2. **Removed 'Other'** from Gender category.
    3. Extracted Year from Donation Date.
    """)
    
    st.subheader("Cleaned Data Preview")
    st.dataframe(df_clean.head(10))
    
    st.write(f"**Total Rows:** {df_clean.shape[0]}")
    st.write(f"**Total Columns:** {df_clean.shape[1]}")
    
    # Check if 'Other' exists
    if 'Gender' in df_clean.columns:
        unique_genders = df_clean['Gender'].unique()
        st.success(f"**Current Genders in Data:** {', '.join(unique_genders)}")

# --- Page 2: Blood Groups ---
elif selection == "2. Blood Group Distribution":
    st.header("🩸 Blood Group Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df_viz, x='Blood_Group', palette='viridis', ax=ax)
    ax.set_title("Count of Donors by Blood Group")
    ax.set_xlabel("Blood Group")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    st.info("This chart highlights the most common blood types in the dataset.")

# --- Page 3: Gender ---
elif selection == "3. Gender Demographics":
    st.header("⚤ Gender Distribution")
    
    gender_counts = df_viz['Gender'].value_counts()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    # Define colors for Male/Female
    colors = ['skyblue', 'lightcoral']
    ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title("Male vs Female Donors")
    st.pyplot(fig)
    st.write("**Note:** The category 'Other' has been excluded.")

# --- Page 4: Years ---
elif selection == "4. Yearly Donation Trends":
    st.header("📅 Donations Over the Years")
    
    if 'Donation_Year' in df_viz.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        donation_per_year = df_viz.groupby('Donation_Year').size()
        donation_per_year.plot(kind='line', marker='o', color='green', ax=ax)
        plt.grid(True)
        ax.set_ylabel("Total Donations")
        ax.set_xlabel("Year")
        ax.set_title("Trend of Blood Donations")
        st.pyplot(fig)
    else:
        st.error("Year data is missing.")

# --- Page 5: Hemoglobin ---
elif selection == "5. Average Hemoglobin Levels":
    st.header("🧪 Average Hemoglobin by Gender")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_hb = df_viz.groupby('Gender')['Hemoglobin_g_dL'].mean()
    mean_hb.plot(kind='bar', color=['#2E8B57', '#FFA07A'], edgecolor='black', ax=ax)
    ax.set_ylabel("Hemoglobin (g/dL)")
    ax.set_title("Comparison of Average Hemoglobin Levels")
    
    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    
    st.pyplot(fig)

# --- Page 6: Weight vs Hemoglobin ---
elif selection == "6. Weight vs. Hemoglobin":
    st.header("⚖️ Relationship: Weight vs. Health")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_viz, x='Weight_kg', y='Hemoglobin_g_dL', hue='Gender', palette='Set1', ax=ax)
    ax.set_title("Donor Weight vs. Hemoglobin Level")
    ax.set_xlabel("Weight (kg)")
    ax.set_ylabel("Hemoglobin (g/dL)")
    st.pyplot(fig)
    st.info("Visualizing the correlation between body weight and hemoglobin levels.")

# --- Page 7: Cities ---
elif selection == "7. Geographic Analysis (Cities)":
    st.header("🏙️ Top Participating Cities")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    top_cities = df_viz['City'].value_counts().head(5)
    top_cities.plot(kind='bar', color='#4C72B0', edgecolor='black', ax=ax)
    ax.set_ylabel("Number of Donors")
    ax.set_title("Top 5 Cities by Donation Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# --- Page 8: Age ---
elif selection == "8. Age Distribution":
    st.header("🎂 Age Distribution of Donors")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.hist(df_viz['Age'], bins=15, color='orange', edgecolor='black')
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Donor Ages")
    st.pyplot(fig)

# --- Page 9: Model Training ---
elif selection == "9. Train Prediction Model":
    st.header("🤖 Train Machine Learning Model")
    
    df_ml = df_clean.copy()
    
    # Preprocessing for ML (Encoding)
    label_cols = ['Gender', 'Blood_Group', 'Eligible_for_Donation']
    encoders = {}
    
    # Re-fit encoders to handle cleaned data
    for col in label_cols:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        encoders[col] = le
        
    st.session_state['encoders'] = encoders

    # Features & Target
    features = ['Age', 'Gender', 'Weight_kg', 'Hemoglobin_g_dL', 'Total_Donations', 'Blood_Group']
    X = df_ml[features]
    y = df_ml['Eligible_for_Donation']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Selection
    model_choice = st.selectbox("Select Algorithm:", 
        ["Random Forest", "Decision Tree", "Logistic Regression", "KNN", "SVM"])

    if st.button("Start Training 🚀"):
        if model_choice == "Random Forest": model = RandomForestClassifier()
        elif model_choice == "Decision Tree": model = DecisionTreeClassifier()
        elif model_choice == "Logistic Regression": model = LogisticRegression()
        elif model_choice == "KNN": model = KNeighborsClassifier()
        else: model = SVC()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        st.session_state['model'] = model
        st.success(f"✅ Model **{model_choice}** trained successfully!")
        st.metric("Accuracy Score", f"{acc*100:.2f}%")

        # Correlation Heatmap
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_ml.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

# --- Page 10: Prediction ---
elif selection == "10. Predict Donor Eligibility":
    st.header("🩺 Check New Donor Eligibility")

    # Ensure model is trained
    if 'model' not in st.session_state:
        st.warning("⚠️ Please train the model first in the 'Train Prediction Model' page.")
        st.stop()

    model = st.session_state['model']
    encoders = st.session_state['encoders']

    # Input Form
    st.subheader("Enter Donor Details:")
    c1, c2 = st.columns(2)
    
    with c1:
        age = st.number_input("Age", min_value=18, max_value=65, value=25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        weight = st.number_input("Weight (kg)", min_value=45.0, max_value=150.0, value=65.0)
        
    with c2:
        hb = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=13.0)
        donations = st.number_input("Previous Donations", min_value=0, max_value=50, value=0)
        bg = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])

    if st.button("Check Eligibility"):
        try:
            # Transform categorical inputs
            g_enc = encoders['Gender'].transform([gender])[0]
            bg_enc = encoders['Blood_Group'].transform([bg])[0]
            
            input_data = np.array([[age, g_enc, weight, hb, donations, bg_enc]])
            
            # Predict
            pred = model.predict(input_data)
            res = encoders['Eligible_for_Donation'].inverse_transform(pred)[0]

            st.markdown("---")
            # Check result
            # Adjust string check based on your CSV values (e.g., "Yes", "1", "Eligible")
            if str(res).lower() in ["yes", "1", "eligible", "true"]:
                st.success("✅ **Eligible:** This donor can donate blood.")
                st.balloons()
            else:
                st.error("❌ **Not Eligible:** This donor cannot donate at this time.")
                
                # Simple Logic Feedback
                if hb < 12.5: st.warning("Reason: Low Hemoglobin level.")
                if weight < 50: st.warning("Reason: Weight below standard limit.")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
