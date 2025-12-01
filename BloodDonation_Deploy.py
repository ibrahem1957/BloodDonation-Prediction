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

# Page Configuration
st.set_page_config(page_title="Blood Donation Prediction", layout="wide")

# --- Title ---
st.title("🩸 Blood Donation Eligibility Prediction System")
st.markdown("""
This application analyzes blood donation data, cleans it, visualizes insights, 
and trains a Machine Learning model (Random Forest) to predict donor eligibility.
""")

# --- Sidebar Navigation ---
st.sidebar.title("Dashboard")
options = st.sidebar.radio("Select a Page:", 
    ["Data Overview & Cleaning", "Exploratory Data Analysis (EDA)", "Model Training", "Prediction"])

# --- Data Loading Function ---
@st.cache_data
def load_data():
    try:
        # Try to read the file locally
        df = pd.read_csv('blood_donation.csv')
        return df
    except FileNotFoundError:
        return None

df = load_data()

# If file is not found automatically
if df is None:
    st.warning("The file 'blood_donation.csv' was not found. Please upload it.")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

# --- Page 1: Data Overview & Cleaning ---
if options == "Data Overview & Cleaning":
    st.header("1. Data Overview & Cleaning")
    
    st.subheader("Raw Data (First 5 Rows)")
    st.write(df.head())

    st.write(f"**Data Shape:** {df.shape}")

    st.subheader("🧹 Data Cleaning Steps")
    
    # Dropping unnecessary columns
    cols_to_drop = ['Full_Name', 'Contact_Number', 'Email', 'Country', 'Donor_ID']
    existing_drop = [c for c in cols_to_drop if c in df.columns]
    
    if existing_drop:
        df_clean = df.drop(columns=existing_drop)
        st.success(f"Dropped unnecessary columns: {existing_drop}")
    else:
        df_clean = df.copy()

    # Date Processing
    if 'Last_Donation_Date' in df_clean.columns:
        df_clean['Last_Donation_Date'] = pd.to_datetime(df_clean['Last_Donation_Date'], format='%d-%m-%Y', errors='coerce')
        df_clean['Donation_Year'] = df_clean['Last_Donation_Date'].dt.year
        df_clean['Donation_Month'] = df_clean['Last_Donation_Date'].dt.month
        st.success("Extracted Year and Month from 'Last_Donation_Date'.")

    st.subheader("Cleaned Data")
    st.dataframe(df_clean.head())
    
    # Store cleaned data in session state
    st.session_state['df_clean'] = df_clean

# --- Page 2: EDA ---
elif options == "Exploratory Data Analysis (EDA)":
    if 'df_clean' not in st.session_state:
        st.error("Please go to the 'Data Overview' page first to load and clean the data.")
        st.stop()
    
    df_viz = st.session_state['df_clean']
    
    st.header("2. Exploratory Data Analysis (EDA)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution of Blood Groups")
        fig, ax = plt.subplots()
        sns.countplot(data=df_viz, x='Blood_Group', palette='viridis', ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Gender Distribution")
        fig, ax = plt.subplots()
        donor_counts = df_viz.groupby('Gender')['Blood_Group'].count()
        ax.pie(donor_counts, labels=donor_counts.index, autopct='%1.1f%%', colors=['skyblue','lightcoral'])
        st.pyplot(fig)

    st.markdown("---")
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Donations per Year")
        if 'Donation_Year' in df_viz.columns:
            donation_per_year = df_viz.groupby('Donation_Year').size()
            fig, ax = plt.subplots()
            donation_per_year.plot(kind='line', marker='o', color='skyblue', ax=ax)
            plt.grid()
            st.pyplot(fig)
    
    with col4:
        st.subheader("Average Hemoglobin by Gender")
        fig, ax = plt.subplots()
        # Use numeric_only=True to avoid errors with non-numeric columns
        mean_hb = df_viz.groupby('Gender')['Hemoglobin_g_dL'].mean()
        mean_hb.plot(kind='bar', color=['#2E8B57', '#FFA07A'], edgecolor='black', ax=ax)
        st.pyplot(fig)

    st.markdown("---")

    st.subheader("Top 5 Cities by Number of Donors")
    fig, ax = plt.subplots(figsize=(10, 4))
    top_cities = df_viz['City'].value_counts().head(5)
    top_cities.plot(kind='bar', color='#4C72B0', edgecolor='black', ax=ax)
    st.pyplot(fig)

    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    plt.hist(df_viz['Age'], bins=15, color='skyblue', edgecolor='black')
    st.pyplot(fig)

    st.subheader("Relationship: Weight vs. Hemoglobin (by Gender)")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_viz, x='Weight_kg', y='Hemoglobin_g_dL', hue='Gender', palette='Set1', ax=ax)
    st.pyplot(fig)


# --- Page 3: Model Training ---
elif options == "Model Training":
    if 'df_clean' not in st.session_state:
        st.error("Data is not ready.")
        st.stop()
    
    df_ml = st.session_state['df_clean'].copy()
    
    st.header("3. Model Training & Evaluation")

    # Feature Engineering (Label Encoding)
    st.markdown("### Feature Engineering (Label Encoding)")
    
    label_cols = ['Gender', 'Blood_Group', 'Eligible_for_Donation']
    encoders = {}
    
    for col in label_cols:
        if col in df_ml.columns:
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            encoders[col] = le
    
    # Save encoders for prediction page
    st.session_state['encoders'] = encoders

    # Feature Selection
    features = ['Age', 'Gender', 'Weight_kg', 'Hemoglobin_g_dL', 'Total_Donations', 'Blood_Group']
    
    # Verify columns exist before proceeding
    missing_cols = [col for col in features if col not in df_ml.columns]
    if missing_cols:
        st.error(f"Missing columns for training: {missing_cols}")
        st.stop()

    X = df_ml[features]
    y = df_ml['Eligible_for_Donation']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Selection Sidebar
    st.sidebar.subheader("Model Settings")
    model_choice = st.sidebar.selectbox("Choose a Model:", 
        ["Random Forest", "Decision Tree", "Logistic Regression", "KNN", "SVM"])

    if model_choice == "Random Forest":
        model = RandomForestClassifier()
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_choice == "Logistic Regression":
        model = LogisticRegression()
    elif model_choice == "KNN":
        model = KNeighborsClassifier()
    else:
        model = SVC()

    if st.button("Start Training 🚀"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(f"Model **{model_choice}** trained successfully!")
        st.metric(label="Model Accuracy", value=f"{acc*100:.2f}%")

        # Save the trained model
        st.session_state['model'] = model
        st.session_state['model_name'] = model_choice

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_ml.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# --- Page 4: Prediction ---
elif options == "Prediction":
    st.header("4. Check Donor Eligibility 🩸")

    if 'model' not in st.session_state or 'encoders' not in st.session_state:
        st.warning("Please train the model first in the 'Model Training' page.")
        st.stop()

    model = st.session_state['model']
    encoders = st.session_state['encoders']

    # User Inputs
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        weight = st.number_input("Weight (kg)", min_value=45.0, max_value=150.0, value=65.0)

    with col2:
        hb = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=13.0)
        donations = st.number_input("Total Previous Donations", min_value=0, max_value=50, value=0)
        blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])

    if st.button("Check Eligibility"):
        try:
            # Transform Inputs
            gender_enc = encoders['Gender'].transform([gender])[0]
            bg_enc = encoders['Blood_Group'].transform([blood_group])[0]

            # Prepare Data
            # Order: ['Age', 'Gender', 'Weight_kg', 'Hemoglobin_g_dL', 'Total_Donations', 'Blood_Group']
            input_data = np.array([[age, gender_enc, weight, hb, donations, bg_enc]])
            
            # Predict
            prediction = model.predict(input_data)
            result_label = encoders['Eligible_for_Donation'].inverse_transform(prediction)[0]

            st.markdown("---")
            
            # Assuming 'Yes'/'1' means Eligible
            # Adjust logic if your label encoder mapped 'Yes' to 1 or 0
            if result_label in ["Yes", 1, "Eligible"]:
                st.success("✅ This person is **ELIGIBLE** to donate blood.")
                st.balloons()
            else:
                st.error("❌ This person is **NOT ELIGIBLE** to donate blood.")
                
                # Rule-based feedback (based on general medical logic)
                st.subheader("Potential Reasons:")
                if hb < 12.5:
                    st.write("- Hemoglobin levels might be too low.")
                if weight < 50:
                    st.write("- Weight is below the standard requirement.")
                if age > 60 or age < 18:
                    st.write("- Age might be a restricting factor.")
                    
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
