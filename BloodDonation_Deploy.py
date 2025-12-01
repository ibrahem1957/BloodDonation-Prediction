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

# Load Data
df = load_data()

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

# Save to session state
st.session_state['df_clean'] = df_clean
df_viz = df_clean.copy()

# ---------------------------------------------------------
# --- Sidebar Navigation ---
# ---------------------------------------------------------
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

pages = [
    "1. Data Overview, Shape & Info",
    "2. Blood Group Distribution",
    "3. Gender Demographics",
    "4. Yearly Donation Trends",
    "5. Donations by Gender (Yearly)",
    "6. Average Hemoglobin Levels",
    "7. Weight vs. Hemoglobin",
    "8. Geographic Analysis (Cities)",
    "9. Age Distribution (Histogram)",
    "10. Age by Blood Group (Boxplot)",
    "11. Train Prediction Model",
    "12. Predict Donor Eligibility"
]

selection = st.sidebar.radio("Go to:", pages)

# =========================================================
# === Page Content ===
# =========================================================

# --- Page 1: Overview, Shape & Info ---
if selection == "1. Data Overview, Shape & Info":
    st.header("📋 Data Overview")
    
    # Data Shape (The code you requested)
    st.subheader("1. Dataset Shape")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", df_clean.shape[0])
    with col2:
        st.metric("Total Columns", df_clean.shape[1])

    # Data Info (Types and Nulls)
    st.subheader("2. Dataset Info (Columns & Types)")
    buffer = pd.DataFrame({
        'Column Name': df_clean.columns,
        'Data Type': df_clean.dtypes.astype(str),
        'Non-Null Count': df_clean.count()
    }).reset_index(drop=True)
    st.dataframe(buffer, use_container_width=True)

    # Preview
    st.subheader("3. Sample Data")
    st.dataframe(df_clean.head())
    st.success("Data cleaning applied: Removed 'Other' gender, dropped ID columns, extracted Year.")

# --- Page 2: Blood Groups ---
elif selection == "2. Blood Group Distribution":
    st.header("🩸 Blood Group Distribution")
    
    # Small size
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(data=df_viz, x='Blood_Group', palette='viridis', ax=ax)
    ax.set_title("Count of Donors by Blood Group", fontsize=10)
    ax.tick_params(labelsize=8)
    st.pyplot(fig)

    # Custom Note
    st.markdown("### 🩸 Compatibility & Distribution Notes")
    st.markdown("""
    | Blood Type | Can Donate To | Can Receive From |
    | :--- | :--- | :--- |
    | **O-** | All types | O- |
    | **O+** | O+, A+, B+, AB+ | O-, O+ |
    | **A-** | A-, A+, AB-, AB+ | A-, O- |
    | **A+** | A+, AB+ | A+, A-, O+, O- |
    | **B-** | B-, B+, AB-, AB+ | B-, O- |
    | **B+** | B+, AB+ | B+, B-, O+, O- |
    | **AB-** | AB-, AB+ | AB-, A-, B-, O- |
    | **AB+** | AB+ | All types |

    **Reasons for Blood Type Distribution in India:**
    * **Genetic reason:** O+ and B+ are more common, A+ less common, AB+ rare.
    * **Compatibility reason:** O+ appears more in donations due to high demand.
    """)

# --- Page 3: Gender ---
elif selection == "3. Gender Demographics":
    st.header("⚤ Gender Distribution")
    gender_counts = df_viz['Gender'].value_counts()
    
    # Small Pie Chart
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['skyblue', 'lightcoral'], startangle=90, textprops={'fontsize': 8})
    ax.set_title("Male vs Female", fontsize=10)
    st.pyplot(fig)

    st.info("**Insight:** Displays the gender ratio (Male vs Female only).")

# --- Page 4: Years ---
elif selection == "4. Yearly Donation Trends":
    st.header("📅 Donations Over the Years")
    if 'Donation_Year' in df_viz.columns:
        # Small Line Chart
        fig, ax = plt.subplots(figsize=(6, 3))
        donation_per_year = df_viz.groupby('Donation_Year').size()
        donation_per_year.plot(kind='line', marker='o', color='green', ax=ax)
        plt.grid(True)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=8)
        st.pyplot(fig)
    else:
        st.error("Year column not found.")

# --- Page 5: Donations by Gender & Year ---
elif selection == "5. Donations by Gender (Yearly)":
    st.header("📊 Donations per Year by Gender")
    
    if 'Donation_Year' in df_viz.columns and 'Gender' in df_viz.columns:
        gender_year = df_viz.groupby("Donation_Year")['Gender'].value_counts().unstack(fill_value=0)
        
        # Small Stacked Bar
        fig, ax = plt.subplots(figsize=(6, 3.5))
        gender_year.plot(kind='bar', ax=ax)
        ax.set_title("Donations per Year by Gender", fontsize=10)
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        plt.xticks(rotation=0, fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(axis='y')
        st.pyplot(fig)
    else:
        st.error("Missing necessary columns.")

# --- Page 6: Hemoglobin ---
elif selection == "6. Average Hemoglobin Levels":
    st.header("🧪 Average Hemoglobin by Gender")
    # Small Bar Chart
    fig, ax = plt.subplots(figsize=(5, 3))
    mean_hb = df_viz.groupby('Gender')['Hemoglobin_g_dL'].mean()
    mean_hb.plot(kind='bar', color=['#2E8B57', '#FFA07A'], edgecolor='black', ax=ax)
    ax.set_ylabel("Hemoglobin (g/dL)", fontsize=9)
    plt.xticks(rotation=0, fontsize=9)
    st.pyplot(fig)

    st.markdown("### 📋 Medical Reference")
    st.markdown("""
    | Gender | Normal Hemoglobin (g/dL) |
    | :--- | :--- |
    | **Men** | 13.8 – 17.2 |
    | **Women** | 12.1 – 15.1 |
    """)

# --- Page 7: Weight vs Hemoglobin ---
elif selection == "7. Weight vs. Hemoglobin":
    st.header("⚖️ Weight vs. Hemoglobin")
    # Small Scatter
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df_viz, x='Weight_kg', y='Hemoglobin_g_dL', hue='Gender', palette='Set1', ax=ax, s=15)
    ax.tick_params(labelsize=8)
    st.pyplot(fig)
    st.info("**Insight:** Correlation between weight and health metrics.")

# --- Page 8: Cities ---
elif selection == "8. Geographic Analysis (Cities)":
    st.header("🏙️ Top Cities by Donation Count")
    # Small Bar Chart
    fig, ax = plt.subplots(figsize=(6, 3.5))
    top_cities = df_viz['City'].value_counts().head(5)
    top_cities.plot(kind='bar', color='#4C72B0', edgecolor='black', ax=ax)
    plt.xticks(rotation=15, fontsize=8)
    plt.yticks(fontsize=8)
    st.pyplot(fig)

# --- Page 9: Age Distribution ---
elif selection == "9. Age Distribution (Histogram)":
    st.header("🎂 Distribution of Donors Age")
    
    # Small Hist
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df_viz['Age'], bins=10, color='skyblue', edgecolor='black')
    ax.set_title('Distribution of Donors Age', fontsize=10)
    ax.set_xlabel('Age', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.tick_params(labelsize=8)
    st.pyplot(fig)

# --- Page 10: Age by Blood Group ---
elif selection == "10. Age by Blood Group (Boxplot)":
    st.header("🩸 Age Distribution by Blood Group")
    
    # Small Boxplot
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df_viz, x='Blood_Group', y='Age', palette='Set2', ax=ax, linewidth=1)
    ax.set_title('Age Distribution by Blood Group', fontsize=10)
    ax.tick_params(labelsize=8)
    st.pyplot(fig)

# --- Page 11: Model Training ---
elif selection == "11. Train Prediction Model":
    st.header("🤖 Train Machine Learning Model")
    
    df_ml = df_clean.copy()
    
    # Encoding
    label_cols = ['Gender', 'Blood_Group', 'Eligible_for_Donation']
    encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        encoders[col] = le
    st.session_state['encoders'] = encoders

    features = ['Age', 'Gender', 'Weight_kg', 'Hemoglobin_g_dL', 'Total_Donations', 'Blood_Group']
    X = df_ml[features]
    y = df_ml['Eligible_for_Donation']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_choice = st.selectbox("Select Algorithm:", 
        ["Random Forest", "Decision Tree", "Logistic Regression", "KNN", "SVM"])

    if st.button("Start Training 🚀"):
        if model_choice == "Random Forest": model = RandomForestClassifier()
        elif model_choice == "Decision Tree": model = DecisionTreeClassifier()
        elif model_choice == "Logistic Regression": model = LogisticRegression()
        elif model_choice == "KNN": model = KNeighborsClassifier()
        else: model = SVC()

        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        st.session_state['model'] = model
        st.success(f"✅ Model **{model_choice}** trained! Accuracy: {acc*100:.2f}%")

        st.subheader("Correlation Matrix")
        # Small Heatmap
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(df_ml.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", ax=ax, annot_kws={"size": 7})
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        st.pyplot(fig)

# --- Page 12: Prediction ---
elif selection == "12. Predict Donor Eligibility":
    st.header("🩺 Predict Eligibility")

    if 'model' not in st.session_state:
        st.warning("⚠️ Please train the model first!")
        st.stop()

    model = st.session_state['model']
    encoders = st.session_state['encoders']

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 18, 65, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        weight = st.number_input("Weight (kg)", 45.0, 150.0, 65.0)
    with c2:
        hb = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0, 13.0)
        donations = st.number_input("Donations Count", 0, 50, 0)
        bg = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])

    if st.button("Check"):
        try:
            g_enc = encoders['Gender'].transform([gender])[0]
            bg_enc = encoders['Blood_Group'].transform([bg])[0]
            input_data = np.array([[age, g_enc, weight, hb, donations, bg_enc]])
            pred = model.predict(input_data)
            res = encoders['Eligible_for_Donation'].inverse_transform(pred)[0]

            st.markdown("---")
            if str(res).lower() in ["yes", "1", "eligible", "true"]:
                st.success("✅ Eligible to Donate")
            else:
                st.error("❌ Not Eligible")
        except:
            st.error("Error in prediction input.")
