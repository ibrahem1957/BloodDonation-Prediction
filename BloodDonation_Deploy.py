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
        # Try reading the file
        df = pd.read_csv('blood_donation.csv')
        return df
    except FileNotFoundError:
        return None

# Load Data
df = load_data()

if df is None:
    st.warning("⚠️ The file 'blood_donation.csv' was not found. Please upload it below.")
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
    "1. Data Overview & Shape",
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

# --- Page 1: Overview ---
if selection == "1. Data Overview & Shape":
    st.header("📋 Data Overview")
    
    st.subheader("1. Dataset Shape")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", df_clean.shape[0])
    with col2:
        st.metric("Total Columns", df_clean.shape[1])

    st.subheader("2. Sample Data (First 5 Rows)")
    st.dataframe(df_clean.head())
    
    st.info("✅ **Status:** Data has been cleaned, ID columns dropped, and 'Other' gender category removed.")

# --- Page 2: Blood Groups ---
elif selection == "2. Blood Group Distribution":
    st.header("🩸 Blood Group Distribution")
    
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(data=df_viz, x='Blood_Group', palette='viridis', ax=ax)
    ax.set_title("Count of Donors by Blood Group", fontsize=10)
    ax.tick_params(labelsize=8)
    st.pyplot(fig)

    # --- INSIGHTS ---
    st.markdown("### 💡 Insights")
    st.info("""
    * **Observation:** O+ is the most common blood group, followed by B+ and A+. Negative groups (AB-, B-) are very rare.
    * **Implication:** This aligns with general population genetics (especially in Asia/India). While O+ stock is usually high, specific campaigns are needed for rare negative groups.
    """)
    
    st.markdown("#### 🩸 Compatibility Table")
    st.markdown("""
    | Type | Donate To | Receive From |
    | :--- | :--- | :--- |
    | **O+** | O+, A+, B+, AB+ | O+, O- |
    | **O-** | All Types | O- |
    | **AB+**| AB+ | All Types |
    """)

# --- Page 3: Gender ---
elif selection == "3. Gender Demographics":
    st.header("⚤ Gender Distribution")
    gender_counts = df_viz['Gender'].value_counts()
    
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['skyblue', 'lightcoral'], startangle=90, textprops={'fontsize': 8})
    ax.set_title("Male vs Female", fontsize=10)
    st.pyplot(fig)

    # --- INSIGHTS ---
    st.markdown("### 💡 Insights")
    st.info("""
    * **Observation:** The split is almost equal (approx. 50% Male, 50% Female).
    * **Conclusion:** This is a highly positive indicator of social awareness and female health, as females are often deferred due to anemia or pregnancy in many regions.
    """)

# --- Page 4: Years ---
elif selection == "4. Yearly Donation Trends":
    st.header("📅 Donations Over the Years")
    if 'Donation_Year' in df_viz.columns:
        fig, ax = plt.subplots(figsize=(6, 3))
        donation_per_year = df_viz.groupby('Donation_Year').size()
        donation_per_year.plot(kind='line', marker='o', color='green', ax=ax)
        plt.grid(True)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=8)
        st.pyplot(fig)

        # --- INSIGHTS ---
        st.markdown("### 💡 Insights")
        st.info("""
        * **Trend:** There is a sharp exponential growth in donations starting from 2022.
        * **Conclusion:** Strategies for donor acquisition have been highly effective post-2021.
        """)
    else:
        st.error("Year column not found.")

# --- Page 5: Donations by Gender & Year ---
elif selection == "5. Donations by Gender (Yearly)":
    st.header("📊 Donations per Year by Gender")
    
    if 'Donation_Year' in df_viz.columns and 'Gender' in df_viz.columns:
        gender_year = df_viz.groupby("Donation_Year")['Gender'].value_counts().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(6, 3.5))
        gender_year.plot(kind='bar', ax=ax)
        ax.set_title("Donations per Year by Gender", fontsize=10)
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        plt.xticks(rotation=0, fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(axis='y')
        st.pyplot(fig)

        # --- INSIGHTS ---
        st.markdown("### 💡 Insights")
        st.info("""
        * **Observation:** Both genders are contributing almost equally every year.
        * **Conclusion:** Marketing campaigns are effectively targeting both demographics without bias.
        """)
    else:
        st.error("Missing necessary columns.")

# --- Page 6: Hemoglobin ---
elif selection == "6. Average Hemoglobin Levels":
    st.header("🧪 Average Hemoglobin by Gender")
    fig, ax = plt.subplots(figsize=(5, 3))
    mean_hb = df_viz.groupby('Gender')['Hemoglobin_g_dL'].mean()
    mean_hb.plot(kind='bar', color=['#2E8B57', '#FFA07A'], edgecolor='black', ax=ax)
    ax.set_ylabel("Hemoglobin (g/dL)", fontsize=9)
    plt.xticks(rotation=0, fontsize=9)
    
    # Annotate bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=8)

    st.pyplot(fig)

    # --- INSIGHTS ---
    st.markdown("### 💡 Medical Insights")
    st.info("""
    * **Observation:** Males average around **14.5 g/dL**, while Females average around **13.0 g/dL**.
    * **Risk Factor:** Since the donation threshold is typically **12.5 g/dL**, females are closer to the limit, meaning they have a higher risk of deferral due to anemia.
    """)
    
    st.markdown("#### Reference Ranges")
    st.markdown("""
    | Gender | Normal Range (g/dL) |
    | :--- | :--- |
    | **Men** | 13.8 – 17.2 |
    | **Women** | 12.1 – 15.1 |
    """)

# --- Page 7: Weight vs Hemoglobin ---
elif selection == "7. Weight vs. Hemoglobin":
    st.header("⚖️ Weight vs. Hemoglobin")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df_viz, x='Weight_kg', y='Hemoglobin_g_dL', hue='Gender', palette='Set1', ax=ax, s=15)
    ax.tick_params(labelsize=8)
    st.pyplot(fig)

    # --- INSIGHTS ---
    st.markdown("### 💡 Insights")
    st.info("""
    * **Correlation:** There is no clear linear correlation between Weight and Hemoglobin.
    * **Gender Separation:** Blue dots (Males) tend to be higher on the Y-axis (Hemoglobin) regardless of weight.
    * **Eligibility:** Donors must satisfy BOTH criteria (Weight > 50kg AND Hb > 12.5) to be eligible.
    """)

# --- Page 8: Cities ---
elif selection == "8. Geographic Analysis (Cities)":
    st.header("🏙️ Top Cities by Donation Count")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    top_cities = df_viz['City'].value_counts().head(5)
    top_cities.plot(kind='bar', color='#4C72B0', edgecolor='black', ax=ax)
    plt.xticks(rotation=15, fontsize=8)
    plt.yticks(fontsize=8)
    st.pyplot(fig)

    # --- INSIGHTS ---
    st.markdown("### 💡 Insights")
    st.info("""
    * **Balance:** The top 5 cities (Kolkata, Mohali, Patna, Jaipur, Kanpur) show very similar donation numbers (~500 each).
    * **Conclusion:** No single city dominates, indicating a well-distributed donation infrastructure across these regions.
    """)

# --- Page 9: Age Distribution ---
elif selection == "9. Age Distribution (Histogram)":
    st.header("🎂 Distribution of Donors Age")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df_viz['Age'], bins=10, color='skyblue', edgecolor='black')
    ax.set_title('Distribution of Donors Age', fontsize=10)
    ax.set_xlabel('Age', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.tick_params(labelsize=8)
    st.pyplot(fig)

    # --- INSIGHTS ---
    st.markdown("### 💡 Insights")
    st.info("""
    * **Active Demographics:** The most active age group is young adults between **25 and 35 years old**.
    * **Drop-off:** Donation numbers drop significantly after age 50, likely due to the onset of chronic diseases (e.g., hypertension, diabetes) which may disqualify donors.
    """)

# --- Page 10: Age by Blood Group ---
elif selection == "10. Age by Blood Group (Boxplot)":
    st.header("🩸 Age Distribution by Blood Group")
    
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df_viz, x='Blood_Group', y='Age', palette='Set2', ax=ax, linewidth=1)
    ax.set_title('Age Distribution by Blood Group', fontsize=10)
    ax.tick_params(labelsize=8)
    st.pyplot(fig)

    # --- INSIGHTS ---
    st.markdown("### 💡 Insights")
    st.info("""
    * **Observation:** The median age is consistent across all blood groups (approx. 30-35 years).
    * **Conclusion:** Blood type does not correlate with age. Donors for rare blood types can be found in any age group, so campaigns should target the general population rather than specific age demographics.
    """)

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
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(df_ml.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", ax=ax, annot_kws={"size": 7})
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        st.pyplot(fig)
        
        # --- INSIGHTS ---
        st.info("""
        **Correlation Insight:** The numbers in the matrix are close to zero (e.g., 0.01), indicating **No strong linear correlation** between variables.
        This means factors like Age, Weight, and Blood Group are independent of each other, which is good for model training as it avoids multicollinearity.
        """)

# --- Page 12: Prediction ---
elif selection == "12. Predict Donor Eligibility":
    st.header("🩺 Predict Eligibility")

    if 'model' not in st.session_state:
        st.warning("⚠️ Please train the model first in the 'Train Prediction Model' page.")
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

    if st.button("Check Eligibility"):
        try:
            g_enc = encoders['Gender'].transform([gender])[0]
            bg_enc = encoders['Blood_Group'].transform([bg])[0]
            input_data = np.array([[age, g_enc, weight, hb, donations, bg_enc]])
            pred = model.predict(input_data)
            res = encoders['Eligible_for_Donation'].inverse_transform(pred)[0]

            st.markdown("---")
            # Adjust based on your dataset's "Yes" / "1" value
            if str(res).lower() in ["yes", "1", "eligible", "true"]:
                st.success("✅ **Result:** Eligible to Donate")
            else:
                st.error("❌ **Result:** Not Eligible")
                
                # Logic-based Feedback
                if hb < 12.5:
                    st.warning(f"⚠️ Hemoglobin ({hb}) is lower than the required 12.5 g/dL.")
                if weight < 50:
                    st.warning(f"⚠️ Weight ({weight}kg) is below the standard 50kg requirement.")
                if age > 60:
                    st.warning("⚠️ Age is on the higher side, medical consultation recommended.")

        except Exception as e:
            st.error("Error in prediction. Please check inputs.")
