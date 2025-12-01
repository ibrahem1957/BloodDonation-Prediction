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

# إعدادات الصفحة
st.set_page_config(page_title="Blood Donation Analysis", layout="wide")

# --- العنوان ---
st.title("🩸 نظام تحليل وتوقع أهلية التبرع بالدم")

# --- دالة تحميل البيانات ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('blood_donation.csv')
        return df
    except FileNotFoundError:
        return None

df = load_data()

# التحقق من وجود الملف
if df is None:
    st.warning("لم يتم العثور على ملف 'blood_donation.csv'. يرجى رفع الملف.")
    uploaded_file = st.file_uploader("ارفع ملف CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

# ---------------------------------------------------------
# --- القائمة الجانبية (توزيع الصفحات لكل رسمة) ---
st.sidebar.title("لوحة التحكم")
st.sidebar.markdown("---")
options = st.sidebar.radio("تصفح الأقسام:", 
    [
        "1. عرض البيانات وتنظيفها",
        "2. رسم: توزيع فصائل الدم",
        "3. رسم: توزيع الجنس",
        "4. رسم: التبرعات عبر السنوات",
        "5. رسم: متوسط الهيموجلوبين",
        "6. رسم: علاقة الوزن بالهيموجلوبين",
        "7. رسم: أفضل المدن",
        "8. رسم: توزيع الأعمار",
        "9. تدريب النموذج (AI)",
        "10. فحص متبرع جديد"
    ])
# ---------------------------------------------------------

# --- معالجة البيانات الأولية (تعمل في الخلفية لكل الصفحات) ---
# تنظيف الأعمدة
cols_to_drop = ['Full_Name', 'Contact_Number', 'Email', 'Country', 'Donor_ID']
existing_drop = [c for c in cols_to_drop if c in df.columns]
if existing_drop:
    df_clean = df.drop(columns=existing_drop)
else:
    df_clean = df.copy()

# معالجة التواريخ
if 'Last_Donation_Date' in df_clean.columns:
    df_clean['Last_Donation_Date'] = pd.to_datetime(df_clean['Last_Donation_Date'], format='%d-%m-%Y', errors='coerce')
    df_clean['Donation_Year'] = df_clean['Last_Donation_Date'].dt.year

# حفظ البيانات في session
st.session_state['df_clean'] = df_clean
df_viz = df_clean  # متغير للعرض

# =========================================================
# === الصفحات ===
# =========================================================

# --- 1. عرض البيانات ---
if options == "1. عرض البيانات وتنظيفها":
    st.header("تنظيف واستعراض البيانات")
    st.write("معاينة أول 5 صفوف من البيانات بعد التنظيف:")
    st.dataframe(df_clean.head())
    st.write(f"**عدد الصفوف والأعمدة:** {df_clean.shape}")
    st.success("تم تجهيز البيانات للتحليل.")

# --- 2. رسم: فصائل الدم ---
elif options == "2. رسم: توزيع فصائل الدم":
    st.header("توزيع فصائل الدم (Blood Groups)")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df_viz, x='Blood_Group', palette='viridis', ax=ax)
    ax.set_title("عدد المتبرعين لكل فصيلة دم")
    st.pyplot(fig)
    st.info("هذا الرسم يوضح أي الفصائل هي الأكثر شيوعاً بين المتبرعين.")

# --- 3. رسم: الجنس ---
elif options == "3. رسم: توزيع الجنس":
    st.header("توزيع الجنس (Gender Distribution)")
    fig, ax = plt.subplots(figsize=(6, 6))
    donor_counts = df_viz.groupby('Gender')['Blood_Group'].count()
    ax.pie(donor_counts, labels=donor_counts.index, autopct='%1.1f%%', colors=['skyblue','lightcoral'])
    ax.set_title("نسبة الذكور إلى الإناث")
    st.pyplot(fig)

# --- 4. رسم: السنوات ---
elif options == "4. رسم: التبرعات عبر السنوات":
    st.header("نشاط التبرع عبر السنوات")
    if 'Donation_Year' in df_viz.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        donation_per_year = df_viz.groupby('Donation_Year').size()
        donation_per_year.plot(kind='line', marker='o', color='green', ax=ax)
        plt.grid()
        ax.set_ylabel("عدد التبرعات")
        st.pyplot(fig)
    else:
        st.error("لا يوجد عمود للتاريخ.")

# --- 5. رسم: الهيموجلوبين ---
elif options == "5. رسم: متوسط الهيموجلوبين":
    st.header("متوسط الهيموجلوبين حسب الجنس")
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_hb = df_viz.groupby('Gender')['Hemoglobin_g_dL'].mean()
    mean_hb.plot(kind='bar', color=['#2E8B57', '#FFA07A'], edgecolor='black', ax=ax)
    ax.set_ylabel("مستوى الهيموجلوبين (g/dL)")
    st.pyplot(fig)

# --- 6. رسم: الوزن والهيموجلوبين ---
elif options == "6. رسم: علاقة الوزن بالهيموجلوبين":
    st.header("العلاقة بين الوزن ومستوى الهيموجلوبين")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_viz, x='Weight_kg', y='Hemoglobin_g_dL', hue='Gender', palette='Set1', ax=ax)
    ax.set_title("توزيع الوزن مقابل الهيموجلوبين")
    st.pyplot(fig)
    st.info("نلاحظ هنا هل هناك علاقة طردية بين وزن المتبرع وصحته (الهيموجلوبين).")

# --- 7. رسم: المدن ---
elif options == "7. رسم: أفضل المدن":
    st.header("أكثر المدن مشاركة في التبرع")
    fig, ax = plt.subplots(figsize=(10, 5))
    top_cities = df_viz['City'].value_counts().head(5)
    top_cities.plot(kind='bar', color='#4C72B0', edgecolor='black', ax=ax)
    ax.set_ylabel("عدد المتبرعين")
    st.pyplot(fig)

# --- 8. رسم: العمر ---
elif options == "8. رسم: توزيع الأعمار":
    st.header("توزيع أعمار المتبرعين")
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.hist(df_viz['Age'], bins=15, color='orange', edgecolor='black')
    ax.set_xlabel("العمر")
    ax.set_ylabel("التكرار")
    st.pyplot(fig)

# --- 9. تدريب النموذج ---
elif options == "9. تدريب النموذج (AI)":
    st.header("🤖 تدريب نموذج الذكاء الاصطناعي")
    
    df_ml = df_clean.copy()
    
    # التجهيز (Encoding)
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

    model_choice = st.selectbox("اختر الخوارزمية:", 
        ["Random Forest", "Decision Tree", "Logistic Regression", "KNN", "SVM"])

    if st.button("ابدأ التدريب"):
        if model_choice == "Random Forest": model = RandomForestClassifier()
        elif model_choice == "Decision Tree": model = DecisionTreeClassifier()
        elif model_choice == "Logistic Regression": model = LogisticRegression()
        elif model_choice == "KNN": model = KNeighborsClassifier()
        else: model = SVC()

        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        
        st.session_state['model'] = model
        st.success(f"تم تدريب {model_choice} بدقة: {acc*100:.2f}%")

        # رسم مصفوفة الارتباط هنا كجزء من تحليل النموذج
        st.subheader("مصفوفة الارتباط (Correlation Matrix)")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_ml.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# --- 10. التوقع ---
elif options == "10. فحص متبرع جديد":
    st.header("فحص أهلية متبرع جديد")

    if 'model' not in st.session_state:
        st.warning("يرجى تدريب النموذج أولاً من صفحة التدريب.")
        st.stop()

    model = st.session_state['model']
    encoders = st.session_state['encoders']

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("العمر", 18, 65, 25)
        gender = st.selectbox("الجنس", ["Male", "Female"])
        weight = st.number_input("الوزن (kg)", 45.0, 150.0, 65.0)
    with c2:
        hb = st.number_input("الهيموجلوبين", 5.0, 20.0, 13.0)
        donations = st.number_input("تبرعات سابقة", 0, 50, 0)
        bg = st.selectbox("فصيلة الدم", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])

    if st.button("هل هو مؤهل؟"):
        try:
            g_enc = encoders['Gender'].transform([gender])[0]
            bg_enc = encoders['Blood_Group'].transform([bg])[0]
            input_data = np.array([[age, g_enc, weight, hb, donations, bg_enc]])
            
            pred = model.predict(input_data)
            res = encoders['Eligible_for_Donation'].inverse_transform(pred)[0]

            if res in ["Yes", 1, "Eligible"]:
                st.success("✅ مؤهل للتبرع")
            else:
                st.error("❌ غير مؤهل")
        except:
            st.error("حدث خطأ في البيانات المدخلة")
