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

# --- إعدادات الصفحة ---
st.set_page_config(page_title="تحليل التبرع بالدم", layout="wide")

# --- العنوان الرئيسي ---
st.title("🩸 نظام تحليل وتوقع أهلية التبرع بالدم")

# --- دالة تحميل البيانات ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('blood_donation.csv')
        return df
    except FileNotFoundError:
        return None

# تحميل البيانات
df = load_data()

if df is None:
    st.warning("لم يتم العثور على ملف 'blood_donation.csv'. يرجى رفع الملف أدناه.")
    uploaded_file = st.file_uploader("رفع ملف CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

# =========================================================
# === تنظيف ومعالجة البيانات ===
# =========================================================

# 1. حذف الأعمدة غير الضرورية
cols_to_drop = ['Full_Name', 'Contact_Number', 'Email', 'Country', 'Donor_ID']
existing_drop = [c for c in cols_to_drop if c in df.columns]
if existing_drop:
    df_clean = df.drop(columns=existing_drop)
else:
    df_clean = df.copy()

# 2. حذف 'Other' من الجنس
if 'Gender' in df_clean.columns:
    df_clean = df_clean[df_clean['Gender'] != 'Other']

# 3. معالجة التواريخ
if 'Last_Donation_Date' in df_clean.columns:
    df_clean['Last_Donation_Date'] = pd.to_datetime(df_clean['Last_Donation_Date'], format='%d-%m-%Y', errors='coerce')
    df_clean['Donation_Year'] = df_clean['Last_Donation_Date'].dt.year

# حفظ البيانات في session state
st.session_state['df_clean'] = df_clean
df_viz = df_clean.copy()

# ---------------------------------------------------------
# --- القائمة الجانبية (التنقل) ---
# ---------------------------------------------------------
st.sidebar.title("لوحة التحكم")
st.sidebar.markdown("---")

pages = [
    "1. نظرة عامة على البيانات",
    "2. توزيع فصائل الدم",
    "3. الديموغرافيا (الجنس)",
    "4. اتجاهات التبرع السنوية",
    "5. التبرعات حسب الجنس (سنوياً)",
    "6. متوسط مستويات الهيموجلوبين",
    "7. العلاقة بين الوزن والهيموجلوبين",
    "8. التحليل الجغرافي (المدن)",
    "9. توزيع الأعمار (Histogram)",
    "10. العمر حسب فصيلة الدم (Boxplot)",
    "11. تدريب نموذج التوقع",
    "12. فحص أهلية متبرع جديد"
]

selection = st.sidebar.radio("انتقل إلى:", pages)

# =========================================================
# === محتوى الصفحات ===
# =========================================================

# --- الصفحة 1: نظرة عامة ---
if selection == "1. نظرة عامة على البيانات":
    st.header("📋 نظرة عامة وتنظيف البيانات")
    st.subheader("معاينة البيانات بعد التنظيف")
    st.dataframe(df_clean.head(10))
    st.write(f"**إجمالي الصفوف:** {df_clean.shape[0]} | **إجمالي الأعمدة:** {df_clean.shape[1]}")
    st.info("تم تنظيف البيانات وحذف القيم غير المرغوبة (مثل الجنس 'Other') وتجهيز التواريخ للتحليل.")

# --- الصفحة 2: فصائل الدم (تمت إضافة ملاحظتك هنا) ---
elif selection == "2. توزيع فصائل الدم":
    st.header("🩸 توزيع فصائل الدم")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df_viz, x='Blood_Group', palette='viridis', ax=ax)
    ax.set_title("عدد المتبرعين لكل فصيلة دم")
    st.pyplot(fig)

    # --- الملاحظة الخاصة بك ---
    st.markdown("### 💡 رؤى حول توزيع فصائل الدم")
    st.markdown("""
    **أسباب التوزيع الحالي لفصائل الدم:**
    * **السبب الجيني:** تعتبر فصيلتا **O+** و **B+** الأكثر شيوعاً بين السكان، بينما **A+** أقل شيوعاً، وتعتبر **AB+** نادرة.
    * **سبب التوافق:** يمكن لفصيلة **O+** التبرع لمعظم الفصائل الموجبة، مما يجعلها تظهر بشكل أكبر في سجلات التبرع. بالمقابل، **AB+** تستقبل من الجميع لكنها تتبرع فقط لـ AB، مما قد يفسر قلتها نسبياً في بعض السياقات.
    """)

# --- الصفحة 3: الجنس ---
elif selection == "3. الديموغرافيا (الجنس)":
    st.header("⚤ توزيع الجنس")
    gender_counts = df_viz['Gender'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['skyblue', 'lightcoral'], startangle=90)
    ax.set_title("نسبة الذكور مقابل الإناث")
    st.pyplot(fig)

    st.info("💡 **ملاحظة:** يوضح الرسم التباين في أعداد المتبرعين. عادةً ما تكون نسبة الذكور أعلى في حملات التبرع لأسباب طبية (مثل الحمل والرضاعة لدى النساء) أو اجتماعية.")

# --- الصفحة 4: السنوات ---
elif selection == "4. اتجاهات التبرع السنوية":
    st.header("📅 التبرعات عبر السنوات")
    if 'Donation_Year' in df_viz.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        donation_per_year = df_viz.groupby('Donation_Year').size()
        donation_per_year.plot(kind='line', marker='o', color='green', ax=ax)
        plt.grid(True)
        ax.set_ylabel("إجمالي التبرعات")
        st.pyplot(fig)
        
        st.info("💡 **ملاحظة:** يساعد هذا الرسم في تتبع نمو الوعي بالتبرع بالدم. الاتجاه التصاعدي يشير إلى نجاح الحملات التوعوية.")
    else:
        st.error("بيانات السنوات غير متوفرة.")

# --- الصفحة 5: التبرعات حسب الجنس والسنة ---
elif selection == "5. التبرعات حسب الجنس (سنوياً)":
    st.header("📊 التبرعات السنوية حسب الجنس")
    
    if 'Donation_Year' in df_viz.columns and 'Gender' in df_viz.columns:
        gender_year = df_viz.groupby("Donation_Year")['Gender'].value_counts().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        gender_year.plot(kind='bar', ax=ax)
        ax.set_title("التبرعات لكل سنة حسب الجنس")
        ax.set_xlabel("السنة")
        ax.set_ylabel("إجمالي التبرعات")
        plt.xticks(rotation=0)
        plt.grid(axis='y')
        st.pyplot(fig)

        st.info("💡 **ملاحظة:** يتيح هذا الرسم مقارنة مساهمة كل جنس عبر الزمن. يمكن استخدامه لمعرفة ما إذا كانت الفجوة بين الجنسين تتقلص أم تزداد مع مرور السنوات.")
    else:
        st.error("الأعمدة المطلوبة مفقودة.")

# --- الصفحة 6: الهيموجلوبين ---
elif selection == "6. متوسط مستويات الهيموجلوبين":
    st.header("🧪 متوسط الهيموجلوبين حسب الجنس")
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_hb = df_viz.groupby('Gender')['Hemoglobin_g_dL'].mean()
    mean_hb.plot(kind='bar', color=['#2E8B57', '#FFA07A'], edgecolor='black', ax=ax)
    ax.set_ylabel("الهيموجلوبين (g/dL)")
    st.pyplot(fig)

    st.info("""
    💡 **معلومة طبية:** * المعدل الطبيعي للرجال: **13.8 - 17.2 g/dL**
    * المعدل الطبيعي للنساء: **12.1 - 15.1 g/dL**
    يظهر الرسم تماشي البيانات مع الحقائق البيولوجية حيث يكون المتوسط لدى الذكور أعلى قليلاً.
    """)

# --- الصفحة 7: الوزن مقابل الهيموجلوبين ---
elif selection == "7. العلاقة بين الوزن والهيموجلوبين":
    st.header("⚖️ الوزن مقابل الهيموجلوبين")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_viz, x='Weight_kg', y='Hemoglobin_g_dL', hue='Gender', palette='Set1', ax=ax)
    st.pyplot(fig)

    st.info("💡 **ملاحظة:** الوزن ومستوى الهيموجلوبين هما من أهم معايير قبول المتبرع. يوضح الرسم توزع المتبرعين، حيث يُرفض عادةً من هم أقل من 50 كجم أو لديهم هيموجلوبين منخفض (أنيميا).")

# --- الصفحة 8: المدن ---
elif selection == "8. التحليل الجغرافي (المدن)":
    st.header("🏙️ أكثر المدن نشاطاً في التبرع")
    fig, ax = plt.subplots(figsize=(10, 5))
    top_cities = df_viz['City'].value_counts().head(5)
    top_cities.plot(kind='bar', color='#4C72B0', edgecolor='black', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.info("💡 **فائدة:** معرفة المدن الأكثر نشاطاً تساعد بنوك الدم في تحسين لوجستيات النقل وتوجيه حملات التبرع للمناطق الأقل نشاطاً.")

# --- الصفحة 9: توزيع الأعمار ---
elif selection == "9. توزيع الأعمار (Histogram)":
    st.header("🎂 التوزيع العمري للمتبرعين")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_viz['Age'], bins=10, color='skyblue', edgecolor='black')
    ax.set_title('توزيع أعمار المتبرعين')
    ax.set_xlabel('العمر')
    ax.set_ylabel('عدد المتبرعين')
    st.pyplot(fig)

    st.info("💡 **ملاحظة:** يوضح الرسم الفئات العمرية الغالبة. غالباً ما تكون الفئة الشبابية (20-40) هي الأكثر نشاطاً، بينما يقل التبرع مع التقدم في العمر لأسباب صحية.")

# --- الصفحة 10: العمر حسب فصيلة الدم ---
elif selection == "10. العمر حسب فصيلة الدم (Boxplot)":
    st.header("🩸 توزيع العمر حسب فصيلة الدم")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df_viz, x='Blood_Group', y='Age', palette='Set2', ax=ax)
    ax.set_title('توزيع العمر لكل فصيلة دم')
    st.pyplot(fig)

    st.info("💡 **ملاحظة:** يبين هذا الرسم (Boxplot) المتوسط والمدى العمري لكل فصيلة. يساعد في التأكد من أن جميع الفصائل ممثلة بشكل متوازن عبر مختلف الأعمار.")

# --- الصفحة 11: تدريب النموذج ---
elif selection == "11. تدريب نموذج التوقع":
    st.header("🤖 تدريب نموذج التعلم الآلي")
    
    df_ml = df_clean.copy()
    
    # ترميز البيانات (Encoding)
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

    if st.button("بدء التدريب 🚀"):
        if model_choice == "Random Forest": model = RandomForestClassifier()
        elif model_choice == "Decision Tree": model = DecisionTreeClassifier()
        elif model_choice == "Logistic Regression": model = LogisticRegression()
        elif model_choice == "KNN": model = KNeighborsClassifier()
        else: model = SVC()

        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        st.session_state['model'] = model
        st.success(f"✅ تم تدريب نموذج **{model_choice}** بنجاح! الدقة: {acc*100:.2f}%")

        st.subheader("مصفوفة الارتباط (Correlation Matrix)")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_ml.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

# --- الصفحة 12: التوقع ---
elif selection == "12. فحص أهلية متبرع جديد":
    st.header("🩺 فحص الأهلية (توقع)")

    if 'model' not in st.session_state:
        st.warning("⚠️ يرجى تدريب النموذج أولاً من صفحة 'تدريب نموذج التوقع'.")
        st.stop()

    model = st.session_state['model']
    encoders = st.session_state['encoders']

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("العمر", 18, 65, 25)
        gender = st.selectbox("الجنس", ["Male", "Female"])
        weight = st.number_input("الوزن (kg)", 45.0, 150.0, 65.0)
    with c2:
        hb = st.number_input("الهيموجلوبين (g/dL)", 5.0, 20.0, 13.0)
        donations = st.number_input("عدد التبرعات السابقة", 0, 50, 0)
        bg = st.selectbox("فصيلة الدم", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])

    if st.button("فحص الأهلية"):
        try:
            g_enc = encoders['Gender'].transform([gender])[0]
            bg_enc = encoders['Blood_Group'].transform([bg])[0]
            input_data = np.array([[age, g_enc, weight, hb, donations, bg_enc]])
            pred = model.predict(input_data)
            res = encoders['Eligible_for_Donation'].inverse_transform(pred)[0]

            st.markdown("---")
            if str(res).lower() in ["yes", "1", "eligible", "true"]:
                st.success("✅ **مؤهل للتبرع:** هذا الشخص يستوفي الشروط بناءً على البيانات المدخلة.")
                st.balloons()
            else:
                st.error("❌ **غير مؤهل للتبرع:** نأسف، هذا الشخص لا يمكنه التبرع حالياً.")
                if hb < 12.5: st.warning("⚠️ السبب المحتمل: مستوى الهيموجلوبين منخفض.")
                if weight < 50: st.warning("⚠️ السبب المحتمل: الوزن أقل من الحد المسموح.")
        except:
            st.error("حدث خطأ في التوقع، تأكد من البيانات.")
