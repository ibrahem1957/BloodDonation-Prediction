import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# إعداد الصفحة
st.set_page_config(page_title="تحليل بيانات التبرع بالدم", layout="wide")

# --- 1. تحميل البيانات ---
@st.cache_data
def load_data():
    try:
        # تأكد أن ملف csv موجود في نفس المجلد مع هذا الملف
        df = pd.read_csv('blood_donation.csv')
        
        # معالجة التواريخ إذا لزم الأمر لإنشاء عمود السنة
        if 'Last_Donation_Date' in df.columns:
            df['Last_Donation_Date'] = pd.to_datetime(df['Last_Donation_Date'], format='%d-%m-%Y', errors='coerce')
            df['Donation_Year'] = df['Last_Donation_Date'].dt.year
            
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is not None:
    st.title("📊 رسومات بيانية لتحليل المتبرعين")

    # --- الرسم الأول: عدد المتبرعين حسب فصيلة الدم ---
    st.subheader("1. عدد المتبرعين حسب فصيلة الدم")
    fig1 = plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Blood_Group', palette=['#440154', '#31688E', '#35B779', '#FDE725', '#F46D43', '#5C4D7D', '#C51B7D', '#FDE0A4'])
    plt.title('Number of Donors by Blood Group')
    plt.xlabel('Blood Group')
    plt.ylabel('Number of Donors')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig1) # هذا الأمر يعرض الرسم في ستريم لت

    # --- الرسم الثاني: توزيع المتبرعين حسب الجنس ---
    st.subheader("2. توزيع المتبرعين حسب الجنس")
    donor_counts = df.groupby('Gender')['Blood_Group'].count()
    fig2 = plt.figure(figsize=(6, 6))
    plt.pie(donor_counts, labels=donor_counts.index, autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
    plt.title('Donor Distribution by Gender')
    st.pyplot(fig2)

    # --- الرسم الثالث: عدد التبرعات لكل سنة ---
    if 'Donation_Year' in df.columns:
        st.subheader("3. عدد التبرعات لكل سنة")
        donation_per_year = df.groupby('Donation_Year')['Blood_Group'].count()
        fig3 = plt.figure(figsize=(8, 5))
        donation_per_year.plot(kind='line', marker='o', color='skyblue')
        plt.title('Number of Donations per Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Donations')
        plt.xticks(donation_per_year.index)
        plt.grid(True)
        st.pyplot(fig3)

        # --- الرسم الرابع: التبرعات لكل سنة حسب الجنس ---
        st.subheader("4. التبرعات لكل سنة حسب الجنس")
        gender_year = df.groupby("Donation_Year")['Gender'].value_counts().unstack(fill_value=0)
        fig4 = plt.figure(figsize=(8, 5))
        # ملاحظة: استخدام plot الخاص بـ pandas مع st.pyplot يتطلب التعامل مع الـ axes
        # الطريقة الأسهل هنا هي استخدام الـ figure الحالي
        gender_year.plot(kind='bar', figsize=(8, 5), ax=plt.gca())
        plt.title("Donations per Year by Gender (Stacked)")
        plt.xlabel("Year")
        plt.ylabel("Total Donations")
        plt.xticks(rotation=0)
        plt.grid(axis='y')
        st.pyplot(plt.gcf()) # gcf = Get Current Figure

    # --- الرسم الخامس: متوسط الهيموجلوبين حسب الجنس ---
    st.subheader("5. متوسط الهيموجلوبين حسب الجنس")
    mean_hb = df.groupby('Gender')['Hemoglobin_g_dL'].mean()
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    mean_hb.plot(kind='bar', color=['#2E8B57', '#FFA07A'], edgecolor='black', ax=ax5)
    
    for p in ax5.patches:
        height = p.get_height()
        ax5.text(p.get_x() + p.get_width()/2, height + 0.1, f'{height:.2f}', 
                ha='center', va='bottom', fontsize=12, color='black')

    plt.title('Average Hemoglobin by Gender')
    plt.ylabel('Hemoglobin (g/dL)')
    plt.xlabel('Gender')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig5)

    # --- الرسم السادس: أعلى 5 مدن ---
    st.subheader("6. أعلى 5 مدن من حيث عدد المتبرعين")
    top_cities = df.groupby("City")['Blood_Group'].count().sort_values(ascending=False).head()
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    top_cities.plot(kind='bar', color=['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974'], edgecolor='black', ax=ax6)
    plt.title('Top 5 Cities by Number of Donors')
    plt.xlabel('City')
    plt.ylabel('Number of Donors')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig6)

    # --- الرسم السابع: توزيع الأعمار ---
    st.subheader("7. توزيع أعمار المتبرعين")
    fig7 = plt.figure(figsize=(8, 5))
    plt.hist(df['Age'], bins=10, color='skyblue', edgecolor='black')
    plt.title('Distribution of Donors Age')
    plt.xlabel('Age')
    plt.ylabel('Number of Donors')
    st.pyplot(fig7)

    # --- الرسم الثامن: توزيع العمر حسب فصيلة الدم ---
    st.subheader("8. توزيع العمر حسب فصيلة الدم")
    fig8 = plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Blood_Group', y='Age', palette='Set2')
    plt.title('Age Distribution by Blood Group')
    st.pyplot(fig8)

    # --- الرسم التاسع: العلاقة بين الوزن والهيموجلوبين ---
    st.subheader("9. العلاقة بين الوزن والهيموجلوبين")
    fig9 = plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Weight_kg', y='Hemoglobin_g_dL', hue='Gender', palette='Set1')
    st.pyplot(fig9)

    # --- الرسم العاشر: مصفوفة الارتباط ---
    st.subheader("10. مصفوفة الارتباط (Correlation Heatmap)")
    fig10 = plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True)
    st.pyplot(fig10)

else:
    st.error("لم يتم العثور على ملف البيانات 'blood_donation.csv'. يرجى التأكد من رفعه بجانب ملف الكود.")
