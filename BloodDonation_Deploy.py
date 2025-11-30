import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# إعداد الصفحة
st.set_page_config(page_title="Blood Donation Analysis", layout="wide")

st.title("Blood Donation Analysis & Prediction")

# قراءة البيانات
df = pd.read_csv("data.csv")

# --------------------------
# واجهة التحليل والرسومات
# --------------------------
st.header("Data Analysis & Visualization")

st.subheader("Number of Donors by Blood Group")
fig1 = plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Blood_Group', palette=['#440154', '#31688E', '#35B779', '#FDE725', '#F46D43', '#5C4D7D', '#C51B7D', '#FDE0A4'])
plt.title('Number of Donors by Blood Group')
plt.xlabel('Blood Group')
plt.ylabel('Number of Donors')
plt.grid(axis='y', linestyle='--', alpha=0.5)
st.pyplot(fig1)

st.subheader("Donor Distribution by Gender")
donor_counts = df.groupby('Gender')['Blood_Group'].count()
fig2 = plt.figure(figsize=(6,6))
plt.pie(donor_counts, labels=donor_counts.index, autopct='%1.1f%%', colors=['skyblue','lightcoral'])
plt.title('Donor Distribution by Gender')
st.pyplot(fig2)

st.subheader("Number of Donations per Year")
donation_per_year = df.groupby('Donation_Year')['Blood_Group'].count()
fig3 = plt.figure(figsize=(8,5))
donation_per_year.plot(kind='line', marker='o', color='skyblue')
plt.title('Number of Donations per Year')
plt.xlabel('Year')
plt.ylabel('Number of Donations')
plt.xticks(donation_per_year.index)  
plt.grid(True)
st.pyplot(fig3)

st.subheader("Donations per Year by Gender (Stacked)")
gender_year = df.groupby("Donation_Year")['Gender'].value_counts().unstack(fill_value=0)
fig4 = plt.figure(figsize=(8,5))
gender_year.plot(kind='bar', figsize=(8,5))
plt.title("Donations per Year by Gender (Stacked)")
plt.xlabel("Year")
plt.ylabel("Total Donations")
plt.xticks(rotation=0)
plt.grid(axis='y')
st.pyplot(fig4)

st.subheader("Average Hemoglobin by Gender")
mean_hb = df.groupby('Gender')['Hemoglobin_g_dL'].mean()
fig5 = plt.figure(figsize=(6,5))
ax = mean_hb.plot(kind='bar', color=['#2E8B57', '#FFA07A'], edgecolor='black')
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2, height + 0.1, f'{height:.2f}', 
            ha='center', va='bottom', fontsize=12, color='black')
plt.title('Average Hemoglobin by Gender')
plt.ylabel('Hemoglobin (g/dL)')
plt.xlabel('Gender')
plt.grid(axis='y', linestyle='--', alpha=0.5)
st.pyplot(fig5)

st.subheader("Top 5 Cities by Number of Donors")
top_cities = df.groupby("City")['Blood_Group'].count().sort_values(ascending=False).head()
fig6 = plt.figure(figsize=(8,5))
ax = top_cities.plot(kind='bar', color=['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974'], edgecolor='black')
plt.title('Top 5 Cities by Number of Donors')
plt.xlabel('City')
plt.ylabel('Number of Donors')
plt.grid(axis='y', linestyle='--', alpha=0.5)
st.pyplot(fig6)

st.subheader("Distribution of Donors Age")
fig7 = plt.figure(figsize=(8,5))
plt.hist(df['Age'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Donors Age')
plt.xlabel('Age')
plt.ylabel('Number of Donors')
st.pyplot(fig7)

st.subheader("Age Distribution by Blood Group")
fig8 = plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='Blood_Group', y='Age', palette='Set2')
plt.title('Age Distribution by Blood Group')
st.pyplot(fig8)

st.subheader("Weight vs Hemoglobin by Gender")
fig9 = plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Weight_kg', y='Hemoglobin_g_dL', hue='Gender', palette='Set1')
st.pyplot(fig9)

st.subheader("Correlation Heatmap")
fig10 = plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True),annot=True)
st.pyplot(fig10)

# --------------------------
# Prediction Interface
# --------------------------
st.header("Predict Blood Donation Eligibility")

# ترميز الأعمدة
label_cols = ['Gender', 'Blood_Group', 'Eligible_for_Donation']
le_dict = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# النموذج
X = df[['Age', 'Gender', 'Weight_kg', 'Hemoglobin_g_dL', 'Total_Donations', 'Blood_Group']]
y = df['Eligible_for_Donation']
model = RandomForestClassifier()
model.fit(X, y)

# مدخلات المستخدم
age = st.number_input("Age", 18, 65, 30)
gender = st.selectbox("Gender", df['Gender'].unique())
blood_group = st.selectbox("Blood Group", df['Blood_Group'].unique())
weight = st.number_input("Weight (kg)", 40, 120, 70)
hb = st.number_input("Hemoglobin (g/dL)", 8.0, 20.0, 14.0)
donations = st.number_input("Total Donations", 0, 20, 2)

# تجهيز بيانات المستخدم
input_df = pd.DataFrame({
    'Age':[age],
    'Gender':[gender],
    'Weight_kg':[weight],
    'Hemoglobin_g_dL':[hb],
    'Total_Donations':[donations],
    'Blood_Group':[blood_group]
})

# الترميز
for col in ['Gender', 'Blood_Group']:
    input_df[col] = le_dict[col].transform(input_df[col].astype(str))

# التنبؤ
prediction = model.predict(input_df)[0]
result = "Eligible" if prediction == 1 else "Not Eligible"
st.subheader("Prediction Result")
st.write(result)
