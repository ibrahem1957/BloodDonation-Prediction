import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# إعداد الصفحة
st.set_page_config(
    page_title="Blood Donation Analysis & Prediction",
    page_icon="🩸",
    layout="wide"
)

# --- 1. تحميل البيانات ---
@st.cache_data
def load_data():
    try:
        # تأكد أن ملف البيانات موجود في نفس المجلد
        df = pd.read_csv('blood_donation.csv')
        return df
    except FileNotFoundError:
        return None

# --- 2. معالجة البيانات وتدريب النموذج ---
def process_and_train(df):
    data = df.copy()
    
    # معالجة التواريخ
    if 'Last_Donation_Date' in data.columns:
        data['Last_Donation_Date'] = pd.to_datetime(data['Last_Donation_Date'], format='%d-%m-%Y', errors='coerce')
        data['Donation_Year'] = data['Last_Donation_Date'].dt.year

    # التشفير (Encoding) للموديل
    encoders = {}
    label_cols = ['Gender', 'Blood_Group', 'Eligible_for_Donation']
    
    for col in label_cols:
        le = LabelEncoder()
        data[f'{col}_Encoded'] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    # تحديد الخصائص والهدف
    feature_cols = ['Age', 'Gender_Encoded', 'Weight_kg', 'Hemoglobin_g_dL', 'Total_Donations', 'Blood_Group_Encoded']
    
    # حذف القيم المفقودة لضمان دقة التدريب
    data = data.dropna(subset=feature_cols)
    
    X = data[feature_cols]
    y = data['Eligible_for_Don
