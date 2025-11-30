import matplotlib.pyplot as plt
import seaborn as sns

# 1. رسم عدد المتبرعين حسب فصيلة الدم (Countplot)
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Blood_Group', palette=['#440154', '#31688E', '#35B779', '#FDE725', '#F46D43', '#5C4D7D', '#C51B7D', '#FDE0A4'])
plt.title('Number of Donors by Blood Group')
plt.xlabel('Blood Group')
plt.ylabel('Number of Donors')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# 2. توزيع المتبرعين حسب الجنس (Pie Chart)
donor_counts = df.groupby('Gender')['Blood_Group'].count()
plt.figure(figsize=(6,6))
plt.pie(donor_counts, labels=donor_counts.index, autopct='%1.1f%%', colors=['skyblue','lightcoral'])
plt.title('Donor Distribution by Gender')
plt.show()

# 3. عدد التبرعات لكل سنة (Line Plot)
donation_per_year = df.groupby('Donation_Year')['Blood_Group'].count()
plt.figure(figsize=(8,5))
donation_per_year.plot(kind='line', marker='o', color='skyblue')
plt.title('Number of Donations per Year')
plt.xlabel('Year')
plt.ylabel('Number of Donations')
plt.xticks(donation_per_year.index)
plt.grid(True)
plt.show()

# 4. التبرعات لكل سنة حسب الجنس (Stacked Bar Plot)
gender_year = df.groupby("Donation_Year")['Gender'].value_counts().unstack(fill_value=0)
gender_year.plot(kind='bar', figsize=(8,5))
plt.title("Donations per Year by Gender (Stacked)")
plt.xlabel("Year")
plt.ylabel("Total Donations")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

# 5. متوسط الهيموجلوبين حسب الجنس (Bar Plot with Text)
mean_hb = df.groupby('Gender')['Hemoglobin_g_dL'].mean()
ax = mean_hb.plot(kind='bar', color=['#2E8B57', '#FFA07A'], edgecolor='black', figsize=(8,5)) # Added figsize for consistency
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2, height + 0.1, f'{height:.2f}',
            ha='center', va='bottom', fontsize=12, color='black')
plt.title('Average Hemoglobin by Gender')
plt.ylabel('Hemoglobin (g/dL)')
plt.xlabel('Gender')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# 6. أعلى 5 مدن من حيث عدد المتبرعين (Bar Plot)
top_cities = df.groupby("City")['Blood_Group'].count().sort_values(ascending=False).head()
ax = top_cities.plot(kind='bar', color=['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974'], edgecolor='black', figsize=(8,5))
plt.title('Top 5 Cities by Number of Donors')
plt.xlabel('City')
plt.ylabel('Number of Donors')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# 7. توزيع أعمار المتبرعين (Histogram)
plt.figure(figsize=(8,5))
plt.hist(df['Age'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Donors Age')
plt.xlabel('Age')
plt.ylabel('Number of Donors')
plt.show()

# 8. توزيع العمر حسب فصيلة الدم (Boxplot)
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='Blood_Group', y='Age', palette='Set2')
plt.title('Age Distribution by Blood Group')
plt.show()

# 9. العلاقة بين الوزن والهيموجلوبين (Scatterplot)
plt.figure(figsize=(8,6)) # Added explicit figure for clarity
sns.scatterplot(data=df, x='Weight_kg', y='Hemoglobin_g_dL', hue='Gender', palette='Set1')
plt.show()

# 10. مصفوفة الارتباط (Heatmap)
plt.figure(figsize=(10,8)) # Added explicit figure size for readability
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()
