# import libraries
import pandas as pd, scipy.stats as sps, seaborn as sns, matplotlib.pyplot as plt, statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Read Csv data file
df = pd.read_csv('/Users/sreeshareddy/Desktop/healthcare-dataset-stroke-data.csv')

# display number of rows and columns
print(df.shape)

# copy and get the statistics for numerical columns
summary = df.describe()
summary.to_clipboard()

# copy and get first 5 rows
summary = df.head()
summary.to_clipboard()

# Calculate skewness for numerical columns
Age = df['age']
Bmi = df['bmi']
Avg_glucose = df['avg_glucose_level']
# Print skewness values
print("Age Skewness:", sps.skew(Age))
print("Bmi Skewness:", sps.skew(Bmi))
print("Avg_glucose:", sps.skew(Avg_glucose))

# Check for NULL values
print(df.isnull().sum())

# count for values in categorical columns
sns.countplot(x='work_type', data=df)
plt.show()
# count for values in categorical columns
sns.countplot(x='Residence_type', data=df)
plt.show()
# count for values in categorical columns
sns.countplot(x='ever_married', data=df)
plt.show()
# count for values in categorical columns
sns.countplot(x='smoking_status', data=df)
plt.show()
# count for values in categorical columns
sns.countplot(x='gender', data=df)
plt.show()



# Drop 'Other' from 'gender' column
df.drop(df[df['gender'] == 'Other'].index, axis=0, inplace=True)
# Drop column smoking status and id
df = df.drop(['smoking_status','id'], axis=1)




# Define a custom mapping dictionary
gender_map = {'Male': 0, 'Female': 1}
married_map = {'No': 0, 'Yes': 1}
work_map = {'Never_worked': 0, 'children': 1, 'Govt_job': 2, 'Private': 3, 'Self-employed': 4}
residence_map = {'Rural': 0, 'Urban': 1}


# Apply custom mapping to the text columns
df['Gender'] = df['gender'].map(gender_map)
df['Married'] = df['ever_married'].map(married_map)
df['Work'] = df['work_type'].map(work_map)
df['Residence'] = df['Residence_type'].map(residence_map)
df.drop(['gender','ever_married','work_type', 'Residence_type'], axis=1, inplace=True)

# Drop Rows with NULL values
df1 = df.dropna(subset=['bmi'])
df1.to_csv('df1.csv', index=False)

# Check for outliers
# Generate boxplot and histplot for  bmi
sns.boxplot(x='bmi', data=df1, color='b')
plt.title("Boxplot of 'bmi'")
plt.show()
sns.histplot(df1, x='bmi', kde=True, color='b')
plt.title("Histplot of 'bmi'")
plt.show()
# Generate boxplot and histplot for age
sns.boxplot(x='age', data=df1, color='y')
plt.title("Boxplot of 'age'")
plt.show()
sns.histplot(df1, x='age', kde=True, color='y')
plt.title("Histplot of 'age'")
plt.show()
# Generate boxplot and histplot for average glucose level
sns.boxplot(x='avg_glucose_level', data=df1, color='r')
plt.title("Boxplot of 'avg_glucose_level'")
plt.show()
sns.histplot(df1, x='avg_glucose_level', kde=True, color='r')
plt.title("Histplot of 'avg_glucose_level'")
plt.show()




column_names = ['bmi', 'avg_glucose_level']

# Calculate the IQR for each column
for bmi in column_names:
    q1 = df1['bmi'].quantile(0.25)
    q3 = df1['bmi'].quantile(0.75)
    iqr = q3 - q1

# Identify outliers in each column
outliers = df1['bmi'][(df1['bmi'] < q1 - 1.5 * iqr) | (df1['bmi'] > q3 + 1.5 * iqr)]

# Remove outliers from both columns
df1 = df1[df1['bmi'].isin(outliers) == False]

# Generate boxplot and histplot after removing outliers
sns.boxplot(x='bmi', data=df1, color='c')
plt.show()
sns.histplot(df1, x='bmi', kde=True, color='c')
plt.show()

# Generate correlation matrix or heatmap
sns.heatmap(df1.corr(), annot=True)
plt.title("Heatmap of stroke Correlation")
plt.show()


# Select predictor (X) and target (y) variables
X_age = df1[["age"]]  # Predictor variable: age
X_hypertension = df1[["hypertension"]]  # Predictor variable: hypertension
X_heart_disease = df1[["heart_disease"]]  # predictor variable: heart_disease
X_Married = df1[["Married"]]  # Predictor variable: Married
X_avg_glucose_level = df1[["avg_glucose_level"]]  # Predictor variable: avg_glucose_level
X_Work = df1[["Work"]]  # Predictor variable: Work
X_bmi = df1[["bmi"]]  # Predictor variable: bmi
y = df1["stroke"]  # Target variable


# Split the data into training and testing sets
X_train_age, X_test_age, y_train, y_test = train_test_split(X_age, y, test_size=0.2, random_state=42)
X_train_hypertension, X_test_hypertension, _, _ = train_test_split(X_hypertension, y, test_size=0.2, random_state=42)
X_train_heart_disease, X_test_heart_disease, _, _ = train_test_split(X_heart_disease, y, test_size=0.2, random_state=42)
X_train_Married, X_test_Married, _, _ = train_test_split(X_Married, y, test_size=0.2, random_state=42)
X_train_avg_glucose_level, X_test_avg_glucose_level, _, _ = train_test_split(X_avg_glucose_level, y, test_size=0.2, random_state=42)
X_train_Work, X_test_Work, _, _ = train_test_split(X_Work, y, test_size=0.2, random_state=42)
X_train_bmi, X_test_bmi, _, _ = train_test_split(X_bmi, y, test_size=0.2, random_state=42)


# Initialize and fit logistic regression models using statsmodels


# Age vs. Stroke
X_train_age = sm.add_constant(X_train_age)
X_test_age = sm.add_constant(X_test_age)
logreg_model_age = sm.Logit(y_train, X_train_age)
logreg_result_age = logreg_model_age.fit()


# Hypertension vs. Stroke
X_train_hypertension = sm.add_constant(X_train_hypertension)
X_test_hypertension = sm.add_constant(X_test_hypertension)
logreg_model_hypertension = sm.Logit(y_train, X_train_hypertension)
logreg_result_hypertension = logreg_model_hypertension.fit()


# Heart disease vs. Stroke
X_train_heart_disease = sm.add_constant(X_train_heart_disease)
X_test_heart_disease = sm.add_constant(X_test_heart_disease)
logreg_model_heart_disease = sm.Logit(y_train, X_train_heart_disease)
logreg_result_heart_disease = logreg_model_heart_disease.fit()


# Ever_Married Vs. Stroke
X_train_Married = sm.add_constant(X_train_Married)
X_test_Married = sm.add_constant(X_test_Married)
logreg_model_Married = sm.Logit(y_train, X_train_Married)
logreg_result_Married = logreg_model_Married.fit()


# Avg Glucose Level vs. Stroke
X_train_avg_glucose_level = sm.add_constant(X_train_avg_glucose_level)
X_test_avg_glucose_level = sm.add_constant(X_test_avg_glucose_level)
logreg_model_avg_glucose_level = sm.Logit(y_train, X_train_avg_glucose_level)
logreg_result_avg_glucose_level = logreg_model_avg_glucose_level.fit()


# Work_type Vs. Stroke
X_train_Work = sm.add_constant(X_train_Work)
X_test_Work = sm.add_constant(X_test_Work)
logreg_model_Work = sm.Logit(y_train, X_train_Work)
logreg_result_Work = logreg_model_Work.fit()


# BMI vs. Stroke
X_train_bmi = sm.add_constant(X_train_bmi)
X_test_bmi = sm.add_constant(X_test_bmi)
logreg_model_bmi = sm.Logit(y_train, X_train_bmi)
logreg_result_bmi = logreg_model_bmi.fit()


# Display the summaries of logistic regression results
print("Logistic Regression Results for Age:")
print(logreg_result_age.summary())


print("Logistic Regression Results for Hypertension:")
print(logreg_result_hypertension.summary())


print("Logistic Regression Results for Heart Disease:")
print(logreg_result_heart_disease.summary())


print("Logistic Regression Results for Married:")
print(logreg_result_Married.summary())


print("Logistic Regression Results for Avg Glucose Level:")
print(logreg_result_avg_glucose_level.summary())


print("Logistic Regression Results for Work:")
print(logreg_result_Work.summary())


print("Logistic Regression Results for BMI:")
print(logreg_result_bmi.summary())


# Make predictions on the test sets
y_pred_age = (logreg_result_age.predict(X_test_age) > 0.5).astype(int)
y_pred_hypertension = (logreg_result_hypertension.predict(X_test_hypertension) > 0.5).astype(int)
y_pred_heart_disease = (logreg_result_heart_disease.predict(X_test_heart_disease) > 0.5).astype(int)
y_pred_Married = (logreg_result_Married.predict(X_test_Married) > 0.5).astype(int)
y_pred_avg_glucose_level = (logreg_result_avg_glucose_level.predict(X_test_avg_glucose_level) > 0.5).astype(int)
y_pred_Work = (logreg_result_Work.predict(X_test_Work) > 0.5).astype(int)
y_pred_bmi = (logreg_result_bmi.predict(X_test_bmi) > 0.5).astype(int)






# Visualize relationships using regplots


# Age vs. Stroke
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
sns.set(color_codes=True)
sns.regplot(x="age", y="stroke", data=df1, logistic=True, scatter_kws={"alpha": 0.3, "color":'b'}, color='orange', label='Age')
plt.title("Age vs. Stroke")
plt.legend(loc='upper right')


# Hypertension vs. Stroke
plt.subplot(2, 3, 2)
sns.set(color_codes=True)
sns.regplot(x="hypertension", y="stroke", data=df1, logistic=True, scatter_kws={"alpha": 0.3, "color":'b'}, color='orange', label='Hypertension')
plt.title("Hypertension vs. Stroke")
plt.legend(loc='upper right')


# Heart_disease vs. Stroke
plt.subplot(2, 3, 3)
sns.set(color_codes=True)
sns.regplot(x="heart_disease", y="stroke", data=df1, logistic=True, scatter_kws={"alpha": 0.3, "color":'b'}, color='orange', label='Heart Disease')
plt.title("Heart Disease vs. Stroke")
plt.legend(loc='upper right')




# Work Vs. Stroke
plt.subplot(2, 3, 4)
sns.set(color_codes=True)
sns.regplot(x="Work", y="stroke", data=df1, logistic=True, scatter_kws={"alpha": 0.3, "color":'b'}, color='orange', label='Work')
plt.title("Work vs. Stroke")
plt.legend(loc='upper right')


# Avg Glucose Level vs. Stroke
plt.subplot(2, 3, 5)
sns.set(color_codes=True)
sns.regplot(x="avg_glucose_level", y="stroke", data=df1, logistic=True, scatter_kws={"alpha": 0.3, "color":'b'}, color='orange', label='Glucose Level')
plt.title("Avg Glucose Level vs. Stroke")
plt.legend(loc='upper right')


# BMI vs. Stroke
plt.subplot(2, 3, 6)
sns.set(color_codes=True)
sns.regplot(x="bmi", y="stroke", data=df1, logistic=True, scatter_kws={"alpha": 0.3, "color":'b'}, color='orange', label='BMI')
plt.title("BMI vs. Stroke")
plt.legend(loc='upper right')



plt.tight_layout()
plt.show()
















