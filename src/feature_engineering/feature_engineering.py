import seaborn as sns
import matplotlib.pyplot as plt

def visualize_data(df):
    numeric_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']
    selected_columns = numeric_columns + ['diabetes']

    sns.pairplot(df[selected_columns], hue='diabetes', diag_kind='hist')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='gender', y='age', data=df, hue='diabetes')
    plt.title('Boxplot of Age by Gender and Diabetes')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.countplot(x='hypertension', data=df, hue='diabetes')
    plt.title('Countplot of Hypertension and Diabetes')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.countplot(x='heart_disease', data=df, hue='diabetes')
    plt.title('Countplot of Heart Disease and Diabetes')
    plt.show()

    selected_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes', 'smoking_history_Unknown', 'smoking_history_current', 'smoking_history_ever','smoking_history_former','smoking_history_never','smoking_history_not current']

    correlation_matrix = df[selected_features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

def visualize_age_smoking_diabetes(df):
    age_bins = list(range(0, 100, 10))
    df['age_group'] = pd.cut(df['age'], bins=age_bins, right=False, labels=[f'{i}-{i+9}' for i in range(0, 90, 10)])
    grouped_data = df.groupby(['age_group', 'smoking_history_former'])['diabetes'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='age_group', y='diabetes', hue='smoking_history_former', data=grouped_data, palette='Set1')
    plt.title('Average Diabetes Probability by Age Group and Smoking History (Former)')
    plt.xlabel('Age Group')
    plt.ylabel('Average Diabetes Probability')
    plt.xticks(rotation=45)
    plt.legend(title='Smoking History (Former)', loc='upper left')
    plt.show()

    df.drop(['age_group'], axis=1, inplace=True)
