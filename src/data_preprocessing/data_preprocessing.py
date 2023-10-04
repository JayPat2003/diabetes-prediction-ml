import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(filename):
    df = pd.read_csv(filename)
    return df

def handle_missing_values(df):
    no_info_count = df['smoking_history'].eq('No Info').sum()
    print("Count of 'No info' in Smoking history:", no_info_count)
    
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    df['smoking_history'].replace('No Info', 'Unknown', inplace=True)

    encoder = OneHotEncoder(sparse_output=False)
    smoking_encoded = encoder.fit_transform(df[['smoking_history']])
    categories = encoder.categories_[0]
    smoking_categories = [f'smoking_history_{category}' for category in categories]

    smoking_encoded_df = pd.DataFrame(smoking_encoded, columns=smoking_categories)
    df = pd.concat([df, smoking_encoded_df], axis=1)
    df.drop(['smoking_history'], axis=1, inplace=True)
    
    return df

def scale_features(df):
    scaler = StandardScaler()
    df[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']] = scaler.fit_transform(df[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']])
    
    return df

def split_data(df):
    X = df.drop(['diabetes'], axis=1)
    y = df['diabetes']
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test