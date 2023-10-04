import logging
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def setup_logging(log_dir='logs'):
    """Set up logging to save logs to a file."""
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, 'project.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO)

def load_data(filename):
    """Load data from a file (e.g., CSV)."""
    try:
        data = pd.read_csv(filename)
        return data
    except Exception as e:
        logging.error(f"Error loading data from {filename}: {str(e)}")
        return None

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

def save_model(model, filename):
    """Save a trained model to a file."""
    try:
        with open(filename, 'wb') as file:
            joblib.dump(model, file)
    except Exception as e:
        logging.error(f"Error saving model to {filename}: {str(e)}")

def load_model(filename):
    """Load a trained model from a file."""
    try:
        with open(filename, 'rb') as file:
            model = joblib.load(file)
        return model
    except Exception as e:
        logging.error(f"Error loading model from {filename}: {str(e)}")
        return None