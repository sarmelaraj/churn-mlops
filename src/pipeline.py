import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    # Drop customerID - not useful for prediction
    df = df.drop(columns=['customerID'], errors='ignore')

    # Drop tenure group - derived column not needed
    df = df.drop(columns=['tenure group'], errors='ignore')

    # Convert TotalCharges to numeric - it loads as string
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill nulls in TotalCharges with column mean
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

    # Drop rows where tenure is 0 - invalid customer records
    df = df.drop(labels=df[df['tenure'] == 0].index, axis=0)

    # Convert Churn from Yes/No to 1/0
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    # Convert SeniorCitizen to int
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

    # Encode all categorical columns
    df = pd.get_dummies(df, drop_first=True)

    # Convert bool columns to int
    df = df.astype({col: int for col in df.select_dtypes(bool).columns})

    # Scale numerical features
    scaler = StandardScaler()
    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
        df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )

    # Save scaler for later use during prediction
    data_path = os.path.join(os.getcwd(), '..', 'data', 'scaler.pkl')
    joblib.dump(scaler, data_path)
    print(f"Scaler saved to: {data_path}")

    # Split into features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # Save split data to CSV
    X_train.to_csv(os.path.join(os.getcwd(), '..', 'data', 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(os.getcwd(), '..', 'data', 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(os.getcwd(), '..', 'data', 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(os.getcwd(), '..', 'data', 'y_test.csv'), index=False)
    print("Train and test data saved to /data")

    return X_train, X_test, y_train, y_test