import json
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(file_path, text_source):
    with open(file_path, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)

    # Filter out unwanted classes
    df = df.loc[~df['tone'].isin(['incoherent', 'dontknow'])].copy()

    # Change any label that isn't "complied" to "rejected"
    df.loc[~df['tone'].isin(['complied', 'rejected']), 'tone'] = 'rejected'

    X = df[text_source].tolist()
    y = df['tone'].tolist()

    return X, y

def split_data(X, y):
    random_state = 0  # use a fixed random state of 0 for reproducibility

    # This yields a 70/15/15 train/validation/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
