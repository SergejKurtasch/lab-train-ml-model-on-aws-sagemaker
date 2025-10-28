#!/usr/bin/env python3

import argparse
import joblib
import os
import pandas as pd
import numpy as np
import io
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import json

def model_fn(model_dir):
    """Load model from file"""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'text/csv':
        # For CSV data
        data = pd.read_csv(io.StringIO(request_body), header=None)
        return data.values
    elif request_content_type == 'application/json':
        # For JSON data
        data = json.loads(request_body)
        return np.array(data)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions using the model"""
    predictions = model.predict(input_data)
    return predictions

def output_fn(prediction, content_type):
    """Format output data"""
    if content_type == 'application/json':
        return json.dumps(prediction.tolist())
    elif content_type == 'text/csv':
        return ','.join(map(str, prediction))
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1.0)
    parser.add_argument('--max-depth', type=int, default=1)
    parser.add_argument('--random-state', type=int, default=42)
    
    # SageMaker arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    train_data = pd.read_csv(os.path.join(args.train, 'train.csv'), header=None)
    val_data = pd.read_csv(os.path.join(args.validation, 'validation.csv'), header=None)
    
    # Split into features and target variable
    X_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values
    X_val = val_data.iloc[:, 1:].values
    y_val = val_data.iloc[:, 0].values
    
    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    
    # Create and train model
    print("Training model...")
    base_estimator = DecisionTreeClassifier(max_depth=args.max_depth, random_state=args.random_state)
    model = AdaBoostClassifier(
        base_estimator=base_estimator,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        random_state=args.random_state
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    val_accuracy = accuracy_score(y_val, val_pred)
    
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Save model
    print("Saving model...")
    model_path = os.path.join(args.model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    
    print("Training completed successfully!")
