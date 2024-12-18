import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.lstm_model import LSTMModel

FEATURE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/features')


def create_sequences_for_classification(df, seq_length=30):
    data = df.copy().dropna()
    # We'll assume all columns except close are features
    feature_cols = [c for c in data.columns if c not in ['close']]
    # We need close as well for labeling
    all_cols = feature_cols + ['close']
    arr = data[all_cols].values

    X_list, y_list = [], []
    for i in range(len(arr) - seq_length - 1):
        X_list.append(arr[i:i + seq_length, :-1])  # features
        # Label: 1 if close[t+1] > close[t], else 0
        close_next = arr[i + seq_length, -1]
        close_current = arr[i + seq_length - 1, -1]
        label = 1.0 if close_next > close_current else 0.0
        y_list.append(label)

    X_arr = np.array(X_list)
    y_arr = np.array(y_list)
    return X_arr, y_arr


def main():
    files = [f for f in os.listdir(FEATURE_DATA_DIR) if f.endswith('_features.csv')]
    if not files:
        raise FileNotFoundError("No feature files found in data/features. Run feature_engineering.py first.")
    file_path = os.path.join(FEATURE_DATA_DIR, files[0])
    df = pd.read_csv(file_path, index_col='time', parse_dates=True)

    X, y = create_sequences_for_classification(df, seq_length=30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)

    input_size = X_train.shape[-1]  # number of features
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.1, learning_rate=1e-3)

    model.fit(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)

    y_pred = model.predict(X_test)
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred > 0.5).astype(float)

    accuracy = np.mean(y_pred_binary == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Confidence scores are simply y_pred (the probabilities)
    # Example: printing average confidence for predictions:
    avg_confidence = np.mean(y_pred)
    print(f"Average Prediction Confidence: {avg_confidence:.4f}")

    # Save model with versioning
    model_path = os.path.join(FEATURE_DATA_DIR, f'lstm_model_v1.pth')
    model.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
