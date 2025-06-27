# bulk_process.py

import os
import pandas as pd
import xgboost as xgb

def extract_features(df):
    # Drop 'future_rating' if present (it's the target)
    df = df.drop(columns=['future_rating'], errors='ignore')

    return pd.DataFrame([{
        'solved_count': df['solved_count'].values[0],
        'avg_rating': df['avg_rating'].values[0],
        'avg_time_taken': df['avg_time_taken'].values[0],
        'hard_ratio': df['hard_ratio'].values[0]
    }])

def load_target(df):
    return df['future_rating'].values[0]

def train_model_for_user(handle):
    processed_file = os.path.join('processed', f'{handle}_processed_data.csv')
    if not os.path.exists(processed_file):
        print(f"❌ Processed data not found for {handle}")
        return

    df = pd.read_csv(processed_file)
    if df.empty:
        print(f"❌ Empty data for {handle}")
        return

    try:
        X = extract_features(df)
        y = load_target(pd.read_csv(processed_file))

        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X, [y])

        os.makedirs('models', exist_ok=True)
        model_path = os.path.join('models', f'{handle}_rating_model.json')
        model.save_model(model_path)
        print(f"✅ Model trained and saved for {handle}")
    except Exception as e:
        print(f"❌ Failed to train model for {handle}: {e}")

if __name__ == '__main__':
    with open('user_handles.txt', 'r') as f:
        handles = [line.strip() for line in f.readlines()]

    for handle in handles:
        train_model_for_user(handle)
