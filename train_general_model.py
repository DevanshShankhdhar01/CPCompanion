import os
import pandas as pd
import xgboost as xgb

def load_all_processed_data(processed_dir='processed'):
    all_data = []
    for filename in os.listdir(processed_dir):
        if filename.endswith('_processed_data.csv'):
            path = os.path.join(processed_dir, filename)
            try:
                df = pd.read_csv(path)
                if set(['solved_count', 'avg_rating', 'avg_time_taken', 'hard_ratio', 'future_rating']).issubset(df.columns):
                    all_data.append(df)
            except Exception as e:
                print(f"⚠️ Error reading {filename}: {e}")
    return pd.concat(all_data, ignore_index=True) if all_data else None

def train_and_save_model(df):
    X = df[['solved_count', 'avg_rating', 'avg_time_taken', 'hard_ratio']]
    y = df['future_rating']

    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)

    os.makedirs('models', exist_ok=True)
    model.save_model('models/general_model.json')
    print("✅ General model trained and saved to models/general_model.json")

if __name__ == '__main__':
    df = load_all_processed_data()
    if df is not None and len(df) >= 10:
        print(f"✅ Loaded {len(df)} user data rows for training")
        train_and_save_model(df)
    else:
        print("❌ Not enough data to train the general model.")
