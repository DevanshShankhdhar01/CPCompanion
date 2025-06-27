import os
import json
import requests
import pandas as pd
import xgboost as xgb
import time

def fetch_user_data(handle):
    # Create data folder if not exists
    os.makedirs('data', exist_ok=True)

    # Fetch rating
    rating_url = f"https://codeforces.com/api/user.rating?handle={handle}"
    rating_res = requests.get(rating_url)
    if rating_res.status_code != 200:
        print("❌ Failed to fetch rating.")
        return None, None
    with open(f"data/{handle}_rating.json", "w") as f:
        json.dump(rating_res.json(), f)

    # Fetch submissions
    sub_url = f"https://codeforces.com/api/user.status?handle={handle}&from=1&count=10000"
    sub_res = requests.get(sub_url)
    if sub_res.status_code != 200:
        print("❌ Failed to fetch submissions.")
        return None, None
    with open(f"data/{handle}_submissions.json", "w") as f:
        json.dump(sub_res.json(), f)

    time.sleep(0.3)  # avoid rate limit
    return rating_res.json(), sub_res.json()

def process_user(handle):
    with open(f"data/{handle}_rating.json") as f:
        rating_data = json.load(f)
    with open(f"data/{handle}_submissions.json") as f:
        submissions_data = json.load(f)

    problems = []
    timestamps = []

    for sub in submissions_data['result']:
        if sub['verdict'] == 'OK':
            prob = sub['problem']
            rating = prob.get('rating')
            if rating:
                problems.append({
                    'rating': rating,
                    'timestamp': sub['creationTimeSeconds']
                })
                timestamps.append(sub['creationTimeSeconds'])

    if not problems:
        print("❌ No rated solved problems found.")
        return None

    df = pd.DataFrame(problems)

    solved_count = len(df)
    avg_rating = df['rating'].mean()
    hard_ratio = (df['rating'] >= 1800).sum() / solved_count
    avg_time_taken = 0
    if len(timestamps) > 1:
        timestamps.sort()
        gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        avg_time_taken = sum(gaps) / len(gaps)

    features = pd.DataFrame([{
        'solved_count': solved_count,
        'avg_rating': avg_rating,
        'avg_time_taken': avg_time_taken,
        'hard_ratio': hard_ratio
    }])
    return features

def get_problemset():
    url = 'https://codeforces.com/api/problemset.problems'
    return requests.get(url).json()['result']['problems']

def extract_solved_set(handle):
    try:
        with open(os.path.join('data', f'{handle}_submissions.json'), 'r') as f:
            submissions = json.load(f)
        return {(sub['problem']['contestId'], sub['problem']['index'])
                for sub in submissions['result'] if sub['verdict'] == 'OK'}
    except:
        return set()

def recommend_problems_general(handle, model_path='models/general_model.json', top_n=10):
    print(f"📌 Generating recommendations for: {handle}")

    # Fetch and process user data
    fetch_user_data(handle)
    user_features = process_user(handle)
    if user_features is None:
        return

    # Load general model
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    # Predict
    predicted_rating = int(model.predict(user_features)[0])
    print(f"🔮 Predicted rating: {predicted_rating}")

    # Recommend problems
    solved = extract_solved_set(handle)
    problems = get_problemset()

    recommendations = []
    for prob in problems:
        rating = prob.get('rating')
        if not rating or abs(rating - predicted_rating) > 150:
            continue
        key = (prob['contestId'], prob['index'])
        if key not in solved:
            recommendations.append((rating, prob['name'], prob.get('tags', []), prob['contestId'], prob['index']))

    recommendations.sort()
    print("\n🧠 Recommended Problems:")
    for i, (rating, name, tags, contestId, index) in enumerate(recommendations[:top_n], 1):
        print(f"{i}. [{name}] - {rating} ({', '.join(tags)})")
        print(f"   🔗 https://codeforces.com/contest/{contestId}/problem/{index}")

# === MAIN ===
if __name__ == '__main__':
    user_handle = input("Enter Codeforces handle: ")
    recommend_problems_general(user_handle)
