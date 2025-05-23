import pandas as pd

dataset_name = 'ratings'

ratings = pd.read_csv(f'{dataset_name}.dat', sep='::', header=None, names=['user', 'item', 'rating', 'timestamp'])

rating_counts = ratings['rating'].value_counts().sort_index()
print(rating_counts)

ratings['rating'] = ratings['rating'].apply(lambda x: 1 if x > 3.0 else -1)

users = ratings['user'].unique()
items = ratings['item'].unique()
print(len(users), len(items))

user_mapping = {user: idx for idx, user in enumerate(users)}
item_mapping = {item: idx for idx, item in enumerate(items)}

ratings['user'] = ratings['user'].map(user_mapping)
ratings['item'] = ratings['item'].map(item_mapping)

ratings_str = ratings.apply(lambda row: f"{row['user']}\t{row['item']}\t{row['rating']}", axis=1)
ratings_str.to_csv(f'{dataset_name}_raw.txt', index=False, header=False)

print(f"saved to {dataset_name}_raw.txt")
