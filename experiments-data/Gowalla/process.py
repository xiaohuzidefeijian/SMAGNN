import pandas as pd

ratings = pd.read_csv('ratings.csv')

rating_counts = ratings['rating'].value_counts().sort_index()

print(rating_counts)
print(ratings['userId'].max(),ratings['movieId'].max())
print(ratings['userId'].min(),ratings['movieId'].min())

ratings['rating'] = ratings['rating'].apply(lambda x: -1 if x < 3.0 else 1)

ratings_str = ratings.apply(lambda row: f"{int(row['userId'])-1}\t{int(row['movieId'])-1}\t{row['rating']}", axis=1)
ratings_str.to_csv('ratings_raw.txt', index=False, header=False)

print("saved to ratings_raw.txt")
