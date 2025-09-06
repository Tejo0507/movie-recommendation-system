import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ratings_dict = {
    'Avatar':       [5, 4, 0, 0, 3],
    'The Dark Knight Rises': [0, 5, 4, 0, 1],
    'Inception':    [4, 0, 5, 3, 0],
    'Titanic':      [0, 0, 3, 5, 5],
    'Gladiator':    [1, 0, 2, 4, 4],
    'Harry Potter': [0, 3, 0, 5, 4]
}

ratings_df = pd.DataFrame(ratings_dict, index=['User1', 'User2', 'User3', 'User4', 'User5'])

user_similarity = cosine_similarity(ratings_df)
user_similarity_df = pd.DataFrame(user_similarity, index=ratings_df.index, columns=ratings_df.index)

def recommend_movies(user_id, ratings=ratings_df, similarity=user_similarity_df):
    if user_id not in ratings.index:
        return f"User '{user_id}' not found."
    
    sim_scores = similarity.loc[user_id].drop(user_id)
    sim_values = sim_scores.values
    other_users = sim_scores.index

    other_ratings = ratings.loc[other_users].T

    weighted_sum = np.dot(other_ratings, sim_values)

    sim_sum = sim_values.sum()

    pred_ratings = weighted_sum / sim_sum

    pred_ratings = pd.Series(pred_ratings, index=other_ratings.index)

    already_rated = ratings.loc[user_id][ratings.loc[user_id] > 0].index
    pred_ratings = pred_ratings.drop(already_rated, errors='ignore')

    top_recommended = pred_ratings.sort_values(ascending=False).head(5).index.tolist()

    return top_recommended

if __name__ == "__main__":
    user = 'User3'
    recommendations = recommend_movies(user)
    print(f"Top 5 movie recommendations for {user}:")
    for movie in recommendations:
        print(movie)
