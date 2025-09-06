import pandas as pd
import zipfile
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

zip_file_path = r'E:\movie_recommendation_project\archive (1).zip'

with zipfile.ZipFile(zip_file_path) as archive:
    with archive.open('tmdb_5000_movies.csv') as movies_file:
        movies_df = pd.read_csv(movies_file)
    with archive.open('tmdb_5000_credits.csv') as credits_file:
        credits_df = pd.read_csv(credits_file)

movies_df = movies_df[['id', 'title', 'genres', 'keywords', 'overview']]
credits_df = credits_df[['movie_id', 'cast']]

movies_df['overview'] = movies_df['overview'].fillna('')
movies_df['keywords'] = movies_df['keywords'].fillna('')
movies_df['genres'] = movies_df['genres'].fillna('')
credits_df['cast'] = credits_df['cast'].fillna('')

def parse_features(text):
    try:
        items = ast.literal_eval(text)
        if isinstance(items, list):
            return [item['name'].replace(" ", "").lower() for item in items]
    except:
        return []
    return []

movies_df['genres'] = movies_df['genres'].apply(parse_features)
movies_df['keywords'] = movies_df['keywords'].apply(parse_features)

def parse_cast(text):
    try:
        cast_list = ast.literal_eval(text)
        if isinstance(cast_list, list):
            cast_names = [member['name'].replace(" ", "").lower() for member in cast_list[:3]]
            return cast_names
    except:
        return []
    return []

credits_df['cast'] = credits_df['cast'].apply(parse_cast)

movies_df = movies_df.merge(credits_df, left_on='id', right_on='movie_id', how='left')

def combine_features(row):
    return ' '.join(row['genres']) + ' ' + ' '.join(row['keywords']) + ' ' + ' '.join(row['cast']) + ' ' + row['overview'].lower()

movies_df['combined_features'] = movies_df.apply(combine_features, axis=1)

# Vectorizing the combined features with TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])

# Computing cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies_df.index, index=movies_df['title'].str.lower())

# Recommendation function
def recommend_movies(title, cosine_sim=cosine_sim):
    title = title.lower()
    if title not in indices:
        return f"Movie '{title}' not found in database."
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Skip first as it's the movie itself
    
    movie_indices = [i[0] for i in sim_scores]
    recommended_titles = movies_df['title'].iloc[movie_indices].tolist()
    return recommended_titles

if __name__ == "__main__":
    movie_name = "The Dark Knight Rises"
    recommendations = recommend_movies(movie_name)
    print(f"Top 5 recommendations for '{movie_name}':")
    for rec in recommendations:
        print(rec)

