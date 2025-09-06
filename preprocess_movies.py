import pandas as pd
import zipfile
import ast

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

print(movies_df[['title', 'combined_features']].head())
