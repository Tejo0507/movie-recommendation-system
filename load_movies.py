import pandas as pd
import zipfile

zip_file_path = r'E:\movie_recommendation_project\archive (1).zip'

with zipfile.ZipFile(zip_file_path, 'r') as archive:
    print("Files in zip:", archive.namelist())

with zipfile.ZipFile(zip_file_path) as archive:
    with archive.open('tmdb_5000_movies.csv') as movies_file:
        movies_df = pd.read_csv(movies_file)
    with archive.open('tmdb_5000_credits.csv') as credits_file:
        credits_df = pd.read_csv(credits_file)

print("Movies DataFrame Sample:")
print(movies_df.head())
print(movies_df.info())

print("\nCredits DataFrame Sample:")
print(credits_df.head())
print(credits_df.info())
