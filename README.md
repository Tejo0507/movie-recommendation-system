# movie-recommendation-system
I built a movie recommendation system in Python that suggests movies based on user preferences. I used data preprocessing, TF-IDF vectorization, cosine similarity to find similar movies or users, and applied both content-based and collaborative filtering to generate recommendations.

Project Structure:

load_movies.py:
 - This script loads the movie and credits data from the dataset file. It helps to quickly check and preview the raw data before any processing.

preprocess_movies.py:
 - Here, I clean up and preprocess the loaded movie and credits data. This includes handling missing values and making sure the data is ready for feature extraction and analysis.

content_recommender.py:
 - In this script, I built a content-based movie recommender. It uses features like genres, cast, and keywords, and applies TF-IDF vectorization and cosine similarity to find and recommend movies most similar to a given movie.

collaborative_recommender.py:
 - This script implements collaborative filtering. It looks at user ratings and identifies users who have similar tastes. Based on that, it recommends movies that are liked by other users with similar preferences.

Dataset:
You need to download the dataset yourself from the Kaggle TMDB movie metadata page:
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

How to proceed:
Make sure you have Python 3.x, and then run:
pip install pandas numpy scikit-learn

Run the Scripts:
 - To load and preview data:
   python load_movies.py
 - To preprocess the data:
   python preprocess_movies.py
 - To get content-based movie recommendations:
   python content_recommender.py
 - To get collaborative filtering recommendations:
   python collaborative_recommender.py
