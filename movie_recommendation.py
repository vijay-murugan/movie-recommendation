import kagglehub
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import joblib
import logging


def read_data():
    # Download latest version
    path = kagglehub.dataset_download("grouplens/movielens-20m-dataset")

    # Load the data
    movies = pd.read_csv(r'data\movie.csv')
    ratings = pd.read_csv(r'data\rating.csv')
    logging.info("Data loaded successfully.")
    return movies, ratings[:20000]


def clean_data(movies, ratings):
    logging.info("Cleaning data...")
    movies.fillna({'director': 'Unknown', 'genres': 'Unknown'}, inplace=True)
    ratings.dropna(inplace=True)

    # Remove duplicates
    ratings.drop_duplicates(subset=['userId', 'movieId'], inplace=True)

    # Standardize text columns
    movies['title'] = movies['title'].str.strip().str.lower()

    # Ensure referential integrity
    ratings = ratings[ratings['movieId'].isin(movies['movieId'])]

    # One-hot encode genres
    movies = movies.join(movies['genres'].str.get_dummies('|'))
    logging.info("Data cleaned.")
    return movies, ratings


def encode_movies(movies):
    logging.info("Encoding movies...")
    movies_new = movies[:10000].copy()  # Use a subset for faster processing
    # Create a DataFrame with the encoded values
    # Using sparse_output instead of sparse in newer scikit-learn versions
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    genre_encoded = encoder.fit_transform(movies_new[['genres']])

    # Now we can use get_feature_names_out
    genre_encoded_df = pd.DataFrame(
        genre_encoded,
        columns=encoder.get_feature_names_out(['genres']),
        index=movies_new.index
    )

    movies_encoded = pd.concat([movies_new, genre_encoded_df], axis=1)
    movies_encoded = movies_encoded.drop('genres', axis=1)
    joblib.dump(movies_encoded, 'models/movies_encoded.joblib')
    joblib.dump(movies_new, 'models/movies_new.joblib')
    logging.info("Movies encoded and saved.")
    return movies_encoded, movies_new

def calculate_similarity():
    logging.info("Calculating cosine similarity...")
    movies_encoded = joblib.load('models/movies_encoded.joblib')
    movies_numeric = movies_encoded.select_dtypes(include=['number'])
    cosine_sim = cosine_similarity(movies_numeric, movies_numeric)
    joblib.dump(cosine_sim, 'models/cosine_sim.joblib')
    logging.info("Cosine similarity calculated and saved.")
    return cosine_sim

def get_content_based_recommendations(movie_id, top_n=10):
    logging.info(f"Getting content-based recommendations for movie_id={movie_id}...")
    movies_new = joblib.load('models/movies_new.joblib')
    cosine_sim = joblib.load('models/cosine_sim.joblib')
    # Find the index of the movie in your DataFrame
    idx = movies_new[movies_new['movieId'] == movie_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Exclude the movie itself
    movie_indices = [i[0] for i in sim_scores]
    logging.info(f"Content-based recommendations generated for movie_id={movie_id}.")
    return list(movies_new.iloc[movie_indices]['movieId'])

def calculate_knn(ratings):
    user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    joblib.dump(user_movie_matrix, 'models/user_movie_matrix.joblib')

    # Fit KNN on users
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_movie_matrix)
    joblib.dump(knn, 'models/knn.joblib')
    logging.info("KNN model calculated and saved.")

def get_collaborative_recommendations(ratings, user_id, top_n=10):
    knn = joblib.load('models/knn.joblib')
    user_movie_matrix = joblib.load('models/user_movie_matrix.joblib')
    user_idx = user_movie_matrix.index.get_loc(user_id)
    distances, indices = knn.kneighbors([user_movie_matrix.iloc[user_idx]], n_neighbors=top_n+1)
    # Get similar users (excluding the user itself)
    similar_users = user_movie_matrix.index[indices[0][1:]]
    # Aggregate movies these users liked
    similar_users_ratings = user_movie_matrix.loc[similar_users]
    # Recommend movies the target user hasn't seen, sorted by average rating
    user_seen = set(user_movie_matrix.columns[user_movie_matrix.iloc[user_idx] > 0])
    avg_ratings = similar_users_ratings.mean().sort_values(ascending=False)
    final_recommends = [mid for mid in avg_ratings.index if mid not in user_seen][:top_n]
    logging.info(f"Collaborative recommendations generated for user_id={user_id}.")
    return final_recommends


def get_hybrid_recommendations(ratings, user_id, movie_id, top_n=10):
    content_recs = get_content_based_recommendations(movie_id, top_n)
    collab_recs = get_collaborative_recommendations(ratings, user_id, top_n)
    # Combine and keep unique movie IDs, preserving order and diversity
    hybrid_recs = []
    for rec in content_recs + collab_recs:
        if rec not in hybrid_recs:
            hybrid_recs.append(rec)
        if len(hybrid_recs) >= top_n:
            break
    logging.info(f"Hybrid recommendations generated for user_id={user_id}, movie_id={movie_id}.")
    return hybrid_recs


def main():
    logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")
    movies, ratings = read_data()
    movies, ratings = clean_data(movies, ratings)
    encode_movies(movies)
    calculate_similarity()
    calculate_knn(ratings)
    logging.info("model run successfully")


if __name__ == "__main__":
    main()




