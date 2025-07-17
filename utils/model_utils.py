import joblib
import pandas as pd

def build_temp_user_matrix(new_user_id, new_ratings):
    """
    Adds a new user (with their initial ratings) as a row to the loaded user-movie matrix.
    new_ratings: list of dicts, e.g. [{movieId: 123, rating: 4}, ...]
    Returns updated user_movie_matrix and new user's index.
    """
    user_movie_matrix = joblib.load('models/user_movie_matrix.joblib')
    all_movies = user_movie_matrix.columns

    # Build a new user row with zeros
    user_row = pd.Series(0, index=all_movies, dtype=float)
    for entry in new_ratings:
        if entry.movieId in all_movies:
            user_row[entry.movieId] = entry.rating
    # Add new user to matrix
    temp_matrix = user_movie_matrix.copy()
    temp_matrix.loc[new_user_id] = user_row
    return temp_matrix, temp_matrix.index.get_loc(new_user_id)


def get_hybrid_recommendations_for_new_user(
        new_user_id, new_ratings, movies_new, top_n=10
):
    temp_matrix, new_idx = build_temp_user_matrix(new_user_id, new_ratings)

    # COLLABORATIVE: KNN
    knn = joblib.load('models/knn.joblib')
    distances, indices = knn.kneighbors([temp_matrix.iloc[new_idx]], n_neighbors=top_n + 1)

    similar_users = temp_matrix.index[indices[0][1:]]
    similar_users_ratings = temp_matrix.loc[similar_users]
    user_seen = set([r.movieId for r in new_ratings])
    avg_ratings = similar_users_ratings.mean().sort_values(ascending=False)
    collab_recs = [mid for mid in avg_ratings.index if mid not in user_seen][:top_n]

    cosine_sim = joblib.load('models/cosine_sim.joblib')

    anchor = max(new_ratings, key=lambda x: x.rating)
    anchor_movie_id = anchor.movieId
    idx = movies_new[movies_new.movieId == anchor_movie_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1: top_n + 1]  # Exclude the movie itself
    content_indices = [i[0] for i in sim_scores]
    content_recs = list(movies_new.iloc[content_indices]['movieId'])
    content_recs = [mid for mid in content_recs if mid not in user_seen]

    hybrid_recs = []
    for rec in content_recs + collab_recs:
        if rec not in hybrid_recs:
            hybrid_recs.append(rec)
        if len(hybrid_recs) >= top_n:
            break
    return hybrid_recs

def get_custom_recommendations(new_user_id, new_ratings):

    movies_new = joblib.load('models/movies_new.joblib')


    hybrid_movie_ids = get_hybrid_recommendations_for_new_user(new_user_id, new_ratings, movies_new)
    recommendations = movies_new[movies_new['movieId'].isin(hybrid_movie_ids)]
    return recommendations

