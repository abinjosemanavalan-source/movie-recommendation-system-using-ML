"""
Movie Recommendation System
============================
Content-based filtering using TF-IDF + NearestNeighbors (cosine similarity).

Workflow:
  1. Load datasets (movies.csv, ratings.csv)
  2. Preprocess (extract year, build tags, aggregate ratings)
  3. Filter  (keep movies with rating_count >= 500)
  4. Feature engineering (TfidfVectorizer on genre tags)
  5. Train model (NearestNeighbors — cosine, brute)
  6. Recommend movies
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR    = r"C:/Users/abinj/OneDrive/Desktop/project/ML/moive bar"
MIN_RATINGS = 500      # minimum number of ratings a movie must have
TOP_N       = 10       # number of recommendations to display


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_dir: str) -> tuple:
    """
    Load movies.csv and ratings.csv from the given directory.
    Returns (movies_df, ratings_df) with only the required columns.
    """
    movies = pd.read_csv(
        os.path.join(data_dir, "movies.csv"),
        usecols=["movieId", "title", "genres"]
    )
    ratings = pd.read_csv(
        os.path.join(data_dir, "ratings.csv"),
        usecols=["movieId", "rating"]
    )
    return movies, ratings


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """
    - Fill missing genres
    - Extract release year from title
    - Create 'tags' column (genre string used as TF-IDF input)
    - Merge aggregated rating stats (avg_rating, rating_count)
    """
    df = movies.copy()

    df["genres"] = df["genres"].fillna("")

    # Extract 4-digit year, e.g. "Toy Story (1995)" -> "1995"
    df["year"] = df["title"].str.extract(r"\((\d{4})\)")

    # Pipe-separated genres used as feature tags
    df["tags"] = df["genres"]

    # Aggregate ratings per movie
    stats = (
        ratings
        .groupby("movieId", as_index=False)
        .agg(avg_rating=("rating", "mean"), rating_count=("rating", "count"))
    )
    stats["avg_rating"] = stats["avg_rating"].round(2)

    df = df.merge(stats, on="movieId", how="left")
    df["avg_rating"]   = df["avg_rating"].fillna(0.0)
    df["rating_count"] = df["rating_count"].fillna(0).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FILTER
# ─────────────────────────────────────────────────────────────────────────────

def filter_movies(df: pd.DataFrame, min_ratings: int = 500) -> pd.DataFrame:
    """
    Keep only movies with rating_count >= min_ratings.
    Index is reset so it stays contiguous and aligned with the TF-IDF matrix.
    """
    filtered = df[df["rating_count"] >= min_ratings].copy()
    filtered.reset_index(drop=True, inplace=True)
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — FEATURE ENGINEERING (TF-IDF)
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf(df: pd.DataFrame, tag_col: str = "tags"):
    """
    Fit TfidfVectorizer on the tags column of the *already filtered* DataFrame.
    Token pattern '[^|]+' treats each genre (e.g. 'Sci-Fi') as one token.

    Returns:
        tfidf_matrix : sparse matrix (n_movies x n_genres)
        vectorizer   : fitted TfidfVectorizer
    """
    vectorizer   = TfidfVectorizer(token_pattern=r"[^|]+")
    tfidf_matrix = vectorizer.fit_transform(df[tag_col])
    return tfidf_matrix, vectorizer


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — TRAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────

def train_model(tfidf_matrix, n_neighbors: int = 11) -> NearestNeighbors:
    """
    Train NearestNeighbors with cosine distance (brute-force).
    n_neighbors=11 so we can skip the query movie and return top 10.
    """
    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1
    )
    model.fit(tfidf_matrix)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — RECOMMENDATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def recommend_movies(
    title:        str,
    movies_df:    pd.DataFrame,
    tfidf_matrix,
    model:        NearestNeighbors,
    top_n:        int = 10
) -> None:
    """
    Print the top `top_n` movies most similar to the given title.

    Args:
        title        : Movie title to search for (case-insensitive, partial match OK).
        movies_df    : Filtered movies DataFrame — must be aligned with tfidf_matrix.
        tfidf_matrix : Sparse TF-IDF matrix fitted on movies_df.
        model        : Trained NearestNeighbors model.
        top_n        : Number of recommendations to return.
    """
    # Case-insensitive search
    mask    = movies_df["title"].str.contains(title, case=False, regex=False)
    matches = movies_df[mask]

    if matches.empty:
        print(f"\n[Not Found] No movie matching '{title}' in the dataset.")
        print("Tip: Try a partial title or check your spelling.\n")
        return

    movie_idx   = matches.index[0]
    movie_title = movies_df.loc[movie_idx, "title"]
    movie_vec   = tfidf_matrix[movie_idx]

    distances, indices = model.kneighbors(movie_vec, n_neighbors=top_n + 1)

    # First result is the query movie itself (distance ≈ 0) — skip it
    distances = distances[0][1:]
    indices   = indices[0][1:]

    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  Top {top_n} recommendations for: {movie_title}")
    print(sep)

    for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
        rec_title  = movies_df.loc[idx, "title"]
        avg_rating = movies_df.loc[idx, "avg_rating"]
        similarity = (1 - dist) * 100   # cosine similarity = 1 - cosine distance

        print(
            f"  {rank:>2}. {rec_title:<44}"
            f"| Similarity: {similarity:5.2f}%"
            f"| Rating: {avg_rating:.2f}"
        )

    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    movies_raw, ratings_raw = load_data(DATA_DIR)
    print(f"  movies : {len(movies_raw):,} rows")
    print(f"  ratings: {len(ratings_raw):,} rows")

    print("\nPreprocessing...")
    movies_full = preprocess(movies_raw, ratings_raw)

    print(f"\nFiltering (rating_count >= {MIN_RATINGS})...")
    movies_filtered = filter_movies(movies_full, min_ratings=MIN_RATINGS)
    print(f"  Movies remaining: {len(movies_filtered):,}")

    print("\nBuilding TF-IDF matrix...")
    tfidf_matrix, _ = build_tfidf(movies_filtered)
    print(f"  Matrix shape: {tfidf_matrix.shape}")

    print("\nTraining NearestNeighbors model...")
    model = train_model(tfidf_matrix, n_neighbors=TOP_N + 1)
    print("  Model ready.\n")

    # ── Example recommendations ──────────────────────────────────────────────
    for query in ["Toy Story", "Iron Man", "The Dark Knight", "Inception"]:
        recommend_movies(query, movies_filtered, tfidf_matrix, model, top_n=TOP_N)


if __name__ == "__main__":
    main()
