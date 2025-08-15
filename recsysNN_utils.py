"""
Utilities for the content-based movie recommender project.

This module prepares item and user feature matrices from MovieLens data.
If the dataset is not found under ./data/ml-latest-small, it will generate
synthetic data so the pipeline remains runnable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class PreparedData:
    item_train: np.ndarray
    user_train: np.ndarray
    y_train: np.ndarray
    item_features: np.ndarray
    user_features: np.ndarray
    item_vecs: np.ndarray
    movie_dict: Dict[str, Any]
    user_to_genre: Dict[int, Dict[str, float]]


def _ensure_genre_columns(df: pd.DataFrame, all_genres: List[str]) -> pd.DataFrame:
    for g in all_genres:
        if g not in df.columns:
            df[g] = 0.0
    # Ensure consistent column ordering
    return df[all_genres]


def _load_movielens_small(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(data_dir / "movies.csv")
    ratings = pd.read_csv(data_dir / "ratings.csv")
    return movies, ratings


def _parse_item_features(movies: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    movies = movies.copy()
    movies["genres"] = movies["genres"].fillna("")
    # Build genre set
    genre_sets = movies["genres"].apply(lambda s: set([]) if s in ("", "(no genres listed)") else set(s.split("|")))
    all_genres = sorted(set.union(*genre_sets) if len(genre_sets) > 0 else set())
    # One-hot encode genres
    for g in all_genres:
        movies[g] = movies["genres"].apply(lambda s: 1.0 if g in s.split("|") else 0.0)
    item_features_df = movies[["movieId", *all_genres]].copy()
    return item_features_df, all_genres


def _compute_user_genre_preferences(ratings: pd.DataFrame, item_features_df: pd.DataFrame, genres: List[str]) -> Tuple[pd.DataFrame, Dict[int, Dict[str, float]]]:
    # Join ratings with item genre one-hot
    joined = ratings.merge(item_features_df, on="movieId", how="left")
    # For each user, compute mean rating per genre (weighted by presence)
    user_genre_vals = []
    for user_id, grp in joined.groupby("userId"):
        prefs = {}
        for g in genres:
            mask = grp[g] > 0.0
            if mask.any():
                prefs[g] = grp.loc[mask, "rating"].mean()
            else:
                prefs[g] = 0.0
        rating_count = float(len(grp))
        rating_mean = float(grp["rating"].mean()) if rating_count > 0 else 0.0
        prefs_row = {"userId": user_id, "rating_count": rating_count, "rating_mean": rating_mean, **prefs}
        user_genre_vals.append(prefs_row)
    user_features_df = pd.DataFrame(user_genre_vals).fillna(0.0)
    # Build user_to_genre mapping
    user_to_genre = {
        int(row["userId"]): {g: float(row[g]) for g in genres}
        for _, row in user_features_df.iterrows()
    }
    return user_features_df, user_to_genre


def _build_training_matrices(
    ratings: pd.DataFrame,
    item_features_df: pd.DataFrame,
    user_features_df: pd.DataFrame,
    genres: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Align indices for fast lookup
    item_feat_map = item_features_df.set_index("movieId")
    user_feat_map = user_features_df.set_index("userId")

    item_rows: List[np.ndarray] = []
    user_rows: List[np.ndarray] = []
    y_vals: List[float] = []

    for _, r in ratings.iterrows():
        movie_id = int(r["movieId"])
        user_id = int(r["userId"])
        rating = float(r["rating"])

        if movie_id not in item_feat_map.index or user_id not in user_feat_map.index:
            continue

        item_vec = item_feat_map.loc[movie_id, genres].to_numpy(dtype=float)
        user_row = user_feat_map.loc[user_id]
        user_vec = user_row[genres].to_numpy(dtype=float)
        rating_count = float(user_row["rating_count"]) if "rating_count" in user_row else 0.0
        rating_mean = float(user_row["rating_mean"]) if "rating_mean" in user_row else 0.0

        # Assemble with ID headers to match notebook slicing
        item_rows.append(np.concatenate([[movie_id], item_vec], dtype=float))
        user_rows.append(np.concatenate([[user_id, rating_count, rating_mean], user_vec], dtype=float))
        y_vals.append(rating)

    item_train = np.vstack(item_rows)
    user_train = np.vstack(user_rows)
    y_train = np.asarray(y_vals, dtype=float)
    return item_train, user_train, y_train


def _prepare_from_movielens(data_dir: Path) -> PreparedData:
    movies, ratings = _load_movielens_small(data_dir)
    item_features_df, genres = _parse_item_features(movies)
    user_features_df, user_to_genre = _compute_user_genre_preferences(ratings, item_features_df, genres)
    item_train, user_train, y_train = _build_training_matrices(ratings, item_features_df, user_features_df, genres)

    # item_features / item_vecs for similarity (pure genre one-hot, order = item_features_df rows)
    item_vecs = item_features_df[genres].to_numpy(dtype=float)
    item_features = item_vecs.copy()
    user_features = user_features_df[genres].to_numpy(dtype=float)

    movie_dict = {
        "id_to_title": {int(m["movieId"]): str(m["title"]) for _, m in movies.iterrows()},
        "id_to_index": {int(row["movieId"]): idx for idx, (_, row) in enumerate(item_features_df.iterrows())},
        "index_to_id": [int(row["movieId"]) for _, row in item_features_df.iterrows()],
        "genres": genres,
    }

    return PreparedData(
        item_train=item_train,
        user_train=user_train,
        y_train=y_train,
        item_features=item_features,
        user_features=user_features,
        item_vecs=item_vecs,
        movie_dict=movie_dict,
        user_to_genre=user_to_genre,
    )


def _prepare_synthetic(n_users: int = 50, n_items: int = 200, seed: int = 42) -> PreparedData:
    rng = np.random.default_rng(seed)
    genres = ["Action", "Comedy", "Drama", "Romance", "Sci-Fi", "Fantasy"]
    n_genres = len(genres)

    # Items: random multi-hot genres
    item_genres = (rng.random((n_items, n_genres)) > 0.6).astype(float)
    # Avoid all-zero rows
    empty_mask = item_genres.sum(axis=1) == 0
    item_genres[empty_mask, rng.integers(0, n_genres, size=empty_mask.sum())] = 1.0

    # Users: random preferences per genre; also rating_count/mean
    user_prefs = rng.normal(loc=3.5, scale=0.8, size=(n_users, n_genres))
    rating_counts = rng.integers(20, 150, size=(n_users, 1)).astype(float)
    rating_means = rng.normal(loc=3.5, scale=0.5, size=(n_users, 1))

    # Build datasets by sampling interactions
    interactions = []
    for user_id in range(1, n_users + 1):
        for _ in range(rng.integers(40, 120)):
            movie_id = int(rng.integers(1, n_items + 1))
            g = item_genres[movie_id - 1]
            base = np.dot(user_prefs[user_id - 1], g) / max(g.sum(), 1.0)
            rating = np.clip(base + rng.normal(0, 0.5), 0.5, 5.0)
            interactions.append((user_id, movie_id, float(rating)))

    interactions = np.array(interactions)
    user_ids = interactions[:, 0].astype(int)
    movie_ids = interactions[:, 1].astype(int)
    y_train = interactions[:, 2].astype(float)

    # Build feature rows
    item_train = np.concatenate([
        movie_ids.reshape(-1, 1).astype(float),
        item_genres[movie_ids - 1],
    ], axis=1)

    user_rows = []
    for uid in user_ids:
        user_rows.append(
            np.concatenate([
                np.array([uid, rating_counts[uid - 1, 0], rating_means[uid - 1, 0]], dtype=float),
                user_prefs[uid - 1],
            ])
        )
    user_train = np.vstack(user_rows)

    # Similarity matrices
    item_features = item_genres.copy()
    user_features = user_prefs.copy()

    movie_dict = {
        "id_to_title": {i + 1: f"Movie {i+1}" for i in range(n_items)},
        "id_to_index": {i + 1: i for i in range(n_items)},
        "index_to_id": [i + 1 for i in range(n_items)],
        "genres": genres,
    }

    user_to_genre = {uid: {g: float(user_prefs[uid - 1, gi]) for gi, g in enumerate(genres)} for uid in range(1, n_users + 1)}

    return PreparedData(
        item_train=item_train,
        user_train=user_train,
        y_train=y_train,
        item_features=item_features,
        user_features=user_features,
        item_vecs=item_features,
        movie_dict=movie_dict,
        user_to_genre=user_to_genre,
    )


def load_data() -> Tuple[Any, Any, Any, Any, Any, Any, Dict[str, Any], Dict[int, Any]]:
    """
    Load and prepare data for the content-based recommender.

    Returns a tuple matching the expected structure:
    (item_train, user_train, y_train, item_features, user_features,
     item_vecs, movie_dict, user_to_genre)

    - item_train: [n_interactions, 1 + n_item_features] first column is movieId
    - user_train: [n_interactions, 3 + n_user_features] first columns are [userId, rating_count, rating_mean]
    - y_train: ratings vector
    - item_features: per-item features (e.g., genre one-hot)
    - user_features: per-user features (same genre order as items)
    - item_vecs: same as item_features for similarity search
    - movie_dict: dict with mappings and metadata
    - user_to_genre: mapping of userId -> {genre: preference}
    """
    data_dir = Path("./data/ml-latest-small")
    try:
        if data_dir.exists():
            prepared = _prepare_from_movielens(data_dir)
        else:
            prepared = _prepare_synthetic()
    except Exception:
        # Fallback to synthetic if any parsing/IO fails
        prepared = _prepare_synthetic()

    return (
        prepared.item_train,
        prepared.user_train,
        prepared.y_train,
        prepared.item_features,
        prepared.user_features,
        prepared.item_vecs,
        prepared.movie_dict,
        prepared.user_to_genre,
    )

