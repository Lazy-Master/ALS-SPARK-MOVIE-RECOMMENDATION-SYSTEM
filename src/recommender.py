"""Core recommendation logic for the movie recommendation system."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Recommendation:
    title: str
    score: float


class MovieRecommender:
    """A lightweight content-based recommender built on movie metadata."""

    def __init__(self) -> None:
        self._movies: pd.DataFrame | None = None
        self._similarity_matrix = None
        self._title_to_index: dict[str, int] = {}
        self._vectorizer = TfidfVectorizer(stop_words="english")

    def fit(self, movies: pd.DataFrame) -> None:
        """Build the similarity matrix from a normalized movie dataframe."""
        if movies.empty:
            raise ValueError("The movie dataset is empty.")

        prepared = movies.copy()
        prepared["combined_features"] = (
            prepared["genres"].fillna("")
            + " "
            + prepared["keywords"].fillna("")
            + " "
            + prepared["overview"].fillna("")
        ).str.lower()

        feature_matrix = self._vectorizer.fit_transform(prepared["combined_features"])
        self._similarity_matrix = cosine_similarity(feature_matrix)
        self._movies = prepared.reset_index(drop=True)
        self._title_to_index = {
            title.lower(): index
            for index, title in enumerate(self._movies["title"])
        }

    def recommend(self, title: str, top_n: int = 5) -> list[Recommendation]:
        """Return the closest matching movies for the supplied title."""
        if self._movies is None or self._similarity_matrix is None:
            raise RuntimeError("The recommender has not been fitted yet.")

        normalized_title = title.strip().lower()
        if normalized_title not in self._title_to_index:
            raise KeyError(f"Movie '{title}' was not found in the dataset.")

        movie_index = self._title_to_index[normalized_title]
        similarity_scores = list(enumerate(self._similarity_matrix[movie_index]))
        ranked_scores = sorted(similarity_scores, key=lambda item: item[1], reverse=True)

        recommendations: list[Recommendation] = []
        for candidate_index, score in ranked_scores:
            if candidate_index == movie_index:
                continue
            candidate_title = self._movies.iloc[candidate_index]["title"]
            recommendations.append(Recommendation(title=candidate_title, score=float(score)))
            if len(recommendations) == top_n:
                break

        return recommendations
