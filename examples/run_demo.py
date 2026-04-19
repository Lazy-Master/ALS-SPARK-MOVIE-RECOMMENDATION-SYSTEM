"""Small demonstration script for the movie recommender."""

from __future__ import annotations

from pathlib import Path

from src.dataset import load_movies
from src.recommender import MovieRecommender


def main() -> None:
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_movies.csv"
    movies = load_movies(data_path)

    recommender = MovieRecommender()
    recommender.fit(movies)

    seed_title = "Interstellar"
    print(f"Demo recommendations for '{seed_title}':")
    for recommendation in recommender.recommend(seed_title, top_n=3):
        print(f"- {recommendation.title} ({recommendation.score:.3f})")


if __name__ == "__main__":
    main()
