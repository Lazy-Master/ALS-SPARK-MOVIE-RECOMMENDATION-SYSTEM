"""CLI entrypoint for running the movie recommender."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset import load_movies
from src.recommender import MovieRecommender


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Movie recommendation system CLI")
    parser.add_argument(
        "--data",
        default=str(Path("data") / "sample_movies.csv"),
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--title",
        required=True,
        help="Movie title to use as the recommendation seed.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of recommendations to return.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    movies = load_movies(args.data)
    recommender = MovieRecommender()
    recommender.fit(movies)
    results = recommender.recommend(args.title, top_n=args.top_n)

    print(f"Recommendations for '{args.title}':")
    for index, recommendation in enumerate(results, start=1):
        print(f"{index}. {recommendation.title} ({recommendation.score:.3f})")


if __name__ == "__main__":
    main()
