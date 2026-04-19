"""Tests for recommendation behavior."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.dataset import load_movies
from src.recommender import MovieRecommender


class MovieRecommenderTests(unittest.TestCase):
    def setUp(self) -> None:
        dataset_path = Path(__file__).resolve().parents[1] / "data" / "sample_movies.csv"
        self.movies = load_movies(dataset_path)
        self.recommender = MovieRecommender()
        self.recommender.fit(self.movies)

    def test_recommend_returns_requested_number_of_items(self) -> None:
        results = self.recommender.recommend("Interstellar", top_n=3)
        self.assertEqual(len(results), 3)

    def test_recommend_omits_source_title(self) -> None:
        results = self.recommender.recommend("The Matrix", top_n=5)
        self.assertNotIn("The Matrix", [item.title for item in results])

    def test_recommend_raises_for_unknown_title(self) -> None:
        with self.assertRaises(KeyError):
            self.recommender.recommend("Unknown Movie")


if __name__ == "__main__":
    unittest.main()
