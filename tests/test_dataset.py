"""Tests for dataset loading and normalization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.dataset import load_movies


class LoadMoviesTests(unittest.TestCase):
    def test_load_movies_rejects_missing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "invalid.csv"
            csv_path.write_text("title,genres\nMovie,Drama\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_movies(csv_path)

    def test_load_movies_drops_blank_titles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "movies.csv"
            csv_path.write_text(
                "title,genres,keywords,overview\n"
                "Movie A,Drama,hero,Story\n"
                ",Sci-Fi,space,Adventure\n",
                encoding="utf-8",
            )

            movies = load_movies(csv_path)
            self.assertEqual(list(movies["title"]), ["Movie A"])


if __name__ == "__main__":
    unittest.main()
