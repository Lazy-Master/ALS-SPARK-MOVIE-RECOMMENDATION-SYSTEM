# ALS Spark Movie Recommendation System

Scalable movie recommendation system using Apache Spark's MLlib with Alternating Least Squares (ALS) collaborative filtering. Demonstrates end-to-end pipeline from data loading through personalized recommendation generation.

## Overview

| Aspect | Details |
|--------|---------|
| Dataset | MovieLens 1M (1,000,209 ratings, 6,040 users, 3,883 movies) |
| Algorithm | ALS (Alternating Least Squares) Collaborative Filtering |
| Framework | Apache Spark 3.5.0, PySpark MLlib |
| Evaluation | RMSE, MAE, Precision@K, Recall@K |

## Techniques

### ALS Collaborative Filtering

Matrix factorization approach that decomposes the user-item rating matrix into two lower-rank matrices:
- **User factors** — latent feature representations for each user
- **Item factors** — latent feature representations for each movie

Key parameters:
- **Rank** — Number of latent factors (tested: 5, 10, 20, 50)
- **RegParam** — L2 regularization strength (tested: 0.05, 0.1, 0.2)
- **MaxIter** — Number of ALS iterations

### Pipeline

```
Load Data → StringIndexer → Train/Test Split → ALS Training → Evaluation → Recommendations
```

### Evaluation Metrics

- **RMSE** — Root Mean Squared Error for rating prediction accuracy
- **MAE** — Mean Absolute Error for average prediction deviation
- **Precision@K** — Fraction of recommended items that are relevant
- **Recall@K** — Fraction of relevant items that are recommended

### Hyperparameter Tuning

Grid search over rank and regularization parameters with RMSE as the optimization target.

### Cold Start Handling

Discusses strategies for new users and new movies:
- Popular movie recommendations for new users
- Content-based features for new movies
- Hybrid collaborative + content-based approaches

## Project Structure

```
ALS-SPARK-MOVIE-RECOMMENDATION-SYSTEM/
├── Movie_Recommender_System.ipynb   # Main notebook
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

```
pyspark==3.5.0
pandas
numpy
matplotlib
seaborn
plotly
```

## Usage

```bash
pip install -r requirements.txt
jupyter notebook Movie_Recommender_System.ipynb
```

## License

MIT
