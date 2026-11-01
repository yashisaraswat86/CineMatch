
"""Leakage-aware offline evaluation for SVD, User-KNN, Item-KNN and hybrid recommenders.

Uses a deterministic per-user 80/20 split.
Ranking metrics use held-out relevant items (rating >= 4)
against 100 sampled unrated negatives.
"""

from pathlib import Path
import json
from itertools import product
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------
# 1. Paths and dataset
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ART = ROOT / "artifacts"

source = next(RAW.rglob("ratings.csv"), None)

if source is None:
    raise SystemExit("Run python scripts/download_data.py first")

r = pd.read_csv(source)[["userId", "movieId", "rating"]].dropna()
r = r.drop_duplicates(["userId", "movieId"])

rng = np.random.default_rng(42)


# --------------------------------------------------
# 2. Deterministic 80/20 train-test split
# --------------------------------------------------

train_parts = []
test_parts = []

for _, group in r.groupby("userId"):

    if len(group) < 5:
        train_parts.append(group)
        continue

    test_count = max(1, int(round(len(group) * 0.2)))

    chosen = rng.choice(
        group.index.to_numpy(),
        size=test_count,
        replace=False
    )

    test_parts.append(group.loc[chosen])
    train_parts.append(group.drop(chosen))


train = pd.concat(train_parts, ignore_index=True)
test = pd.concat(test_parts, ignore_index=True)

users = np.sort(train.userId.unique())
movies = np.sort(r.movieId.unique())

u2i = {int(x): i for i, x in enumerate(users)}
m2i = {int(x): i for i, x in enumerate(movies)}

train = train[
    train.userId.isin(u2i) &
    train.movieId.isin(m2i)
]


# --------------------------------------------------
# 3. Sparse user-item matrix
# --------------------------------------------------

mat = csr_matrix(
    (
        train.rating.astype("float32"),
        (
            train.userId.map(u2i),
            train.movieId.map(m2i)
        )
    ),
    shape=(len(users), len(movies))
)

counts = np.asarray((mat != 0).sum(axis=1)).ravel()

global_mean = float(train.rating.mean())

means = np.divide(
    np.asarray(mat.sum(axis=1)).ravel(),
    counts,
    out=np.full(len(users), global_mean),
    where=counts > 0
)


# --------------------------------------------------
# 4. SVD model
# --------------------------------------------------

coo = mat.tocoo()

centered = csr_matrix(
    (
        coo.data - means[coo.row],
        (coo.row, coo.col)
    ),
    shape=mat.shape
)

k = max(2, min(40, min(centered.shape) - 1))

svd = TruncatedSVD(
    n_components=k,
    n_iter=7,
    random_state=42
)

uf = svd.fit_transform(centered)
itf = svd.components_.T


# --------------------------------------------------
# 5. User and item nearest neighbours
# --------------------------------------------------

def fit_neighbors(matrix, k=40):

    n = matrix.shape[0]

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=min(k + 1, n),
        n_jobs=-1
    ).fit(matrix)

    distances, indices = model.kneighbors(matrix)

    similarities = np.maximum(0, 1 - distances)

    return indices, similarities


print("Fitting User-KNN neighbours...")

ui, us = fit_neighbors(mat, 40)

print("Fitting Item-KNN neighbours...")

ii, ins = fit_neighbors(mat.T.tocsr(), 40)


# --------------------------------------------------
# 6. Rating prediction
# --------------------------------------------------

actual = {
    name: [] for name in
    ["SVD", "User-KNN", "Item-KNN", "Hybrid"]
}

predicted = {
    name: [] for name in actual
}


for row in test.itertuples():

    if row.userId not in u2i or row.movieId not in m2i:
        continue

    u = u2i[row.userId]
    i = m2i[row.movieId]

    # ------------------------------
    # SVD prediction
    # ------------------------------

    svd_pred = float(
        np.clip(
            means[u] + uf[u] @ itf[i],
            0.5,
            5
        )
    )

    # ------------------------------
    # User-KNN prediction
    # Only retrieve ratings for target movie i
    # ------------------------------

    neigh = ui[u]
    weights = us[u]

    keep = neigh != u

    neigh = neigh[keep]
    weights = weights[keep]

    neighbor_ratings = mat[neigh, i].toarray().ravel()

    known_mask = neighbor_ratings != 0

    if np.any(known_mask):

        valid_weights = weights[known_mask]

        valid_deviations = (
            neighbor_ratings[known_mask]
            - means[neigh[known_mask]]
        )

        denominator = np.sum(valid_weights)

        numerator = np.sum(
            valid_weights * valid_deviations
        )

        user_pred = float(
            np.clip(
                means[u] +
                numerator / (denominator + 1e-9),
                0.5,
                5
            )
        )

    else:
        user_pred = global_mean

    # ------------------------------
    # Item-KNN prediction
    # ------------------------------

    known = mat.getrow(u).tocoo()

    item_weights = ins[known.col]
    item_ids = ii[known.col]

    mask = item_ids == i

    w = item_weights[mask]

    if len(w):

        source_ratings = np.broadcast_to(
            known.data[:, None],
            item_ids.shape
        )[mask]

        deviations = source_ratings - 3.0

        item_pred = float(
            np.clip(
                3.0 +
                np.sum(w * deviations) /
                (np.abs(w).sum() + 1e-9),
                0.5,
                5
            )
        )

    else:
        item_pred = global_mean

    # ------------------------------
    # Hybrid prediction
    # ------------------------------

    hybrid_pred = (
        0.50 * svd_pred +
        0.25 * user_pred +
        0.25 * item_pred
    )

    predictions = [
        ("SVD", svd_pred),
        ("User-KNN", user_pred),
        ("Item-KNN", item_pred),
        ("Hybrid", hybrid_pred)
    ]

    for name, value in predictions:

        actual[name].append(float(row.rating))
        predicted[name].append(value)


# --------------------------------------------------
# 7. RMSE and MAE
# --------------------------------------------------

metrics = {}

for name in actual:

    if not actual[name]:
        continue

    metrics[name] = {
        "RMSE": round(
            float(
                mean_squared_error(
                    actual[name],
                    predicted[name]
                ) ** 0.5
            ),
            5
        ),

        "MAE": round(
            float(
                mean_absolute_error(
                    actual[name],
                    predicted[name]
                )
            ),
            5
        ),

        "test_interactions": len(actual[name])
    }


# --------------------------------------------------
# 7A. Hybrid weight search for rating prediction
# --------------------------------------------------

# Search weights in 0.1 increments; all three weights must sum to 1.
WEIGHT_GRID = [
    (round(a / 10, 1), round(b / 10, 1), round((10 - a - b) / 10, 1))
    for a in range(11)
    for b in range(11 - a)
]

rating_weight_results = []

y_true = np.asarray(actual["SVD"], dtype=float)
svd_values = np.asarray(predicted["SVD"], dtype=float)
user_values = np.asarray(predicted["User-KNN"], dtype=float)
item_values = np.asarray(predicted["Item-KNN"], dtype=float)

for w_svd, w_user, w_item in WEIGHT_GRID:

    combined = np.clip(
        w_svd * svd_values +
        w_user * user_values +
        w_item * item_values,
        0.5,
        5.0
    )

    rating_weight_results.append({
        "weights": {
            "SVD": w_svd,
            "User-KNN": w_user,
            "Item-KNN": w_item
        },
        "RMSE": float(mean_squared_error(y_true, combined) ** 0.5),
        "MAE": float(mean_absolute_error(y_true, combined))
    })

best_rating_rmse = min(rating_weight_results, key=lambda x: x["RMSE"])
best_rating_mae = min(rating_weight_results, key=lambda x: x["MAE"])


# --------------------------------------------------
# 8. Ranking evaluation
# --------------------------------------------------

def ranking_for_user(
    user_id,
    heldout,
    negatives=100,
    top_k=10
):

    if user_id not in u2i:
        return None

    u = u2i[user_id]

    positives = [
        m2i[int(x.movieId)]
        for x in heldout.itertuples()
        if x.rating >= 4 and int(x.movieId) in m2i
    ]

    if not positives:
        return None

    seen = set(mat.getrow(u).indices.tolist())

    available = np.array(
        [
            i for i in range(len(movies))
            if i not in seen and i not in positives
        ],
        dtype=int
    )

    if len(available) > negatives:

        available = rng.choice(
            available,
            size=negatives,
            replace=False
        )

    candidates = np.unique(
        np.concatenate(
            [
                np.asarray(positives, dtype=int),
                available
            ]
        )
    )

    # ------------------------------
    # SVD scores
    # ------------------------------

    svd_scores = (
        means[u] +
        uf[u] @ itf[candidates].T
    )

    # ------------------------------
    # User-KNN scores
    # ------------------------------

    neigh = ui[u]
    weights = us[u]

    keep = neigh != u

    neigh = neigh[keep]
    weights = weights[keep]

    nr = mat[neigh][:, candidates].toarray()

    # Only ratings that neighbours have actually provided should contribute.
    known_mask = nr != 0

    # Deviation from each neighbour's own average rating.
    dev = nr - means[neigh, None]

    # Exclude missing ratings from both numerator and denominator.
    weighted_dev = weights[:, None] * dev * known_mask

    den = (
        weights[:, None] * known_mask
    ).sum(axis=0)

    num = weighted_dev.sum(axis=0)

    user_scores = means[u] + np.divide(
        num,
        den,
        out=np.zeros(len(candidates), dtype=float),
        where=den > 1e-9
    )

    # ------------------------------
    # Item-KNN scores
    # ------------------------------

    history = mat.getrow(u).tocoo()

    item_scores = np.full(
        len(candidates),
        global_mean,
        dtype=float
    )

    history_weights = ins[history.col]
    history_ids = ii[history.col]

    history_deviations = np.broadcast_to(
        history.data[:, None] - 3.0,
        history_ids.shape
    )

    for j, candidate in enumerate(candidates):

        loc = history_ids == candidate

        if loc.any():

            ww = history_weights[loc]
            deviations = history_deviations[loc]

            item_scores[j] = (
                3.0 +
                np.sum(ww * deviations) /
                (np.abs(ww).sum() + 1e-9)
            )

    # ------------------------------
    # Hybrid CF scores
    # ------------------------------

    hybrid = (
        0.50 * svd_scores +
        0.25 * user_scores +
        0.25 * item_scores
    )

    scores = {
        "SVD": svd_scores,
        "User-KNN": user_scores,
        "Item-KNN": item_scores,
        "Hybrid-CF": hybrid
    }

    relevant = set(positives)

    def ranking_metrics_for_scores(values):
        order = np.argsort(values)[::-1][:top_k]
        ranked_items = [int(candidates[x]) for x in order]
        hits = [item for item in ranked_items if item in relevant]

        precision = len(hits) / top_k
        recall = len(hits) / len(relevant)

        dcg = sum(
            1 / np.log2(rank + 2)
            for rank, item in enumerate(ranked_items)
            if item in relevant
        )

        ideal = sum(
            1 / np.log2(rank + 2)
            for rank in range(min(len(relevant), top_k))
        )

        ndcg = dcg / ideal if ideal else 0.0
        return precision, recall, ndcg

    out = {
        name: ranking_metrics_for_scores(values)
        for name, values in scores.items()
    }

    # Evaluate every weight combination on exactly the same candidate set.
    weight_out = {}

    for w_svd, w_user, w_item in WEIGHT_GRID:
        combined = (
            w_svd * svd_scores +
            w_user * user_scores +
            w_item * item_scores
        )

        key = (w_svd, w_user, w_item)
        weight_out[key] = ranking_metrics_for_scores(combined)

    return out, weight_out


# --------------------------------------------------
# 9. Aggregate ranking metrics
# --------------------------------------------------

rank_values = {
    name: []
    for name in
    ["SVD", "User-KNN", "Item-KNN", "Hybrid-CF"]
}

rank_weight_values = {weights: [] for weights in WEIGHT_GRID}

print("Calculating ranking metrics and searching hybrid weights...")

for uid, group in test.groupby("userId"):

    result = ranking_for_user(int(uid), group)

    if result:
        component_result, weight_result = result

        for name, values in component_result.items():
            rank_values[name].append(values)

        for weights, values in weight_result.items():
            rank_weight_values[weights].append(values)


def summarize_ranking(rows):
    arr = np.asarray(rows)
    return {
        "Precision@10": round(float(arr[:, 0].mean()), 5),
        "Recall@10": round(float(arr[:, 1].mean()), 5),
        "NDCG@10": round(float(arr[:, 2].mean()), 5),
        "evaluated_users": len(rows)
    }


ranking_metrics = {
    name: summarize_ranking(rows)
    for name, rows in rank_values.items()
    if rows
}

ranking_weight_results = []

for weights, rows in rank_weight_values.items():
    if rows:
        summary = summarize_ranking(rows)
        ranking_weight_results.append({
            "weights": {
                "SVD": weights[0],
                "User-KNN": weights[1],
                "Item-KNN": weights[2]
            },
            **summary
        })

best_ranking_ndcg = max(
    ranking_weight_results,
    key=lambda x: x["NDCG@10"]
)

best_ranking_precision = max(
    ranking_weight_results,
    key=lambda x: x["Precision@10"]
)

best_ranking_recall = max(
    ranking_weight_results,
    key=lambda x: x["Recall@10"]
)


# --------------------------------------------------
# 10. Save evaluation report
# --------------------------------------------------

report = {

    "dataset": source.parent.name,

    "split": "deterministic 80/20 per-user holdout",

    "seed": 42,

    "relevance_threshold": 4.0,

    "negative_samples_per_user": 100,

    "rating_prediction": metrics,

    "ranking": ranking_metrics,

    "weight_optimization": {
        "search": "all SVD/User-KNN/Item-KNN weight combinations in 0.1 increments summing to 1.0",
        "rating_best_RMSE": best_rating_rmse,
        "rating_best_MAE": best_rating_mae,
        "ranking_best_NDCG@10": best_ranking_ndcg,
        "ranking_best_Precision@10": best_ranking_precision,
        "ranking_best_Recall@10": best_ranking_recall
    },

    "notes": [
        "All collaborative models are fitted using training interactions only.",

        "Ranking metrics use sampled negatives and should be interpreted as sampled-ranking estimates.",

        "Weight search uses the same holdout set for model selection; selected weights are exploratory and require confirmation on a separate untouched test set.",

        "Hybrid-CF combines SVD, User-KNN and Item-KNN; content-based ranking is evaluated separately in the live personalized engine."
    ]
}

ART.mkdir(exist_ok=True)

(ART / "evaluation.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\n========== EVALUATION RESULTS ==========\n")
print(json.dumps(report, indent=2))
print("\nEvaluation completed successfully.")
print("Best rating RMSE weights:", best_rating_rmse["weights"])
print("Best ranking NDCG@10 weights:", best_ranking_ndcg["weights"])
