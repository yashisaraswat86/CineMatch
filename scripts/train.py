"""Train scalable CineMatch artifacts from MovieLens ratings.csv and movies.csv."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
RAW, ART = ROOT / 'data' / 'raw', ROOT / 'artifacts'
ART.mkdir(parents=True, exist_ok=True)

base = next(RAW.rglob('ratings.csv'), None)
if base is None:
    raise SystemExit('Dataset missing. Run: python scripts/download_data.py')
movies = pd.read_csv(base.parent / 'movies.csv')
ratings = pd.read_csv(base)[['userId', 'movieId', 'rating']].dropna()
movies = movies[['movieId', 'title', 'genres']].drop_duplicates('movieId')
ratings = ratings[ratings.movieId.isin(movies.movieId)].copy()
# Keep a deterministic, interaction-rich subset for large datasets. Avoid dense matrices.
if len(ratings) > 1_500_000:
    keep = ratings.groupby('userId').size().nlargest(20_000).index
    ratings = ratings[ratings.userId.isin(keep)]
users = np.sort(ratings.userId.unique())
mids = np.sort(movies.movieId.unique())
u2i = {int(v): i for i, v in enumerate(users)}
m2i = {int(v): i for i, v in enumerate(mids)}
rows = ratings.userId.map(u2i).to_numpy()
cols = ratings.movieId.map(m2i).to_numpy()
vals = ratings.rating.to_numpy(dtype=np.float32)
mat = csr_matrix((vals, (rows, cols)), shape=(len(users), len(mids)), dtype=np.float32)

# User means and a sparse user-centred matrix for latent-factor learning.
counts = np.asarray((mat != 0).sum(axis=1)).ravel()
sums = np.asarray(mat.sum(axis=1)).ravel()
global_mean = float(vals.mean())
user_means = np.divide(sums, counts, out=np.full(len(users), global_mean, dtype=np.float32), where=counts > 0)
coo = mat.tocoo()
centered = csr_matrix((coo.data - user_means[coo.row], (coo.row, coo.col)), shape=mat.shape, dtype=np.float32)
k = max(2, min(50, min(centered.shape) - 1))
svd = TruncatedSVD(n_components=k, n_iter=7, random_state=42)
user_factors = svd.fit_transform(centered).astype(np.float32)
item_factors = svd.components_.T.astype(np.float32)

# Store only top-K neighbors; this avoids quadratic dense similarity matrices.
def top_neighbors(matrix, k=40):
    n = matrix.shape[0]
    if n <= 1:
        return np.zeros((n, 0), dtype=np.int32), np.zeros((n, 0), dtype=np.float32)
    nn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=min(k + 1, n), n_jobs=-1)
    nn.fit(matrix)
    distances, indices = nn.kneighbors(matrix)
    out_i = np.zeros((n, min(k, n - 1)), dtype=np.int32)
    out_s = np.zeros_like(out_i, dtype=np.float32)
    for i in range(n):
        pairs = [(int(j), float(1.0 - d)) for j, d in zip(indices[i], distances[i]) if j != i]
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:out_i.shape[1]]
        for t, (j, sim) in enumerate(pairs):
            out_i[i, t], out_s[i, t] = j, max(0.0, sim)
    return out_i, out_s

user_neighbor_idx, user_neighbor_sim = top_neighbors(mat, 40)
item_neighbor_idx, item_neighbor_sim = top_neighbors(mat.T.tocsr(), 40)

movies['genres'] = movies.genres.fillna('').replace('(no genres listed)', '')
movies['content_text'] = movies.title.fillna('') + ' ' + movies.genres.str.replace('|', ' ', regex=False)
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), sublinear_tf=True, min_df=1)
content = vectorizer.fit_transform(movies.content_text)
content_neighbor_idx, content_neighbor_sim = top_neighbors(content.tocsr(), 30)

stats = ratings.groupby('movieId').rating.agg(['mean', 'count'])
movies['mean_rating'] = movies.movieId.map(stats['mean']).fillna(global_mean)
movies['rating_count'] = movies.movieId.map(stats['count']).fillna(0).astype(int)
movies = movies.set_index('movieId').reindex(mids).reset_index()
genres = sorted({g for s in movies.genres for g in str(s).split('|') if g and g != '(no genres listed)'})
metadata = {'users': int(len(users)), 'movies': int(len(mids)), 'ratings': int(len(ratings)), 'latent_factors': int(k), 'dataset': base.parent.name, 'similarity_storage': 'top-k neighbors (sparse)', 'user_neighbors': 40, 'item_neighbors': 40, 'content_neighbors': 30}
bundle = {
    'movies': movies, 'movie_to_idx': m2i, 'user_to_idx': u2i,
    'ratings_matrix': mat, 'user_means': user_means,
    'user_neighbor_idx': user_neighbor_idx, 'user_neighbor_sim': user_neighbor_sim,
    'item_neighbor_idx': item_neighbor_idx, 'item_neighbor_sim': item_neighbor_sim,
    'content_neighbor_idx': content_neighbor_idx, 'content_neighbor_sim': content_neighbor_sim,
    'user_factors': user_factors, 'item_factors': item_factors,
    'global_mean': global_mean, 'vectorizer': vectorizer, 'genres': genres, 'metadata': metadata,
}
joblib.dump(bundle, ART / 'model_bundle.joblib', compress=3)
(ART / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print(json.dumps(metadata, indent=2))