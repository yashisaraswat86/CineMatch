"""Hybrid recommendation engine backed by sparse top-K neighbor artifacts."""
from pathlib import Path
import joblib
import numpy as np
from .config import ARTIFACT_DIR, WEIGHTS, TOP_N

class RecommendationEngine:
    def __init__(self, artifact_dir=ARTIFACT_DIR):
        self.artifact_dir = Path(artifact_dir)
        self.loaded = False

    def load(self):
        path = self.artifact_dir / 'model_bundle.joblib'
        if not path.exists():
            raise FileNotFoundError('Model artifacts missing. Run: python scripts/train.py')
        self.__dict__.update(joblib.load(path))
        self.loaded = True
        return self

    def _ensure(self):
        if not self.loaded:
            self.load()

    @staticmethod
    def _norm(values):
        x = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(x)
        if not finite.any():
            return np.zeros_like(x)
        lo, hi = np.percentile(x[finite], [5, 95])
        if hi <= lo:
            return np.zeros_like(x)
        return np.clip((np.nan_to_num(x, nan=lo) - lo) / (hi - lo), 0, 1)

    def _svd(self, user_idx):
        if user_idx is None:
            return np.full(len(self.movies), self.global_mean, dtype=np.float32)
        return np.clip(self.user_means[user_idx] + self.user_factors[user_idx] @ self.item_factors.T, .5, 5)

    def _user_cf(self, user_idx):
        if user_idx is None:
            return np.full(len(self.movies), self.global_mean, dtype=np.float32)
        neighbors = self.user_neighbor_idx[user_idx]
        weights = self.user_neighbor_sim[user_idx]
        if len(neighbors) == 0 or weights.sum() <= 1e-9:
            return np.full(len(self.movies), self.global_mean, dtype=np.float32)
        # Predict from neighbors' deviations from their own mean; unseen items get global fallback.
        neighbor_ratings = self.ratings_matrix[neighbors].toarray()
        deviations = neighbor_ratings - self.user_means[neighbors, None]
        numerator = (weights[:, None] * deviations).sum(axis=0)
        denominator = (weights[:, None] * (neighbor_ratings != 0)).sum(axis=0)
        pred = self.user_means[user_idx] + np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-9)
        return np.clip(np.where(denominator > 1e-9, pred, self.global_mean), .5, 5)

    def _item_cf(self, indices, ratings):
        if not indices:
            return np.full(len(self.movies), self.global_mean, dtype=np.float32)
        accum = np.zeros(len(self.movies), dtype=np.float64)
        weights = np.zeros(len(self.movies), dtype=np.float64)
        for idx, rating in zip(indices, ratings):
            neighbors = self.item_neighbor_idx[idx]
            sims = self.item_neighbor_sim[idx]
            valid = sims > 0
            neighbors, sims = neighbors[valid], sims[valid]
            if len(neighbors):
                accum[neighbors] += sims * (rating - 3.0)
                weights[neighbors] += sims
        pred = 3.0 + np.divide(accum, weights, out=np.zeros_like(accum), where=weights > 1e-9)
        return np.clip(np.where(weights > 1e-9, pred, self.global_mean), .5, 5)

    def _content(self, indices, ratings):
        if not indices:
            return np.zeros(len(self.movies), dtype=np.float32)
        liked = [i for i, r in zip(indices, ratings) if r >= 4]
        disliked = [i for i, r in zip(indices, ratings) if r <= 2]
        if not liked:
            liked = indices
        scores = np.zeros(len(self.movies), dtype=np.float64)
        for i in liked:
            neighbors, sims = self.content_neighbor_idx[i], self.content_neighbor_sim[i]
            scores[neighbors] += sims
        scores /= max(1, len(liked))
        # Mildly suppress items similar to explicitly disliked titles.
        for i in disliked:
            neighbors, sims = self.content_neighbor_idx[i], self.content_neighbor_sim[i]
            scores[neighbors] -= 0.5 * sims / max(1, len(disliked))
        return scores

    def recommend(self, ratings, n=TOP_N):
        self._ensure()
        known = [(int(m), float(r)) for m, r in ratings if int(m) in self.movie_to_idx]
        if not known:
            result = self.movies.sort_values(['mean_rating', 'rating_count'], ascending=False).head(n).copy()
            result['score'] = result['mean_rating'] / 5.0
            result['reason'] = 'Popular with highly rated by the community'
            return result
        indices = [self.movie_to_idx[m] for m, _ in known]
        values = [r for _, r in known]
        user_idx = None
        # Map an existing MovieLens user only when the user's own rating vector is sufficiently similar.
        if len(known) >= 3:
            profile = np.zeros(self.ratings_matrix.shape[1], dtype=np.float32)
            profile[indices] = values
            best_user, best_sim = None, -1.0
            for candidate in range(self.ratings_matrix.shape[0]):
                row = self.ratings_matrix.getrow(candidate).toarray().ravel()
                mask = row[indices] > 0
                if mask.sum() < 2:
                    continue
                a, b = profile[indices][mask], row[indices][mask]
                denom = np.linalg.norm(a) * np.linalg.norm(b)
                sim = float(np.dot(a, b) / denom) if denom else 0.0
                if sim > best_sim:
                    best_user, best_sim = candidate, sim
            if best_sim > 0:
                user_idx = best_user
        svd = self._svd(user_idx)
        ucf = self._user_cf(user_idx)
        icf = self._item_cf(indices, values)
        content = self._content(indices, values)
        components = {'svd': self._norm(svd), 'user_cf': self._norm(ucf), 'item_cf': self._norm(icf), 'content': self._norm(content)}
        total_weight = sum(max(0.0, WEIGHTS[k]) for k in components)
        score = sum(max(0.0, WEIGHTS[k]) * v for k, v in components.items()) / max(total_weight, 1e-9)
        for idx in indices:
            score[idx] = -1
        chosen = np.argsort(score)[::-1][:n]
        output = self.movies.iloc[chosen].copy()
        output['score'] = score[chosen]
        reasons = []
        for idx in chosen:
            if content[idx] > 0.15:
                reasons.append('Similar genres and movie themes to titles you enjoyed')
            elif icf[idx] > 3.2:
                reasons.append('Similar to movies you rated positively')
            elif svd[idx] >= 4.0:
                reasons.append('High predicted rating from your latent taste profile')
            else:
                reasons.append('Recommended by combining your preference signals')
        output['reason'] = reasons
        return output

    def search(self, query, limit=24):
        self._ensure()
        query = query.lower().strip()
        movies = self.movies.copy()
        if not query:
            return movies.sort_values(['mean_rating', 'rating_count'], ascending=False).head(limit)
        mask = movies.title.str.lower().str.contains(query, regex=False, na=False) | movies.genres.str.lower().str.contains(query, regex=False, na=False)
        return movies[mask].sort_values(['rating_count', 'mean_rating'], ascending=False).head(limit)

_engine = None
def get_engine():
    global _engine
    if _engine is None:
        _engine = RecommendationEngine().load()
    return _engine
