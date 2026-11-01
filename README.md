# CineMatch — Personalized Movie Recommendation System

CineMatch is a professional Django + machine-learning movie recommender built around **collaborative filtering**. It combines User-Based CF, Item-Based CF, SVD matrix factorization and TF-IDF content similarity into a hybrid ranking model.

## Highlights

- Secure Django authentication and per-user profiles
- MovieLens training pipeline
- User-based collaborative filtering
- Item-based collaborative filtering
- SVD latent-factor model
- TF-IDF title/genre similarity
- Hybrid ensemble ranking
- Explainable recommendations
- Search and movie detail pages
- Rating feedback loop
- Genre preference onboarding
- Offline evaluation script with RMSE and MAE
- Health endpoint for model readiness
- Admin interface
- Responsive dark UI

## Architecture

```text
MovieLens
   ↓
Preprocessing
   ├── User × Movie matrix → User-CF / Item-CF
   ├── Mean-centered matrix → SVD
   └── Title + Genres → TF-IDF content model
             ↓
       Hybrid Ranker
             ↓
     Personalized Top-N
             ↓
        Django UI
             ↓
       New user ratings
             └──────────────→ feedback loop
```

## Model score

The default ensemble is:

`0.35 × SVD + 0.25 × Item-CF + 0.15 × User-CF + 0.25 × Content`

Scores are percentile-normalized before blending. Already-rated movies are removed from the final candidate set.

## Setup

Django 6.1 supports Python 3.12+; the project follows Django's standard authentication/session patterns. See the official Django documentation for authentication and project structure.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python scripts/download_data.py
python scripts/train.py
python scripts/evaluate.py
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Open `http://127.0.0.1:8000/`.

## Evaluation

The evaluation script performs an 80/20 per-user holdout for users with at least five ratings and reports:

- RMSE
- MAE
- test interactions

For a research submission, extend this with Precision@K, Recall@K, MAP@K and NDCG@K and compare Popularity, User-KNN, Item-KNN, SVD, SVD++ and the Hybrid model.

## Project layout

```text
CineMatch_Movie_Recommender/
├── config/                 # Django project configuration
├── recommender/            # Web app + recommendation engine
│   ├── migrations/
│   ├── templates/
│   ├── static/
│   ├── engine.py
│   ├── forms.py
│   ├── models.py
│   └── views.py
├── scripts/
│   ├── download_data.py
│   ├── train.py
│   └── evaluate.py
├── data/raw/               # downloaded dataset (ignored by git)
├── artifacts/              # trained model (ignored by git)
├── manage.py
└── requirements.txt
```

## Production notes

Before deployment, set `DJANGO_SECRET_KEY`, `DJANGO_DEBUG=0`, and `DJANGO_ALLOWED_HOSTS`; run `collectstatic`; use a production WSGI/ASGI server; and move from SQLite to PostgreSQL for multi-user production workloads.

## ML notes

The collaborative components use cosine similarity over the user-movie interaction matrix, while the content component uses normalized TF-IDF vectors and cosine similarity. These are standard, interpretable building blocks for a hybrid recommender.
