# 🎥 Movie Recommender

A content-based movie recommendation system built on the
[TMDB 5000 Movie Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
dataset. Pick a movie, and the app returns the five most similar titles with
their posters fetched live from the TMDB API.

**Live app → [whatowatchnext.streamlit.app](https://whatowatchnext.streamlit.app/)**

---

## 📘 Index

| Reference | Description |
| --- | --- |
| 🔗 [Recommender](https://whatowatchnext.streamlit.app/) | Opens the deployed web app. |
| 🔗 [Kaggle Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) | Source data (TMDB 5000 movies + credits). |
| 🔗 [TMDB API](https://developer.themoviedb.org/docs) | Poster / metadata API used at runtime. |

---

## ✨ What it does

- Search or select any of ~4,800 movies from a dropdown.
- Get 5 recommendations ranked by cosine similarity over a bag-of-words
  "tag" profile built from overview, genres, keywords, top-3 cast and director.
- Posters are pulled on demand from TMDB, with automatic retry on connection
  failure.
- A **Watch Later** sidebar lets you upload poster images and pin them in a
  two-column grid for the session.

---

## 🧠 How the recommender works

The model is **content-based** — there is no user history, no ratings matrix,
and no collaborative filtering. Everything happens offline in
`trainer/Movie_Recommender.ipynb`, and the app only consumes the resulting
artifacts.

| Step | What happens |
| --- | --- |
| 1. Merge | `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` are merged on `title` → 4,809 rows. |
| 2. Trim | Keep `movie_id`, `title`, `overview`, `genres`, `keywords`, `cast`, `crew`. |
| 3. Clean | Drop 3 rows with a null `overview` → 4,806 rows. No duplicates. |
| 4. Parse | JSON-ish string columns are parsed with `ast.literal_eval` and flattened to name lists. Cast is capped at the **top 3** billed actors; crew is reduced to the **director only**. |
| 5. Normalise | Spaces are stripped inside each token and everything is lowercased, so `"James Cameron"` → `jamescameron` and stays a single token. |
| 6. Tag | All lists are concatenated into one `tags` string per movie. |
| 7. Vectorise | `CountVectorizer(max_features=4500, stop_words='english')` → a 4806 × 4500 count matrix. |
| 8. Stem | Porter stemming (NLTK) collapses word variants (`parapleg`, `marin`, …). |
| 9. Similarity | `cosine_similarity` over the vectors produces a 4806 × 4806 matrix. Cosine is used instead of Euclidean because angular distance stays meaningful in high dimensions. |
| 10. Export | The trimmed dataframe and the similarity matrix are pickled and gzipped. |

At request time, `recommend()` looks up the movie's row index, sorts its
similarity row descending, and takes entries `[1:6]` — index `0` is skipped
because it is the movie compared with itself.

---

## 🗂️ Project structure

```
03_Movie_Recommender/
├── app.py                        # Streamlit web app (UI + recommend + poster fetch)
├── style.css                     # Custom CSS injected into the Streamlit page
├── pyproject.toml                # Project metadata + dependency declarations (uv)
├── uv.lock                       # Exact resolved versions — committed on purpose
├── requirements.txt              # Generated from uv.lock for Streamlit Cloud
├── LICENSE                       # GNU GPL v3
├── movie_list.pkl.gz             # Model: gzipped DataFrame (movie_id, title, tags)
├── similarity.pkl.gz             # Model: gzipped 4806 × 4806 cosine similarity matrix
├── trainer/                      # Offline training — not used by the running app
│   ├── Movie_Recommender.ipynb   # Pipeline: EDA → vectorise → similarity → export
│   └── Data/                     # Local-only, git-ignored (see below)
│       ├── tmdb_5000_movies.csv
│       ├── tmdb_5000_credits.csv
│       ├── pure_API testing.py   # Scratch script for TMDB daily ID-export downloads
│       └── *_ids.json/           # TMDB daily ID exports (movies, people, TV, …)
├── .venv/                        # Local-only, created by uv, git-ignored
└── docs/                         # Local-only, git-ignored working notes
```

> **Note:** `Data/`, `docs/` and `.venv/` are listed in `.gitignore`. The raw
> CSVs are not committed, so a fresh clone can run the **app** but cannot re-run
> the **notebook** until the dataset is downloaded from Kaggle.

**Two halves, one hand-off.** `trainer/` builds the model; the root serves it.
The dataset lives inside `trainer/Data/` because only the trainer reads it — the
app never touches it. The notebook reads `Data/*.csv` from beside itself and
writes `../movie_list.pkl.gz` and `../similarity.pkl.gz` up into the project
root, which is exactly where `app.py` loads them from. The app never imports the
notebook; it only consumes those two files, so deploying requires no training
step and no dataset.

---

## 🚀 Getting started

There are two ways to run this project. **Both are fully supported** — pick
whichever suits you. Either way the app opens at `http://localhost:8501`, and
because `movie_list.pkl.gz` and `similarity.pkl.gz` are committed, **no training
step is required**.

### Option A — with uv (recommended)

Install [uv](https://docs.astral.sh/uv/) once with `winget install astral-sh.uv`
(Windows) or `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS / Linux),
then:

```bash
git clone https://github.com/xxwizardxx117/movie_recommender.git
cd movie_recommender

uv run streamlit run app.py
```

That single command creates the virtual environment, installs the exact versions
recorded in `uv.lock`, and starts the app. There is no separate activate step,
and you do not need Python pre-installed — uv fetches an interpreter itself.

To build the environment without launching anything, use `uv sync`.

### Option B — classic venv + pip

The original workflow still works unchanged. It needs **Python 3.11 or newer
already installed** on your system:

```bash
git clone https://github.com/xxwizardxx117/movie_recommender.git
cd movie_recommender

python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

`requirements.txt` is an ordinary pip requirements file, so this resolves to
exactly the same versions uv installs — it is generated *from* `uv.lock`. The
only differences are that you manage the virtual environment yourself and you
must supply your own Python.

> **If you contribute via this route,** note that `requirements.txt` is
> generated and should not be hand-edited. Dependency changes belong in
> `pyproject.toml`; see [Managing dependencies](#managing-dependencies).

### Re-train the model

Training lives in `trainer/` and is entirely separate from the app. It needs the
raw dataset plus the offline-only tooling in the `dev` dependency group
(`scikit-learn`, `nltk`, `numpy`, `jupyter`), which the deployed app never
imports:

```bash
uv sync                       # installs runtime + dev groups
uv run jupyter notebook       # or select .venv as the kernel in your IDE
```

1. Download the [TMDB 5000 dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
2. Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in `trainer/Data/`.
3. Run `trainer/Movie_Recommender.ipynb` top to bottom. The final cell rewrites
   `movie_list.pkl.gz` and `similarity.pkl.gz` **in the project root**, replacing
   the committed model.

> The notebook's paths are relative to its own folder — it reads `Data/…` from
> beside itself and writes `../movie_list.pkl.gz` up to the root. Jupyter and
> VS Code both set the working directory to the notebook's folder by default, so
> this works as-is; if you launch it with the working directory forced elsewhere,
> the paths will not resolve.

### Managing dependencies

```bash
uv add <package>              # add a runtime dependency
uv add --dev <package>        # add a notebook-only dependency
uv lock --upgrade             # refresh the lockfile
```

After any dependency change, regenerate the deployment manifest:

```bash
uv export --no-dev --no-hashes --no-emit-project -o requirements.txt
```

`requirements.txt` is **generated, not hand-edited**. Streamlit Community Cloud
reads it rather than `pyproject.toml`, so it is kept in the repository as the
deployment manifest and stays pinned to whatever `uv.lock` resolves.

---

## 🛠️ Tech stack

| Layer | Tools |
| --- | --- |
| App / UI | Streamlit, Bootstrap 5.3 (CDN), custom CSS |
| Data | pandas, NumPy |
| ML | scikit-learn (`CountVectorizer`, `cosine_similarity`), NLTK (`PorterStemmer`) |
| External API | TMDB — posters and movie metadata |
| Images | Pillow |
| Packaging | uv — `pyproject.toml` + `uv.lock` |
| Deployment | Streamlit Community Cloud |

Runtime dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.
The notebook-only tooling lives in the `dev` dependency group and is excluded
from the deployment manifest.

---

## ⚠️ Known limitations

- **Content-based only.** Recommendations reflect textual overlap, not
  popularity or quality. `vote_count`, `popularity` and `revenue` are selected
  in the notebook but never used in the model.
- **Static catalogue.** The dataset is a 2017-era snapshot, so recent films are
  absent. Re-running the notebook is the only way to refresh it.
- **Vocabulary cap.** `max_features=4500` truncates the vocabulary, and stemming
  is applied *after* vectorisation, so it does not actually merge tokens in the
  matrix that was used for similarity.
- **Cold start on load.** The 39 MB similarity matrix is un-gzipped and
  unpickled on every app start, which makes the first request slow.
- **API key in source.** The TMDB key is hardcoded in `app.py`; it should move
  to Streamlit secrets or an environment variable.

---

## 🤝 Contributing

📌 _If you find any errors, or would like to improve or extend this repository,
please consider contributing by opening an issue or a pull request._
