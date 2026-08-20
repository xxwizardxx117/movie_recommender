# 🎥 Movie Recommender

A content-based movie recommendation system built on the
[TMDB 5000 Movie Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
dataset. Pick a film and the app returns the **ten** most similar titles with
posters fetched live from the TMDB API, a watchlist you can share by link, and a
light/dark theme.

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

- **Search 4,799 films** from a single dropdown and get **10 recommendations**
  in a 5 × 2 poster grid, each with its release year.
- **Ranked by cosine similarity** over a bag-of-words "tag" profile built from
  the overview, genres, keywords, top-3 billed cast and director.
- **Watch Later panel** — hit *+ Watch later* under any result to pin it. Press
  again to remove it; there is no dead end.
- **The watchlist lives in the URL**, so it survives a refresh, can be
  bookmarked, and is shared simply by sending the link. No database, no
  accounts, no sign-in. A **share button** renders the link as a QR code for
  handing a list to a phone.
- **Two layouts for saved films** — a compact list or a two-up poster grid — and
  the choice travels in the link as well.
- **Light and dark themes** ("matinee" and "evening show") with a toggle in the
  masthead. The choice is held in the URL, so it survives a reload.
- **Degrades gracefully.** Posters are fetched in parallel and cached; a film
  with no artwork gets a placeholder card, and a TMDB outage gets a *different*
  placeholder plus a one-line notice — so "no poster exists" and "TMDB is down"
  never look the same. Recommendations come from the local model and are
  unaffected either way.

---

## 🧠 How the recommender works

The model is **content-based** — there is no user history, no ratings matrix and
no collaborative filtering. Everything happens offline in
`trainer/Movie_Recommender.ipynb`; the app only consumes the resulting artifacts.

| Step | What happens |
| --- | --- |
| 1. Merge | `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` are merged on the **numeric id** (`movies.id` ↔ `credits.movie_id`) after dropping the duplicate `title` column → 4,803 rows. |
| 2. Trim | Keep `movie_id`, `title`, `release_date`, `overview`, `genres`, `keywords`, `cast`, `crew` (plus `vote_count`, `popularity`, `budget`, `revenue`, which are loaded but not used by the model). |
| 3. Clean | `dropna().reset_index(drop=True)` → **4,799 rows with a contiguous `0…4798` index**. |
| 4. Parse | The JSON-ish string columns are parsed with `ast.literal_eval` and flattened to name lists. Cast is capped at the **top 3** billed actors; crew is reduced to the **director** only. |
| 5. Normalise | Spaces are stripped inside each token and everything is lowercased, so `"James Cameron"` → `jamescameron` and stays a single token. |
| 6. Tag | All the lists are concatenated into one `tags` string per film. |
| 7. Stem | Porter stemming (NLTK) collapses word variants — `paraplegic` → `parapleg`, `marines` → `marin`. |
| 8. Vectorise | `CountVectorizer(max_features=4500, stop_words='english')` over the **stemmed** text → a 4799 × 4500 count matrix. |
| 9. Similarity | `cosine_similarity` produces a 4799 × 4799 matrix. Cosine rather than Euclidean because angular distance stays meaningful in high dimensions. |
| 10. Export | The trimmed dataframe and the similarity matrix are pickled and gzipped into the project root. |

> **Order matters in steps 7–8.** Stemming has to run *before* vectorising, or
> the vectors are built from unstemmed text and the stemming pass changes
> nothing. Fixing that ordering altered 45% of the vocabulary and the top-5 for
> 99.6% of the catalogue.

At request time `recommend()` finds the film's **positional** index, sorts its
similarity row descending, filters out the query film itself and takes the top
ten. Positions are used rather than pandas labels, and the query film is removed
by identity rather than by slicing off rank 0 — both so a gapped index or a
duplicate title can never serve someone else's recommendations.

---

## 🗂️ Project structure

```
03_Movie_Recommender/
├── app.py                        # Streamlit app — UI, recommend, poster fetch, watchlist
├── style.css                     # Stylesheet, targeted at stable data-testid selectors
├── pyproject.toml                # Project metadata + dependency declarations (uv)
├── requirements.txt              # Generated deployment manifest — Streamlit Cloud reads this
├── LICENSE                       # GNU GPL v3
├── movie_list.pkl.gz             # Model: gzipped DataFrame (movie_id, title, release_date, tags)
├── similarity.pkl.gz             # Model: gzipped 4799 × 4799 cosine similarity matrix
├── .streamlit/
│   ├── config.toml               # Base Streamlit chrome — theme colours, toolbar, telemetry
│   └── secrets.toml              # Local-only, git-ignored — holds TMDB_API_KEY
├── assets/                       # Bundled placeholder posters, one pair per theme
│   ├── no-poster-dark.png        #   film has no artwork in TMDB
│   ├── no-poster-light.png
│   ├── poster-offline-dark.png   #   TMDB could not be reached
│   └── poster-offline-light.png
├── trainer/                      # Offline training — not used by the running app
│   ├── Movie_Recommender.ipynb   # Pipeline: parse → tag → stem → vectorise → similarity → export
│   └── Data/                     # Local-only, git-ignored (see below)
│       ├── tmdb_5000_movies.csv
│       ├── tmdb_5000_credits.csv
│       ├── pure_API testing.py   # Scratch script for TMDB daily ID-export downloads
│       └── *_ids.json/           # TMDB daily ID exports (movies, people, TV, …)
├── .devcontainer/                # GitHub Codespaces config — not used by the deployment
├── .venv/                        # Local-only, created by uv, git-ignored
├── uv.lock                       # Local-only, git-ignored — see "Managing dependencies"
└── docs/                         # Local-only, git-ignored working notes
```

> **Note:** `Data/`, `docs/`, `.venv/`, `uv.lock` and `.streamlit/secrets.toml`
> are all in `.gitignore`. The raw CSVs are not committed, so a fresh clone can
> run the **app** but cannot re-run the **notebook** until the dataset is
> downloaded from Kaggle.

**Two halves, one hand-off.** `trainer/` builds the model; the root serves it.
The dataset lives inside `trainer/Data/` because only the trainer reads it — the
app never touches it. The notebook reads `Data/*.csv` from beside itself and
writes `../movie_list.pkl.gz` and `../similarity.pkl.gz` up into the project
root, which is exactly where `app.py` loads them from. The app never imports the
notebook; it only consumes those two files, so deploying requires no training
step and no dataset.

---

## 🚀 Getting started

Because `movie_list.pkl.gz` and `similarity.pkl.gz` are committed, **no training
step is required** — clone, add a key, run. The app opens at
`http://localhost:8501`.

### Prerequisite — a TMDB API key

The app reads the key from `st.secrets` first and falls back to the
`TMDB_API_KEY` environment variable; if neither is set it stops with an
explanatory message rather than failing halfway through a render. Get a free key
from [TMDB → Settings → API](https://www.themoviedb.org/settings/api), then
create `.streamlit/secrets.toml`:

```toml
TMDB_API_KEY = "your_key_here"
```

That file is git-ignored — **never commit it**. Everything except the posters
works without a key, but the app deliberately refuses to start without one, so
that a missing key is never mistaken for a broken TMDB.

### Option A — with uv (recommended)

Install [uv](https://docs.astral.sh/uv/) once with `winget install astral-sh.uv`
(Windows) or `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS / Linux),
then:

```bash
git clone https://github.com/xxwizardxx117/movie_recommender.git
cd movie_recommender

uv run streamlit run app.py
```

That single command creates the virtual environment, resolves the dependencies
declared in `pyproject.toml` and starts the app. There is no separate activate
step, and you do not need Python pre-installed — uv fetches an interpreter
itself.

To build the environment without launching anything, use `uv sync`.

> The lockfile is **not** committed (see
> [Managing dependencies](#managing-dependencies)), so uv resolves fresh from
> the version ranges in `pyproject.toml` and may pick newer releases than the
> deployment runs. If you need the exact deployed versions, install from the
> pinned `requirements.txt` instead — `uv pip install -r requirements.txt`, or
> Option B below.

### Option B — classic venv + pip

The original workflow still works unchanged. It needs **Python 3.11 or newer**
already installed:

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

`requirements.txt` is fully pinned, so this reproduces the deployed environment
exactly. You manage the virtual environment yourself and supply your own Python.

> **If you contribute via this route,** note that `requirements.txt` is
> generated and should not be hand-edited. Dependency changes belong in
> `pyproject.toml` — see [Managing dependencies](#managing-dependencies).

### Re-train the model

Training lives in `trainer/` and is entirely separate from the app. It needs the
raw dataset plus the offline-only tooling in the `dev` dependency group
(`scikit-learn`, `nltk`, `numpy`, `jupyter`), none of which the deployed app
imports:

```bash
uv sync                       # installs the runtime + dev groups
uv run jupyter notebook       # or just select .venv as the kernel in your IDE
```

1. Download the [TMDB 5000 dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
2. Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in `trainer/Data/`.
3. Run `trainer/Movie_Recommender.ipynb` top to bottom. The final cell rewrites
   `movie_list.pkl.gz` and `similarity.pkl.gz` **in the project root**, replacing
   the committed model.

> The notebook's paths are relative to its own folder — it reads `Data/…` from
> beside itself and writes `../movie_list.pkl.gz` up to the root. Jupyter and
> VS Code both set the working directory to the notebook's folder by default, so
> this works as-is; if you launch it with the working directory forced
> elsewhere, the paths will not resolve.

> **Both artifacts must be regenerated together.** They are positionally
> coupled: row *i* of the dataframe is row *i* of the similarity matrix. Mixing
> a new `movie_list.pkl.gz` with an old `similarity.pkl.gz` produces
> plausible-looking recommendations for the wrong films.

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

`requirements.txt` is **generated, not hand-edited**, and `--no-dev` keeps the
notebook tooling out of production.

> **Why `uv.lock` is git-ignored.** Streamlit Community Cloud picks the first
> dependency file it finds, and `uv.lock` ranks *above* `requirements.txt`.
> Committing it would silently switch the deployment to `uv sync` resolving from
> `pyproject.toml` — which installs dependency groups by default and would drag
> Jupyter and scikit-learn into the running app. The lockfile is kept local; the
> pins that matter are exported to `requirements.txt` instead.

---

## ☁️ Deployment

Deployed on **Streamlit Community Cloud**, which installs from
`requirements.txt` and serves `app.py` on **Python 3.14**. The TMDB key lives in
the app's **Settings → Secrets** as `TMDB_API_KEY`, in the same top-level TOML
form as the local `.streamlit/secrets.toml`.

Everything the app needs at runtime is committed — the two model artifacts, the
stylesheet, `.streamlit/config.toml` and the four placeholder images — so a push
is the whole deployment.

---

## 🛠️ Tech stack

| Layer | Tools |
| --- | --- |
| App / UI | Streamlit, custom CSS (stable `data-testid` selectors, CSS custom properties for theming) |
| Data | pandas, NumPy |
| ML | scikit-learn (`CountVectorizer`, `cosine_similarity`), NLTK (`PorterStemmer`) |
| External API | TMDB — posters and film metadata |
| Share links | segno — pure-Python QR generation, no native build step |
| Images | Pillow (via Streamlit) |
| Packaging | uv — `pyproject.toml`, exported to `requirements.txt` |
| Deployment | Streamlit Community Cloud (Python 3.14) |

Runtime dependencies are declared in `pyproject.toml`. The notebook-only tooling
lives in the `dev` dependency group and is excluded from the deployment
manifest.

---

## ⚠️ Known limitations

- **Content-based only.** Recommendations reflect textual overlap, not
  popularity or quality. `vote_count`, `popularity`, `budget` and `revenue` are
  loaded in the notebook but never reach the model, so a beloved film and a
  forgotten one with the same tags rank identically.
- **Static catalogue.** The dataset is a 2017-era snapshot, so recent films are
  absent. Re-running the notebook against a refreshed dataset is the only way to
  update it.
- **No evaluation.** There is no held-out test set or ranking metric, so changes
  to the pipeline can be shown to be *different* but not *better*. Judgement is
  currently by inspection.
- **Vocabulary cap.** `max_features=4500` truncates the vocabulary to the most
  frequent stems; rarer but more distinctive tokens fall out. Raw counts also
  weight a ubiquitous word as heavily as a rare one — TF-IDF would not.
- **The similarity matrix is shipped whole.** 4799 × 4799 floats compress to a
  47.8 MB artifact that is loaded into memory once per server process. Storing
  the top-N neighbours per film instead would cut it to a few MB; it has not
  been done yet.
- **Duplicate titles are real.** Three titles appear twice in the dropdown —
  they are genuinely different films that share a name (two *Batman*s, and so
  on), not a data error.
- **Watchlists live in the URL.** That is what makes them shareable with no
  accounts and no database, but a very long list makes a very long link, and
  clearing the address bar clears the list.
- **TMDB is needed for artwork only.** If it is unreachable the recommendations
  still work; the posters fall back to a placeholder and the app says so.

---

## 🤝 Contributing

📌 _If you find any errors, or would like to improve or extend this repository,
please consider contributing by opening an issue or a pull request._
