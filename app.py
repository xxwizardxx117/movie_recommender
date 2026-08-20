"""What to Watch Next - a content-based movie recommender.

Pick a film, get ten similar ones with posters from TMDB.

The model is built offline in trainer/Movie_Recommender.ipynb and shipped as two
gzipped pickles; this file only consumes them. See docs/ for the issue tracker.
"""

import base64
import concurrent.futures
import gzip
import io
import os
import pickle
import time
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import pandas as pd
import requests
import segno
import streamlit as st

BASE_DIR = Path(__file__).parent

RESULT_COUNT = 10
GRID_COLUMNS = 5
# Bits per film in a shared watchlist. The largest TMDB id in the catalogue is
# 447027, which needs 19 bits; 19 leaves headroom up to 524287 so re-exporting
# with newer films will not silently overflow. If a future catalogue ever
# exceeds that, this must go up AND the old width must stay readable.
ID_BITS = 19
# Public URL used when the app cannot work out its own address (see share_url).
FALLBACK_ORIGIN = "https://whatowatchnext.streamlit.app/"
# Share-QR ink and field. Deliberately NOT theme-derived - see share_qr_svg.
QR_DARK = "#17140F"
QR_LIGHT = "#F6EFE2"
# base64url alphabet, used to reject a mangled share token (see decode_watchlist).
_B64URL_ALPHABET = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)
# Height of the scrollable saved-films region, in px. Chosen so the panel
# header and the pinned hint both stay on screen on a typical laptop.
SAVED_LIST_HEIGHT = 620
POSTER_BASE = "https://image.tmdb.org/t/p/w500"
# A sentinel rather than a path. posterfetcher is cached, so if it returned a
# concrete file the placeholder would be frozen to whichever theme was active
# when the fetch first failed. The themed file is resolved at render time.
PLACEHOLDER_POSTER = "__no_poster__"
# Kept separate from PLACEHOLDER_POSTER on purpose. Both render a placeholder
# card, but they mean opposite things: "no poster" is a permanent fact about the
# film, "unavailable" is a transient network failure worth retrying. Collapsing
# them into one value made a TMDB outage look identical to a film with no
# artwork, with nothing on screen to tell them apart (issue UX-16).
UNAVAILABLE_POSTER = "__poster_unavailable__"
# How long a failed fetch is remembered. Long enough that the reruns fired by
# Save, Remove and the theme toggle do not each re-attempt ten fetches with
# retries and backoff; short enough that a brief outage does not leave the grid
# looking broken for the rest of the day. See posterfetcher.
POSTER_RETRY_AFTER = 60


# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------
# Two palettes, swapped at runtime. config.toml can only set one theme at
# startup, so the toggle works by injecting these as CSS custom properties and
# letting style.css reference them.

THEMES = {
    "dark": {
        "bg": "#17140F",
        "panel": "#13100C",
        "card": "#241E17",
        "border": "#3A3128",
        "text": "#F2E4CE",
        "muted": "#8A7A63",
        "placeholder": "#6B5F4D",
        "accent": "#E8A33D",
        "on-accent": "#412402",
        "input": "#0F0D0A",
        "rule": "#A33B2A",
    },
    "light": {
        "bg": "#F6EFE2",
        "panel": "#EFE6D5",
        "card": "#EAE0CD",
        "border": "#D6C7AE",
        "text": "#1E1913",
        "muted": "#7A6A52",
        "placeholder": "#A3927A",
        # The dark accent (#E8A33D) fails contrast on cream, so light mode uses
        # a deeper amber. The oxblood rule is the only colour shared by both.
        "accent": "#8A5A0C",
        "on-accent": "#FDFAF3",
        "input": "#FDFAF3",
        "rule": "#A33B2A",
    },
}

ICONS = {
    "play": '<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>',
    "github": '<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"><path d="M12 .3a12 12 0 00-3.8 23.4c.6.1.8-.3.8-.6v-2.2c-3.3.7-4-1.6-4-1.6-.6-1.4-1.4-1.8-1.4-1.8-1.1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1.1 1.8 2.8 1.3 3.5 1 .1-.8.4-1.3.8-1.6-2.7-.3-5.5-1.3-5.5-5.9 0-1.3.5-2.4 1.2-3.2-.1-.3-.5-1.5.1-3.2 0 0 1-.3 3.3 1.2a11.5 11.5 0 016 0C17.3 4.5 18.3 4.8 18.3 4.8c.6 1.7.2 2.9.1 3.2.8.8 1.2 1.9 1.2 3.2 0 4.6-2.8 5.6-5.5 5.9.4.4.8 1.1.8 2.2v3.3c0 .3.2.7.8.6A12 12 0 0012 .3"/></svg>',
    "linkedin": '<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"><path d="M20.4 20.4h-3.6v-5.6c0-1.3 0-3-1.9-3s-2.1 1.4-2.1 2.9v5.7H9.2V9h3.4v1.6h.1a3.8 3.8 0 013.4-1.9c3.6 0 4.3 2.4 4.3 5.5v6.2zM5.3 7.4a2.1 2.1 0 110-4.2 2.1 2.1 0 010 4.2zM7.1 20.4H3.5V9h3.6v11.4zM22.2 0H1.8C.8 0 0 .8 0 1.7v20.6c0 .9.8 1.7 1.8 1.7h20.4c1 0 1.8-.8 1.8-1.7V1.7c0-.9-.8-1.7-1.8-1.7z"/></svg>',
    "x": '<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"><path d="M18.9 1.2h3.7l-8 9.2 9.4 12.4h-7.3l-5.8-7.5-6.6 7.5H.6l8.6-9.8L.2 1.2h7.5l5.2 6.9zm-1.3 19.4h2L6.5 3.2H4.4z"/></svg>',
    "sun": '<svg viewBox="0 0 24 24" width="24" height="24"><path d="M12 7a5 5 0 100 10 5 5 0 000-10m0-5v2m0 16v2M4.2 4.2l1.4 1.4m12.8 12.8 1.4 1.4M2 12h2m16 0h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"/></svg>',
    "moon": '<svg viewBox="0 0 24 24" width="24" height="24"><path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z" stroke="currentColor" stroke-width="2" fill="none" stroke-linejoin="round"/></svg>',
    "ticket": '<svg viewBox="0 0 24 24" width="20" height="20"><path d="M3 8a2 2 0 012-2h14a2 2 0 012 2v2a2 2 0 000 4v2a2 2 0 01-2 2H5a2 2 0 01-2-2v-2a2 2 0 000-4z" stroke="currentColor" stroke-width="1.6" fill="none"/></svg>',
    "list": '<svg viewBox="0 0 24 24" width="24" height="24"><path d="M4 6h3v3H4zm0 6h3v3H4zm0 6h3v3H4zM10 7h10M10 13h10M10 19h10" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"/></svg>',
    "grid": '<svg viewBox="0 0 24 24" width="24" height="24"><path d="M4 4h7v7H4zm9 0h7v7h-7zM4 13h7v7H4zm9 0h7v7h-7z" stroke="currentColor" stroke-width="2" fill="none" stroke-linejoin="round"/></svg>',
}


# set_page_config must be the first Streamlit call in the script. It is placed
# here, above load_tmdb_api_key(), because that function calls st.error() when
# the key is missing - and any st.* render call before set_page_config raises
# StreamlitAPIException, which would hide the actual problem.
st.set_page_config(
    page_title="What to Watch Next",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def current_theme():
    """Return the active theme, read from the URL so it survives a refresh.

    Kept in query params rather than session state so the choice survives a
    reload and travels in a shared link. The toggle itself is a Streamlit
    button in its own masthead column - as an <a> it caused a full page
    navigation, which flashed config.toml's dark base before the new
    stylesheet landed.
    """
    return "light" if st.query_params.get("theme") == "light" else "dark"


def theme_toggle_href():
    """URL for the opposite theme, preserving everything else in the query.

    Values MUST be percent-encoded: 61 film titles contain "&" and 63 contain
    ",", so an unencoded "Batman & Robin" would split the query and silently
    truncate the film to "Batman".
    """
    params = {k: v for k, v in st.query_params.items()}
    params["theme"] = "dark" if current_theme() == "light" else "light"
    return "?" + "&".join(
        f"{quote(str(k), safe='')}={quote(str(v), safe='')}"
        for k, v in params.items()
    )


def current_view():
    """Watch Later layout: 'list' (one per row) or 'grid' (two per row)."""
    return "grid" if st.query_params.get("view") == "grid" else "list"


def poster_src(poster):
    """Resolve a poster value to something st.image can render.

    Real TMDB URLs pass through; each sentinel becomes its own placeholder card
    for the currently active theme. The two cards are deliberately different
    artwork - a slashed play mark reads as "this failed", a plain one as "there
    is nothing here" - so the distinction survives into the Watch Later panel,
    where there is no room for a message.
    """
    if poster == PLACEHOLDER_POSTER:
        return str(BASE_DIR / "assets" / f"no-poster-{current_theme()}.png")
    if poster == UNAVAILABLE_POSTER:
        return str(BASE_DIR / "assets" / f"poster-offline-{current_theme()}.png")
    return poster


def inject_theme():
    """Write the active palette out as CSS custom properties."""
    palette = THEMES[current_theme()]
    variables = "\n".join(f"    --wtw-{k}: {v};" for k, v in palette.items())
    st.markdown(f"<style>:root {{\n{variables}\n}}</style>", unsafe_allow_html=True)


def load_stylesheet():
    with open(BASE_DIR / "style.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Model artifacts
# --------------------------------------------------------------------------


@st.cache_resource
def load_artifacts():
    """Load the model once per server process.

    Streamlit re-executes this script on every widget interaction. Without the
    cache, the 47 MB similarity matrix would be un-gzipped and unpickled each
    time (issue PERF-01).
    """
    with gzip.open(BASE_DIR / "movie_list.pkl.gz", "rb") as f:
        movies = pickle.load(f)
    with gzip.open(BASE_DIR / "similarity.pkl.gz", "rb") as f:
        similarity = pickle.load(f)
    return movies, similarity


# --------------------------------------------------------------------------
# TMDB
# --------------------------------------------------------------------------


def load_tmdb_api_key():
    """Read the key from Streamlit secrets, falling back to the environment.

    StreamlitSecretNotFoundError subclasses FileNotFoundError (no secrets file);
    a missing key inside an existing file raises KeyError.
    """
    try:
        key = st.secrets["TMDB_API_KEY"]
    except (KeyError, FileNotFoundError):
        key = os.environ.get("TMDB_API_KEY", "")

    if not key:
        st.error(
            "TMDB_API_KEY is not configured. Add it to .streamlit/secrets.toml "
            "or set it as an environment variable, then reload the app."
        )
        st.stop()

    return key


TMDB_API_KEY = load_tmdb_api_key()


def make_request_with_retry(url, params=None, max_retries=3):
    """GET with backoff.

    Catches RequestException rather than only ConnectionError: raise_for_status
    throws HTTPError, which is a sibling of ConnectionError, not a subclass, so
    4xx/5xx used to bypass the retry loop entirely (issue BUG-06).

    The API key travels in `params`, never in `url`, so it cannot leak into the
    exception message and from there into the logs (issue SEC-04).
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=(3.05, 10))
            response.raise_for_status()
            return response

        except requests.exceptions.HTTPError as err:
            status = err.response.status_code
            # A 4xx means the request itself is wrong and retrying cannot help.
            # 429 is the exception - it means "slow down", so it IS retryable.
            if 400 <= status < 500 and status != 429:
                raise
            last_error = err

        except requests.exceptions.RequestException as err:
            last_error = err

        if attempt < max_retries - 1:
            time.sleep(2**attempt)

    raise requests.exceptions.RequestException(
        f"Failed to reach {url} after {max_retries} attempts"
    ) from last_error


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_poster_url(web_movie_id):
    """Ask TMDB for one poster. Raises on a network failure - see posterfetcher.

    TMDB returns "poster_path": null for some titles; concatenating that raises
    TypeError and used to take down the whole row (issue BUG-05). A null path is
    a real, cacheable answer, so it returns the placeholder sentinel rather than
    raising.

    Letting network errors propagate is what keeps them OUT of this cache:
    @st.cache_data stores returned values, never raised exceptions. So a
    successful lookup is remembered for a day, and a failure is not remembered
    here at all.
    """
    url = f"https://api.themoviedb.org/3/movie/{web_movie_id}"
    response = make_request_with_retry(url, params={"api_key": TMDB_API_KEY})
    path = response.json().get("poster_path")
    return POSTER_BASE + path if path else PLACEHOLDER_POSTER


@st.cache_data(ttl=POSTER_RETRY_AFTER, show_spinner=False)
def posterfetcher(web_movie_id):
    """Return a poster URL, or a sentinel saying which kind of nothing it is.

    This NEVER raises, and that matters: Streamlit reruns the whole script on
    every widget interaction, so if a failure propagated, each Save, Remove and
    theme toggle would re-attempt all ten fetches with retries and backoff -
    about 18s of frozen UI per click on a network that cannot reach TMDB.

    The two caches do different jobs, which is the whole point of splitting
    them. The inner one holds SUCCESSES for a day. This one holds whatever came
    back - including a failure - for POSTER_RETRY_AFTER seconds only. So a burst
    of reruns during an outage costs one attempt per film per minute rather than
    ten attempts per click, and a blip that clears in 30 seconds does not leave
    the grid looking broken until tomorrow. When the outer entry expires on a
    healthy network the inner cache answers instantly, with no request.
    """
    try:
        return fetch_poster_url(web_movie_id)
    except requests.exceptions.RequestException:
        return UNAVAILABLE_POSTER


def fetch_posters(movie_ids):
    """Fetch posters in parallel, degrading to the placeholder individually.

    Serial fetching made ten results roughly twice as slow as five (PERF-03),
    and one failed poster used to discard all ten recommendations (BUG-12).
    """

    def safe_fetch(movie_id):
        try:
            return posterfetcher(movie_id)
        except requests.exceptions.RequestException:
            # Belt and braces: posterfetcher already swallows these. Kept so a
            # future edit that lets one escape cannot lose the other nine.
            return UNAVAILABLE_POSTER

    # One worker per poster, so all ten resolve in a single wave. With five
    # workers they ran as two batches, which doubles the wait whenever TMDB is
    # slow or unreachable - and that wait is exactly when the UI feels stuck.
    workers = max(1, min(len(movie_ids), RESULT_COUNT))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(safe_fetch, movie_ids))


# --------------------------------------------------------------------------
# Recommender
# --------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def recommend_cached(title, count=RESULT_COUNT):
    """Cached wrapper keyed on the film title. Returns films WITHOUT posters.

    Without this, every rerun recomputes the whole ranking - and a rerun happens
    on each Save, Remove and theme toggle.

    Posters are deliberately NOT part of what is cached here. They used to be,
    and that quietly outranked the poster cache: a search made during a TMDB
    outage baked the failure sentinels into this entry for a full hour, so the
    grid stayed broken long after the network recovered. The ranking is stable
    for an hour; a poster's availability is not. Each is now cached on its own
    terms - see posterfetcher.
    """
    movies, similarity = load_artifacts()
    return recommend(movies, similarity, title, count)


def attach_posters(films):
    """Resolve a poster for each film. Cheap on reruns - posterfetcher caches."""
    if not films:
        return []
    posters = fetch_posters([film["movie_id"] for film in films])
    return [dict(film, poster=poster) for film, poster in zip(films, posters)]


def recommend(movies, similarity, title, count=RESULT_COUNT):
    """Return the `count` most similar films to `title`.

    Uses a positional lookup rather than the label index. The two coincide now
    that the notebook calls reset_index(), but the label form silently returned
    another film's recommendations for 44.6% of the catalogue before that fix
    (issue BUG-01).
    """
    matches = movies.index[movies["title"] == title]
    if len(matches) == 0:
        return []

    position = movies.index.get_loc(matches[0])
    scores = similarity[position]

    # Rank by score, skipping the query film itself. Filtering by index is
    # safer than slicing [1:n] - that assumed the self-match always lands
    # at rank 0, which stopped being true when duplicate titles existed.
    ranked = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
    picks = [j for j in ranked if j != position][:count]

    rows = movies.iloc[picks]

    results = []
    for _, row in rows.iterrows():
        year = str(row.get("release_date", ""))[:4]
        results.append(
            {
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "year": year,
            }
        )
    return results


# --------------------------------------------------------------------------
# Watch Later
# --------------------------------------------------------------------------
# The list is held in the URL as ?saved=19995,285,206647 so it survives a
# refresh, can be bookmarked, and can be shared by sending the link. No
# database and no accounts.
#
# A local JSON file was considered and rejected: Streamlit Cloud's filesystem
# is shared between all visitors and wiped on redeploy, so every user would
# see and edit the same list.


def read_film():
    """The film currently being searched, held in the URL.

    Session state is NOT usable for this. The theme toggle is an <a> link and
    saving a film rewrites the query string - both are full navigations, which
    start a new Streamlit session and wipe session state. Keeping the search in
    the URL means it survives every one of those, plus a refresh.
    """
    film = st.query_params.get("film", "")
    return film if film else None


def write_film(title):
    if title:
        st.query_params["film"] = title
    elif "film" in st.query_params:
        del st.query_params["film"]


def encode_watchlist(ids):
    """Pack TMDB ids into a compact, URL-safe string.

    The old format was decimal ids joined by commas. That is wasteful twice
    over: a 6-digit id costs 6 characters to carry 19 bits of information, and
    every comma percent-encodes to a 3-character "%2C". Ten films came to a
    123-character URL.

    Here each id is written as a fixed ID_BITS field in one big integer, then
    base64url-encoded. Ten films come to 76 characters, which matters because
    the share QR's density is driven by URL length - the old format needed a
    version-8 symbol at ten films, this one needs version 5.

    TMDB ids are used rather than row numbers in movies_df on purpose. Row
    numbers would be 13 bits instead of 19, but they are only meaningful for
    one exact build of movie_list.pkl.gz: re-export the catalogue in a
    different order and every link ever shared would quietly resolve to the
    wrong films. TMDB ids are external and permanent, so a shared list keeps
    working across retrains, and a film dropped from the catalogue is simply
    skipped by lookup_films instead of poisoning the whole list.
    """
    if not ids:
        return ""
    packed = 0
    for movie_id in ids:
        packed = (packed << ID_BITS) | (int(movie_id) & (2**ID_BITS - 1))
    width = (len(ids) * ID_BITS + 7) // 8
    raw = packed.to_bytes(width, "big")
    # Strip "=" padding: it is legal in a query value but urlencodes to %3D in
    # some clients, which would put the wasted characters straight back.
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_watchlist(token):
    """Unpack encode_watchlist output. Returns [] on anything malformed.

    A share link is user-editable text, so every failure mode here is
    reachable by hand. Bad input must degrade to an empty list rather than
    raising - a typo'd URL should show an empty Watch Later, not a traceback.
    """
    if not isinstance(token, str) or not token:
        return []

    # The alphabet is checked by hand, deliberately. base64 has a validate=
    # flag for exactly this, but urlsafe_b64decode does not accept it - it
    # takes only the string. Passing it raises TypeError, and an "except
    # (ValueError, TypeError)" here swallowed that and made EVERY token,
    # valid ones included, decode to an empty list.
    #
    # Without the check, decoding is lenient: it silently discards characters
    # outside the alphabet, so a mangled "not!!valid" becomes "notvalid" and
    # yields real-looking ids rather than nothing.
    if not _B64URL_ALPHABET.issuperset(token):
        return []

    try:
        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
    except ValueError:  # binascii.Error subclasses ValueError
        return []

    count = len(raw) * 8 // ID_BITS
    if not count:
        return []
    packed = int.from_bytes(raw, "big")
    mask = 2**ID_BITS - 1
    ids = []
    for position in range(count - 1, -1, -1):
        movie_id = (packed >> (position * ID_BITS)) & mask
        # Zero is not a valid TMDB id; it only shows up as leading padding
        # when the byte count did not divide evenly by ID_BITS.
        if movie_id and movie_id not in ids:
            ids.append(movie_id)
    return ids


def read_watchlist():
    """Read the saved films from the URL, newest format first.

    "saved" is the original comma-separated format. It is still read so links
    shared before the switch keep working; write_watchlist replaces it with
    the compact "s" form on the next change.
    """
    token = st.query_params.get("s", "")
    if token:
        return decode_watchlist(token)

    ids = []
    for chunk in st.query_params.get("saved", "").split(","):
        chunk = chunk.strip()
        if chunk.isdigit():
            movie_id = int(chunk)
            if movie_id not in ids:
                ids.append(movie_id)
    return ids


def write_watchlist(ids):
    """Persist the list back to the URL, or clear the params when empty."""
    if ids:
        st.query_params["s"] = encode_watchlist(ids)
    elif "s" in st.query_params:
        del st.query_params["s"]
    # Always drop the legacy key, so a link that arrived in the old format is
    # upgraded rather than leaving both to drift apart.
    if "saved" in st.query_params:
        del st.query_params["saved"]


def toggle_saved(movie_id):
    """Add or remove a film, then rerun.

    Assigning to st.query_params updates the address bar but does NOT trigger
    a rerun on its own - verified in the browser: the URL changed to view=grid
    while the DOM kept rendering list tiles. Without the explicit rerun the
    page and the URL drift out of sync.
    """
    ids = read_watchlist()
    if movie_id in ids:
        ids.remove(movie_id)
    else:
        ids.append(movie_id)
    write_watchlist(ids)
    st.rerun()


def share_url(ids):
    """Absolute URL that reproduces this watchlist for someone else.

    st.context.url is the browser's address, so this works unchanged on
    localhost, on a LAN address and on Community Cloud. It can be missing when
    there is no live browser session (a bare script run, some embeds), hence
    the fallback to the published address.

    The link deliberately carries only the list and the layout. "theme" is
    left off so the recipient keeps their own light/dark choice, and "film" is
    left off so they land on the shared list rather than on whatever happened
    to be in the search box.
    """
    try:
        origin = st.context.url or FALLBACK_ORIGIN
    except Exception:
        origin = FALLBACK_ORIGIN

    parts = urlsplit(origin)
    if not parts.scheme or not parts.netloc:
        parts = urlsplit(FALLBACK_ORIGIN)

    query = f"s={quote(encode_watchlist(ids), safe='')}"
    if current_view() == "grid":
        query += "&view=grid"
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", query, ""))


def share_qr_svg(url):
    """Render the share link as an inline SVG QR.

    The colours are FIXED, not taken from the active palette, and that is the
    whole point of this comment. Theming the QR was the obvious thing to do -
    cream modules on the near-black background in dark mode - and it produces
    an inverted symbol. The QR spec assumes dark modules on a light field;
    invert it and a large share of scanners simply will not lock on. Measured
    with OpenCV's detector on the exact palettes in THEMES: the light-mode
    symbol decoded, the dark-mode one did not, and inverting that same image
    made it decode again, which isolates polarity as the only cause.

    So it stays dark-on-light in both themes. The warm off-white keeps it from
    reading as a stark white rectangle in the dark layout, which was the real
    concern behind theming it in the first place.

    SVG rather than PNG so it stays sharp at any size and any zoom level.
    """
    buffer = io.BytesIO()
    segno.make(url, error="m").save(
        buffer,
        kind="svg",
        # Quiet zone. The spec asks for 4 modules; below that many phone
        # scanners will not lock on at all.
        border=4,
        dark=QR_DARK,
        light=QR_LIGHT,
        # Let CSS size it, and drop the XML prolog so it can be inlined
        # directly into an st.markdown block.
        svgversion=None,
        xmldecl=False,
        omitsize=True,
    )
    return buffer.getvalue().decode("utf-8")


def lookup_films(movie_ids):
    """Rebuild title/year/poster for saved ids, preserving the saved order."""
    if not movie_ids:
        return []

    by_id = movies_df.set_index("movie_id")
    known = [i for i in movie_ids if i in by_id.index]
    if not known:
        return []

    posters = fetch_posters(known)
    films = []
    for movie_id, poster in zip(known, posters):
        row = by_id.loc[movie_id]
        if isinstance(row, pd.DataFrame):  # duplicate ids, take the first
            row = row.iloc[0]
        films.append(
            {
                "movie_id": movie_id,
                "title": row["title"],
                "year": str(row.get("release_date", ""))[:4],
                "poster": poster,
            }
        )
    return films


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

inject_theme()
load_stylesheet()

movies_df, similarity_matrix = load_artifacts()


def render_masthead():
    """Brand, social links and theme toggle - all in ONE flex row.

    The toggle is a fourth <a> inside .wtw-links deliberately. It was briefly
    moved into its own Streamlit column to avoid the page reload a link
    causes, but that put it in a different centring context from the icons
    beside it and left it sitting 8px high. Same flex row is the only way to
    guarantee they line up.

    Trade-off accepted: clicking the toggle is a real navigation, so there is
    a brief flash as the page reloads and repaints.
    """
    st.markdown(
        f"""
        <div class="wtw-masthead">
          <div class="wtw-brand">
            <div class="wtw-mark">{ICONS['play']}</div>
            <div>
              <div class="wtw-title">WHAT TO WATCH NEXT</div>
              <div class="wtw-kicker">NOW SHOWING &middot; {len(movies_df):,} FILMS</div>
            </div>
          </div>
          <div class="wtw-links">
            <a href="https://github.com/xxwizardxx117" target="_blank" rel="noopener"
               aria-label="GitHub">{ICONS['github']}</a>
            <a href="https://www.linkedin.com/in/sujal-sharma-731758238/"
               target="_blank" rel="noopener"
               aria-label="LinkedIn">{ICONS['linkedin']}</a>
            <a href="https://twitter.com/Xx_Sujal_xX" target="_blank" rel="noopener"
               aria-label="X">{ICONS['x']}</a>
            <a href="{theme_toggle_href()}" target="_self" class="wtw-toggle"
               aria-label="Switch between light and dark"
               title="{'Matinee' if current_theme() == 'dark' else 'Evening show'}"
               >{ICONS['sun'] if current_theme() == 'dark' else ICONS['moon']}</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_masthead()

with st.container(key="page_split"):
    main_col, panel_col = st.columns([4, 1], gap="large")

with main_col:
    st.markdown('<div class="wtw-label">PICK A FILM</div>', unsafe_allow_html=True)

    picker_col, button_col = st.columns([4, 1], gap="small")

    titles = list(movies_df["title"])
    from_url = read_film()

    # Seed the picker from the URL ONCE per session, then let the widget own
    # its value.
    #
    # Passing index= on every rerun is what broke typing: Streamlit re-applies
    # that index each run, so a title typed while a search was still in flight
    # got reset back to whatever the URL said. Clicking an option happened to
    # commit before the reset landed, which is why the mouse worked and Enter
    # did not. With a key, session state wins and index= is only the initial
    # default - so a fresh load (or a theme toggle, which is a real navigation)
    # still restores from the URL.
    if "film_picker" not in st.session_state:
        st.session_state["film_picker"] = from_url if from_url in titles else None

    with picker_col:
        selected = st.selectbox(
            "Film",
            titles,
            index=None,
            key="film_picker",
            placeholder="Search the catalogue",
            label_visibility="collapsed",
        )

    with button_col:
        go = st.button("Recommend", type="primary", use_container_width=True)

    # Search as soon as a film is chosen. Streamlit reruns when the selectbox
    # value changes - on Enter or on click - so comparing against the URL is
    # enough to fire automatically. The button stays for people who expect to
    # press something.
    if go and not selected:
        st.warning("Pick a film first.")

    if selected and selected != from_url:
        write_film(selected)
        from_url = selected

    results, results_for = [], selected

    # A placeholder for the idle state. Streamlit keeps the PREVIOUS render on
    # screen (dimmed) while a script runs, so writing the intro unconditionally
    # left it sitting behind the spinner, blurred but legible. Clearing this
    # slot before the fetch removes it immediately, so the spinner appears on
    # its own and the results fade in cleanly.
    intro_slot = st.empty()

    if selected:
        intro_slot.empty()
        with st.spinner("Rolling the projector..."):
            results = attach_posters(recommend_cached(selected))
    elif not go:
        intro_slot.markdown(
            '<div class="wtw-rule"><span>PICK A FILM TO BEGIN</span></div>'
            '<div class="wtw-hint wtw-hint-centred">Choose any title above and '
            "we will find ten films with a similar cast, crew, genre and "
            "premise.</div>",
            unsafe_allow_html=True,
        )

    if results:
        st.markdown(
            f'<div class="wtw-rule"><span>{len(results)} FILMS LIKE '
            f"{results_for.upper()}</span></div>",
            unsafe_allow_html=True,
        )

        # Say so when posters are missing because TMDB is unreachable, rather
        # than leaving the user to guess whether these films simply have no
        # artwork. The recommendations themselves never depend on TMDB - they
        # come from the local model - and that is the part worth stating.
        unreachable = sum(1 for f in results if f["poster"] == UNAVAILABLE_POSTER)
        if unreachable:
            st.markdown(
                '<div class="wtw-notice">'
                f"Couldn't load {unreachable} of {len(results)} posters - TMDB "
                "isn't responding. The recommendations below are unaffected, "
                "and the posters retry on their own."
                "</div>",
                unsafe_allow_html=True,
            )

        saved_ids = read_watchlist()

        for row_start in range(0, len(results), GRID_COLUMNS):
            row = results[row_start : row_start + GRID_COLUMNS]
            for column, film in zip(st.columns(GRID_COLUMNS), row):
                with column:
                    st.image(poster_src(film["poster"]), use_container_width=True)
                    st.markdown(
                        f'<div class="wtw-caption">'
                        f'<div class="wtw-name">{film["title"]}</div>'
                        f'<div class="wtw-year">{film["year"]}</div></div>',
                        unsafe_allow_html=True,
                    )
                    already = film["movie_id"] in saved_ids
                    # Deliberately NOT disabled when already saved. toggle_saved
                    # removes as readily as it adds, so disabling it made the
                    # saved state a dead end - you could add a film but not
                    # take it back from the same button. Primary styling
                    # carries the "on" state instead.
                    if st.button(
                        "✓ Saved" if already else "+ Watch later",
                        key=f"save_{film['movie_id']}",
                        use_container_width=True,
                        type="primary" if already else "secondary",
                        help=(
                            "Remove from Watch Later"
                            if already
                            else "Add to Watch Later"
                        ),
                    ):
                        toggle_saved(film["movie_id"])

with panel_col:
    # Resolved before the header is drawn because the share control only
    # appears once there is something worth sharing.
    watchlist = lookup_films(read_watchlist())

    # Title and share button share one row. The title stays an HTML block
    # rather than becoming an st.subheader so it keeps the Bebas Neue face and
    # letter-spacing the rest of the design uses.
    with st.container(key="panel_head"):
        title_col, share_col = st.columns(
            [4, 1], gap="small", vertical_alignment="center"
        )
        with title_col:
            st.markdown(
                '<div class="wtw-panel-title">WATCH LATER</div>',
                unsafe_allow_html=True,
            )
        with share_col:
            if watchlist:
                link = share_url([f["movie_id"] for f in watchlist])
                with st.popover(
                    "",
                    icon=":material/ios_share:",
                    help="Share this watchlist",
                    use_container_width=True,
                    key="share_pop",
                ):
                    st.markdown(
                        '<div class="wtw-share-head">SCAN TO OPEN</div>'
                        f'<div class="wtw-qr">{share_qr_svg(link)}</div>',
                        unsafe_allow_html=True,
                    )
                    # st.code carries a copy-to-clipboard button of its own,
                    # which is why this is not a plain st.text: the clipboard
                    # API is unreachable from an st.markdown block.
                    st.code(link, language=None, wrap_lines=True)
                    st.markdown(
                        '<div class="wtw-share-note">'
                        f"{len(watchlist)} films &middot; opens in their theme"
                        "</div>",
                        unsafe_allow_html=True,
                    )

    st.markdown('<div class="wtw-panel-underline"></div>', unsafe_allow_html=True)

    if watchlist:
        view = current_view()

        # Fixed header - count on the left, view switch on the right.
        #
        # These are real Streamlit BUTTONS, not links. An <a href> is a full
        # page navigation, which starts a new session, resets the picker and
        # made the whole page reload on every view change. Buttons only trigger
        # a rerun, so the search survives. The view is still written to the URL
        # so it stays shareable and survives a refresh.
        label_col, list_col, grid_col = st.columns(
            [3, 1, 1], gap="small", vertical_alignment="center"
        )

        with label_col:
            st.markdown(
                f'<div class="wtw-label wtw-label-inline">{len(watchlist)} '
                "SAVED</div>",
                unsafe_allow_html=True,
            )

        with list_col:
            if st.button(
                "☰",
                key="view_list",
                help="List view",
                type="primary" if view == "list" else "secondary",
                use_container_width=True,
            ):
                st.query_params["view"] = "list"
                st.rerun()

        with grid_col:
            if st.button(
                "▦",
                key="view_grid",
                help="Grid view",
                type="primary" if view == "grid" else "secondary",
                use_container_width=True,
            ):
                st.query_params["view"] = "grid"
                st.rerun()

        # st.container(height=...) gives a genuinely scrollable region without
        # hand-rolled CSS or nested-scroll hacks: Streamlit sets overflow on
        # the block wrapper itself. The saved films live in here; the header
        # above and the hint below stay outside, so they never scroll away.
        # key= gives the element a stable .st-key-saved_scroll class, which
        # style.css uses to override this pixel height with a calc() so the
        # panel reaches the footer instead of stopping short on tall windows.
        with st.container(
            height=SAVED_LIST_HEIGHT, border=False, key="saved_scroll"
        ):

            def saved_card(film, compact=False):
                st.image(poster_src(film["poster"]), use_container_width=True)
                st.markdown(
                    f'<div class="wtw-caption{" wtw-compact" if compact else ""}">'
                    f'<div class="wtw-name">{film["title"]}</div>'
                    f'<div class="wtw-year">{film["year"]}</div></div>',
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Remove",
                    key=f"drop_{film['movie_id']}",
                    use_container_width=True,
                ):
                    toggle_saved(film["movie_id"])
                st.markdown(
                    '<div style="height:0.9rem"></div>', unsafe_allow_html=True
                )

            if view == "grid":
                # Wrapped in a keyed container so style.css can target grid
                # cards separately from list rows. Both views produce a
                # stHorizontalBlock, but it means different things: in list
                # view it is ONE film, in grid view it is a row of two. Styling
                # that element blindly gave both grid films a single shared
                # card. The key lets each view get its own card selector.
                with st.container(key="saved_grid"):
                    # Chunked into rows rather than dealt into two long
                    # columns, so cards stay aligned when titles wrap.
                    for start in range(0, len(watchlist), 2):
                        pair = watchlist[start : start + 2]
                        for column, film in zip(st.columns(2, gap="small"), pair):
                            with column:
                                saved_card(film, compact=True)
            else:
                # A real list tile: small thumbnail, title and year, then a
                # remove control on the right edge - not a vertical stack of
                # full-size posters, which is just the grid one column wide.
                with st.container(key="saved_list"):
                    for film in watchlist:
                        thumb, meta, action = st.columns(
                            [1, 2.6, 0.8], gap="small", vertical_alignment="center"
                        )
                        with thumb:
                            st.image(
                                poster_src(film["poster"]), use_container_width=True
                            )
                        with meta:
                            st.markdown(
                                f'<div class="wtw-tile">'
                                f'<div class="wtw-tile-name">{film["title"]}</div>'
                                f'<div class="wtw-tile-year">{film["year"]}</div>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        with action:
                            if st.button(
                                "✕",
                                key=f"del_{film['movie_id']}",
                                help="Remove from Watch Later",
                            ):
                                toggle_saved(film["movie_id"])

        # Pinned under the scroll region, always visible.
        st.markdown(
            '<div class="wtw-hint wtw-hint-pinned">This list lives in the page '
            "address, so it survives a refresh. Bookmark or share the URL to "
            "keep it.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            # The hint sits INSIDE the dashed rail. Left outside, it gets
            # pushed to the very bottom of the page by the rail's height and
            # reads as orphaned footer text.
            f'<div class="wtw-empty">{ICONS["ticket"]}'
            "<div>Your list is empty</div>"
            '<div class="wtw-hint">Search for a film, then hit '
            "<b>+ Watch later</b> under any result to pin it here.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
