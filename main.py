import os
import pickle

import numpy as np
import pandas as pd
import requests

from flask import Flask, render_template, request, url_for

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

from tmdbv3api import TMDb, Movie


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# Load TMDB_API_KEY from the project's .env file before reading environment variables.
load_dotenv()


# ============================================================
# TMDB CONFIGURATION
# ============================================================

tmdb = TMDb()

# IMPORTANT:
# Do not publish your real API key on GitHub.
# Prefer setting it as an environment variable.
TMDB_API_KEY = os.environ.get(
    "TMDB_API_KEY",
    ""
)

tmdb.api_key = TMDB_API_KEY

TMDB_DETAILS_URL = "https://api.themoviedb.org/3/movie/{}"
TMDB_REVIEWS_URL = "https://api.themoviedb.org/3/movie/{}/reviews"

IMAGE_BASE_URL = "https://image.tmdb.org/t/p/original"


# ============================================================
# REQUEST SESSION
# ============================================================

tmdb_session = requests.Session()

retry_strategy = Retry(
    total=2,
    backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("GET",)
)

adapter = HTTPAdapter(max_retries=retry_strategy)

tmdb_session.mount("https://", adapter)
tmdb_session.mount("http://", adapter)


# ============================================================
# SENTIMENT MODEL
# ============================================================

sentiment_model = None

MODEL_FILE = "sentiment_model.pkl"

if os.path.exists(MODEL_FILE):
    try:
        with open(MODEL_FILE, "rb") as file:
            sentiment_model = pickle.load(file)

        print("Sentiment model loaded successfully.")

    except Exception as error:
        print("Could not load sentiment model:")
        print(error)

        sentiment_model = None

else:
    print(
        "WARNING: sentiment_model.pkl not found. "
        "Sentiment analysis will be disabled."
    )


# ============================================================
# GLOBAL RECOMMENDATION DATA
# ============================================================

data = None
similarity_matrix = None


# ============================================================
# TMDB FUNCTIONS
# ============================================================

def get_tmdb_details(movie_id):
    """
    Fetch complete movie details from TMDb.
    """

    response = tmdb_session.get(
        TMDB_DETAILS_URL.format(movie_id),
        params={
            "api_key": TMDB_API_KEY,
            "language": "en-US"
        },
        timeout=10
    )

    response.raise_for_status()

    return response.json()


def get_tmdb_reviews(movie_id):
    """
    Fetch movie reviews from TMDb.

    This replaces IMDb scraping.
    """

    response = tmdb_session.get(
        TMDB_REVIEWS_URL.format(movie_id),
        params={
            "api_key": TMDB_API_KEY,
            "language": "en-US",
            "page": 1
        },
        timeout=10
    )

    response.raise_for_status()

    return response.json()


def search_tmdb_movie(movie_title):
    """
    Search TMDb for a movie.
    """

    movie_api = Movie()

    try:
        results = movie_api.search(movie_title)

        if not results:
            return None

        return results[0]

    except Exception as error:
        print("TMDb movie search error:", error)
        return None


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def predict_sentiment(review):
    """
    Predict sentiment of a review.

    Returns:
        Good
        Bad
        Unknown
    """

    if sentiment_model is None:
        return "Unknown"

    try:
        prediction = sentiment_model.predict([review])[0]

        # Handle numeric labels
        if isinstance(prediction, (int, np.integer, float, np.floating)):
            return "Good" if int(prediction) == 1 else "Bad"

        # Handle string labels
        prediction_string = str(prediction).lower()

        if prediction_string in [
            "1",
            "positive",
            "pos",
            "good",
            "true"
        ]:
            return "Good"

        if prediction_string in [
            "0",
            "negative",
            "neg",
            "bad",
            "false"
        ]:
            return "Bad"

        return str(prediction)

    except Exception as error:
        print("Sentiment prediction error:", error)
        return "Unknown"


# ============================================================
# RECOMMENDATION SYSTEM
# ============================================================

def create_similarity_matrix():
    """
    Create movie similarity matrix from main_data.csv.
    """

    global data
    global similarity_matrix

    data = pd.read_csv("main_data.csv")

    # Make sure required columns exist
    if "movie_title" not in data.columns:
        raise ValueError(
            "main_data.csv must contain a 'movie_title' column."
        )

    if "comb" not in data.columns:
        raise ValueError(
            "main_data.csv must contain a 'comb' column."
        )

    data["movie_title"] = (
        data["movie_title"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    data["comb"] = (
        data["comb"]
        .fillna("")
        .astype(str)
    )

    vectorizer = CountVectorizer()

    count_matrix = vectorizer.fit_transform(data["comb"])

    similarity_matrix = cosine_similarity(count_matrix)

    print("Recommendation model loaded.")
    print("Movies:", len(data))


def rcmd(movie_title):
    """
    Return 10 recommended movies.
    """

    global data
    global similarity_matrix

    if data is None or similarity_matrix is None:
        create_similarity_matrix()

    movie_title = movie_title.strip().lower()

    if not movie_title:
        return "Please enter a movie name."

    if movie_title not in data["movie_title"].values:
        return (
            "Sorry! The movie you searched is not in our database. "
            "Please check the spelling or try another movie."
        )

    movie_index = data.index[
        data["movie_title"] == movie_title
    ][0]

    scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove the searched movie itself
    scores = scores[1:11]

    recommendations = []

    for index, score in scores:
        title = data.iloc[index]["movie_title"]
        recommendations.append(title)

    return recommendations


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def list_of_genres(genre_json):
    """
    Convert TMDb genre list into a string.
    """

    if not genre_json:
        return "N/A"

    genres = []

    for genre in genre_json:
        if isinstance(genre, dict) and "name" in genre:
            genres.append(genre["name"])

    return ", ".join(genres) if genres else "N/A"


def date_convert(date_string):
    """
    Convert YYYY-MM-DD into Month DD YYYY.
    """

    if not date_string:
        return "N/A"

    try:
        date = pd.to_datetime(date_string)

        return date.strftime("%B %d %Y")

    except Exception:
        return "N/A"


def mins_to_hours(duration):
    """
    Convert minutes to readable hours/minutes.
    """

    if not duration:
        return "N/A"

    try:
        duration = int(duration)

        hours = duration // 60
        minutes = duration % 60

        if hours == 0:
            return f"{minutes} minutes"

        if minutes == 0:
            return f"{hours} hours"

        return f"{hours} hours {minutes} minutes"

    except Exception:
        return "N/A"


def get_suggestions():
    """
    Get movie names for autocomplete.
    """

    df = pd.read_csv("main_data.csv")

    if "movie_title" not in df.columns:
        return []

    return (
        df["movie_title"]
        .dropna()
        .astype(str)
        .str.title()
        .tolist()
    )


def get_local_movie(movie_title):
    """
    Find movie in local recommendation database.
    """

    global data

    if data is None:
        create_similarity_matrix()

    matches = data[
        data["movie_title"].str.lower()
        == movie_title.strip().lower()
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


# ============================================================
# LOCAL FALLBACK
# ============================================================

def render_local_recommendations(movie):
    """
    Render recommendations even if TMDb is unavailable.
    """

    recommendations = rcmd(movie)

    if isinstance(recommendations, str):
        return render_template(
            "recommend.html",
            movie=movie.upper(),
            r=recommendations,
            t="s",
            suggestions=get_suggestions()
        )

    local_movie = get_local_movie(movie)

    if local_movie is None:
        return render_template(
            "recommend.html",
            movie=movie.upper(),
            r="Movie not found.",
            t="s",
            suggestions=get_suggestions()
        )

    # IMPORTANT:
    # SimpleNamespace is imported at the top.
    from types import SimpleNamespace

    result = SimpleNamespace(
        title=str(
            local_movie.get(
                "movie_title",
                movie
            )
        ).title(),

        overview=(
            "Movie details are unavailable while "
            "TMDb cannot be reached."
        ),

        vote_average="N/A"
    )

    genres = local_movie.get("genres", "N/A")

    return render_template(
        "recommend.html",

        movie=movie.upper(),

        mtitle=recommendations,

        t="l",

        cards={},

        result=result,

        reviews={},

        img_path=url_for(
            "static",
            filename="mrswsa1.gif"
        ),

        genres=genres,

        vote_count="N/A",

        release_date="N/A",

        status="N/A",

        runtime="N/A",

        offline=True,

        suggestions=get_suggestions()
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    suggestions = get_suggestions()

    return render_template(
        "home.html",
        suggestions=suggestions
    )


# ============================================================
# RECOMMENDATION PAGE
# ============================================================

@app.route("/recommend")
def recommend():

    movie_query = request.args.get(
        "movie",
        ""
    ).strip()

    if not movie_query:
        return render_template(
            "home.html",
            suggestions=get_suggestions()
        )

    # --------------------------------------------------------
    # STEP 1: LOCAL RECOMMENDATION
    # --------------------------------------------------------

    recommendations = rcmd(movie_query)

    if isinstance(recommendations, str):

        return render_template(
            "recommend.html",

            movie=movie_query.upper(),

            r=recommendations,

            t="s",

            suggestions=get_suggestions()
        )

    # --------------------------------------------------------
    # STEP 2: SEARCH TMDB
    # --------------------------------------------------------

    try:

        tmdb_movie = search_tmdb_movie(movie_query)

        if tmdb_movie is None:
            print("Movie not found on TMDb.")

            return render_local_recommendations(
                movie_query
            )

        movie_id = tmdb_movie.id

        # ----------------------------------------------------
        # STEP 3: GET MOVIE DETAILS
        # ----------------------------------------------------

        details = get_tmdb_details(movie_id)

        movie_title = details.get(
            "title",
            movie_query
        )

        poster_path = details.get(
            "poster_path"
        )

        if poster_path:
            img_path = (
                IMAGE_BASE_URL
                + poster_path
            )
        else:
            img_path = url_for(
                "static",
                filename="mrswsa1.gif"
            )

        # ----------------------------------------------------
        # GENRES
        # ----------------------------------------------------

        genres = list_of_genres(
            details.get(
                "genres",
                []
            )
        )

        # ----------------------------------------------------
        # RELEASE DATE
        # ----------------------------------------------------

        release_date = date_convert(
            details.get(
                "release_date"
            )
        )

        # ----------------------------------------------------
        # RATING
        # ----------------------------------------------------

        vote_average = details.get(
            "vote_average"
        )

        if vote_average is None:
            vote_average = "N/A"
        else:
            try:
                vote_average = round(
                    float(vote_average),
                    1
                )
            except Exception:
                vote_average = "N/A"

        # ----------------------------------------------------
        # VOTE COUNT
        # ----------------------------------------------------

        vote_count_value = details.get(
            "vote_count"
        )

        if vote_count_value is None:
            vote_count = "N/A"
        else:
            try:
                vote_count = "{:,}".format(
                    int(vote_count_value)
                )
            except Exception:
                vote_count = "N/A"

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        status = details.get(
            "status",
            "N/A"
        )

        # ----------------------------------------------------
        # RUNTIME
        # ----------------------------------------------------

        runtime = mins_to_hours(
            details.get(
                "runtime"
            )
        )

        # ----------------------------------------------------
        # REVIEWS
        # ----------------------------------------------------

        movie_reviews = {}

        try:

            review_data = get_tmdb_reviews(
                movie_id
            )

            review_results = review_data.get(
                "results",
                []
            )

            # Only process first 10 reviews
            for review in review_results[:10]:

                content = review.get(
                    "content",
                    ""
                ).strip()

                if not content:
                    continue

                sentiment = predict_sentiment(
                    content
                )

                movie_reviews[
                    content
                ] = sentiment

        except requests.exceptions.RequestException as error:

            print(
                "TMDb review request failed:",
                error
            )

        except Exception as error:

            print(
                "Review processing error:",
                error
            )

        # ----------------------------------------------------
        # RECOMMENDED MOVIE CARDS
        # ----------------------------------------------------

        movie_cards = {}

        for recommended_title in recommendations:

            try:

                recommended_movie = search_tmdb_movie(
                    recommended_title
                )

                if recommended_movie is None:
                    continue

                recommended_id = recommended_movie.id

                recommended_details = get_tmdb_details(
                    recommended_id
                )

                recommended_poster = (
                    recommended_details.get(
                        "poster_path"
                    )
                )

                if recommended_poster:

                    poster_url = (
                        IMAGE_BASE_URL
                        + recommended_poster
                    )

                    movie_cards[
                        poster_url
                    ] = recommended_details.get(
                        "title",
                        recommended_title
                    )

            except Exception as error:

                print(
                    "Recommended movie error:",
                    recommended_title,
                    error
                )

                # Continue with next movie
                continue

        # ----------------------------------------------------
        # RESULT OBJECT
        # ----------------------------------------------------

        from types import SimpleNamespace

        result = SimpleNamespace(

            title=movie_title,

            overview=details.get(
                "overview",
                "No overview available."
            ),

            vote_average=vote_average
        )

        # ----------------------------------------------------
        # RENDER PAGE
        # ----------------------------------------------------

        return render_template(

            "recommend.html",

            movie=movie_query.upper(),

            mtitle=recommendations,

            t="l",

            cards=movie_cards,

            result=result,

            reviews=movie_reviews,

            img_path=img_path,

            genres=genres,

            vote_count=vote_count,

            release_date=release_date,

            status=status,

            runtime=runtime,

            offline=False,

            suggestions=get_suggestions()
        )

    # ========================================================
    # TMDB ERROR
    # ========================================================

    except requests.exceptions.RequestException as error:

        print(
            "TMDb connection error:",
            error
        )

        return render_local_recommendations(
            movie_query
        )

    except Exception as error:

        print(
            "Unexpected error:",
            error
        )

        return render_local_recommendations(
            movie_query
        )


# ============================================================
# ERROR HANDLER
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(
        "home.html",
        suggestions=get_suggestions()
    ), 404


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Movie Recommendation System")
    print("=" * 60)
    print("Open: http://127.0.0.1:5000")
    print("=" * 60)

    app.run(
        debug=True
    )
