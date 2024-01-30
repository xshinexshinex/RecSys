import os
import pandas as pd
import numpy as np
import hashlib
from catboost import CatBoostClassifier
from fastapi import FastAPI
from schema import PostGet, Response
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from dotenv import load_dotenv

# Loading environment variables
load_dotenv()
url_object = URL.create(
    drivername="postgresql",
    username=os.environ.get("POSTGRES_USER"),
    password=os.environ.get("POSTGRES_PASSWORD"),
    host=os.environ.get("POSTGRES_HOST"),
    port=os.environ.get("POSTGRES_PORT"),
    database=os.environ.get("POSTGRES_DATABASE")
    )

# Create the app
app = FastAPI()


# Function for downloading data in batches
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(url_object)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


# Function for loading models inside the LMS
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        if path == "C:/Users/masli/pys/ML/Task_22 РЕКОМЕНДАТЕЛЬНЫЕ " \
                   "СИСТЕМЫ/Final_project/models/control_catboost_model":
            MODEL_PATH = '/workdir/user_input/models/model_control'
        elif path == "C:/Users/masli/pys/ML/Task_22 РЕКОМЕНДАТЕЛЬНЫЕ " \
                     "СИСТЕМЫ/Final_project/test_catboost_model":
            MODEL_PATH = '/workdir/user_input/model_test'
    else:
        MODEL_PATH = path
    return MODEL_PATH


# Function for loading models
def load_models():
    from_file_1, from_file_2 = CatBoostClassifier(), CatBoostClassifier()

    # Loading the control model
    model_control = from_file_1.load_model(
        get_model_path(
            "C:/Users/masli/pys/ML/Task_22 РЕКОМЕНДАТЕЛЬНЫЕ "
            "СИСТЕМЫ/Final_project/venv/control_catboost_model"
            )
        )

    # Loading the test model
    model_test = from_file_2.load_model(
        get_model_path(
            "C:/Users/masli/pys/ML/Task_22 РЕКОМЕНДАТЕЛЬНЫЕ "
            "СИСТЕМЫ/Final_project/venv/test_catboost_model"
            )
        )
    return model_control, model_test


# Function for loading data
def load_features():
    liked_posts_query = """
        SELECT distinct post_id, user_id
        FROM public.feed_data
        WHERE action = 'like' """

    liked_posts = batch_load_sql(liked_posts_query)

    posts_features = batch_load_sql(
        "SELECT * FROM a_maslennikov_post_features_lesson_4_22"
        )

    user_features = batch_load_sql(
        "SELECT * FROM public.user_data"
        )
    return [liked_posts, posts_features, user_features]


# Constants for get_exp_group()
salt = "salty_chips"
split_percentage = 50  # Split percentage (50/50)


# Defining a group for a user based on user_id
def get_exp_group(user_id: int) -> str:
    data = f"{user_id}{salt}".encode()
    md5_hash = hashlib.md5(data).hexdigest()
    hash_as_int = int(md5_hash, 16)
    if hash_as_int % 100 < split_percentage:
        return "control"
    else:
        return "test"


# Loading control and test models
model_control, model_test = load_models()

# Features for test model
features = load_features()

# Features for control model
post_text_df = batch_load_sql(
    'SELECT * FROM a_maslennikov_post_features_lesson_22'
    )
raw_post_text = batch_load_sql('SELECT * FROM public.post_text_df')
df = features[2]


# Function for prediction by control model
def get_recommended_feed_control(id: int, time: datetime, limit: int = 5):
    numeric_columns = ['gender', 'age', 'exp_group', 'count_words', 'month',
                       'TextCluster', 'day_of_week', 'hour']

    # Create dataframe with features for given user
    df_to_pred = df[df["user_id"] == id].drop("user_id", axis=1)
    df_to_pred["month"] = time.month
    df_to_pred["day_of_week"] = time.weekday()
    df_to_pred['hour'] = time.hour

    liked_posts = features[0]

    # Creating a dataframe with posts for recommendations
    posts_to_rec = post_text_df[
        ~post_text_df.index.isin(
            (liked_posts[liked_posts.user_id == id]
             .drop("user_id", axis=1)
             .sort_values(by="post_id").set_index("post_id")
             ).index
            )
    ]

    # Carrying out the necessary manipulations
    df_to_pred = pd.concat(
        [df_to_pred, posts_to_rec.drop("post_id", axis=1)], axis=1
        ) \
        .fillna(method='ffill').dropna()
    df_to_pred[numeric_columns] = df_to_pred[numeric_columns].astype(np.int16)

    # Creating a response dataframe
    resp_df = pd.DataFrame(
        np.isin(
            model_control.predict_proba(df_to_pred)[:, 1],
            sorted(
                model_control.predict_proba(df_to_pred)[:, 1],
                reverse=True
                )[:limit]
            ),
        columns=["response"]
        )

    resp_df = resp_df.loc[resp_df.response, :][0:limit]

    return raw_post_text.loc[resp_df.index.to_list()].rename(
        columns={"post_id": "id"}
        ).to_dict("records")


# Function for prediction by test model
def get_recommended_feed_test(id: int, time: datetime, limit: int = 5):
    user_features = features[2].loc[features[2].user_id == id]
    user_features = user_features.drop("user_id", axis=1)

    posts_features = features[1].drop(["index", "text"], axis=1)
    content = features[1][["post_id", "text", "topic"]]

    add_user_features = dict(
        zip(user_features.columns, user_features.values[0])
        )
    user_posts_features = posts_features.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index("post_id")

    user_posts_features["hour"] = time.hour
    user_posts_features["month"] = time.month

    cols_ = [
        'hour', 'month', 'gender', 'age', 'country', 'city', 'exp_group', 'os',
        'source', 'topic',
        'TextCluster', 'DistanceToCluster_0', 'DistanceToCluster_1',
        'DistanceToCluster_2',
        'DistanceToCluster_3', 'DistanceToCluster_4', 'DistanceToCluster_5',
        'DistanceToCluster_6',
        'DistanceToCluster_7', 'DistanceToCluster_8', 'DistanceToCluster_9',
        'DistanceToCluster_10',
        'DistanceToCluster_11', 'DistanceToCluster_12', 'DistanceToCluster_13',
        'DistanceToCluster_14',
        'TotalTfIdf'
        ]

    user_posts_features = user_posts_features[cols_]

    predicts = model_test.predict_proba(user_posts_features)[:, 1]
    user_posts_features["predicts"] = predicts

    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_posts_features[
        ~user_posts_features.index.isin(liked_posts)]

    recommended_posts = filtered_.sort_values("predicts")[-limit:].index

    return [
        PostGet(
            **{
                "id": i,
                "text": content[content.post_id == i].text.values[0],
                "topic": content[content.post_id == i].topic.values[0]
                }
            ) for i in recommended_posts
        ]


# Endpoint that returns recommendations and experimental group
@app.get("/post/recommendations", response_model=Response)
def recommended_posts(id: int, time: datetime, limit: int = 5) -> Response:
    exp_group = get_exp_group(id)
    if exp_group == 'control':
        recommendations = get_recommended_feed_control(id, time, limit)
        return {'exp_group': exp_group, 'recommendations': recommendations}
    elif exp_group == 'test':
        recommendations = get_recommended_feed_test(id, time, limit)
        return {'exp_group': exp_group, 'recommendations': recommendations}
    else:
        raise ValueError('unknown group')
