import os
from datetime import datetime
from schema import PostGet
import pandas as pd
from typing import List
from catboost import CatBoostClassifier
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from dotenv import load_dotenv


# Загрузим переменные окружения
load_dotenv()
url_object = URL.create(
    drivername="postgresql",
    username=os.environ.get("POSTGRES_USER"),
    password=os.environ.get("POSTGRES_PASSWORD"),
    host=os.environ.get("POSTGRES_HOST"),
    port=os.environ.get("POSTGRES_PORT"),
    database=os.environ.get("POSTGRES_DATABASE")
    )

app = FastAPI()  # Создадим приложение


# Функция загрузки данных частями
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(url_object)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


# Подгрузка модели в зависимости от устройтва (ЛМС или локально)
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


# Функция для загрузки модели
def load_models():
    from_file = CatBoostClassifier()
    model = from_file.load_model(
        get_model_path(
            "C:/Users/masli/pys/ML/Task_22 РЕКОМЕНДАТЕЛЬНЫЕ "
            "СИСТЕМЫ/Final_project/models/test_catboost_model"
            )
        )
    return model


# Функция для загрузки необходимых признаков
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


model = load_models()  # Загружаем модель
features = load_features()  # Загружаем признаки


# Делаем прогнозы с помощью функции ниже
def get_recommended_feed(id: int, time: datetime, limit: int = 5):
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

    cols_ = ['hour', 'month', 'gender', 'age', 'country', 'city', 'exp_group',
             'os', 'source', 'topic',
             'TextCluster', 'DistanceToCluster_0', 'DistanceToCluster_1',
             'DistanceToCluster_2',
             'DistanceToCluster_3', 'DistanceToCluster_4',
             'DistanceToCluster_5', 'DistanceToCluster_6',
             'DistanceToCluster_7', 'DistanceToCluster_8',
             'DistanceToCluster_9', 'DistanceToCluster_10',
             'DistanceToCluster_11', 'DistanceToCluster_12',
             'DistanceToCluster_13', 'DistanceToCluster_14',
             'TotalTfIdf'
             ]

    user_posts_features = user_posts_features[cols_]

    predicts = model.predict_proba(user_posts_features)[:, 1]
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


# Сам эндпоинт
@app.get("/post/recommendations", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:
    return get_recommended_feed(id, time, limit)
