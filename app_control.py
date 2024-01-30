import os
from datetime import datetime
from typing import List
from fastapi import FastAPI
from schema import PostGet
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from dotenv import load_dotenv

# Подгрузим переменные окружения
load_dotenv()
url_object = URL.create(
    drivername="postgresql",
    username=os.environ.get("POSTGRES_USER"),
    password=os.environ.get("POSTGRES_PASSWORD"),
    host=os.environ.get("POSTGRES_HOST"),
    port=os.environ.get("POSTGRES_PORT"),
    database=os.environ.get("POSTGRES_DATABASE")
    )


# Функция для загрузки модели внутри LMS
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
            "СИСТЕМЫ/Final_project/models/control_catboost_model"
            )
        )
    return model


# Функция для загрузки информации из БД батчами
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(url_object)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


# Загрузка признаков пользователей
def load_features() -> pd.DataFrame:
    df = batch_load_sql('SELECT * FROM public.user_data')
    return df


df = load_features()  # Загрузим признаки пользователей

model = load_models()  # Загрузим модель

app = FastAPI()  # Создаём приложение

# Загружаем оставшиеся необходимые таблицы
post_text_df = batch_load_sql(
    'SELECT * FROM a_maslennikov_post_features_lesson_22'
    )

raw_post_text = batch_load_sql('SELECT * FROM public.post_text_df')

liked_posts = batch_load_sql(
    'SELECT * FROM a_maslennikov_liked_posts_lesson_22'
    )

numeric_columns = ['gender', 'age', 'exp_group', 'count_words', 'month',
                   'TextCluster', 'day_of_week', 'hour']


# Эндпоинт. Обработка запроса
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
        id: int,
        time: datetime,
        limit: int = 5
        ) -> List[PostGet]:
    df_to_pred = df[df["user_id"] == id].drop("user_id", axis=1)
    df_to_pred["month"] = time.month
    df_to_pred["day_of_week"] = time.weekday()
    df_to_pred['hour'] = time.hour
    posts_to_rec = post_text_df[~post_text_df.index.isin(
        (liked_posts[liked_posts.user_id == id]
         .drop("user_id", axis=1)
         .sort_values(by="post_id").set_index("post_id")).index
        )]
    df_to_pred = pd.concat(
        [df_to_pred, posts_to_rec.drop("post_id", axis=1)], axis=1
        ) \
        .fillna(method='ffill').dropna()
    df_to_pred[numeric_columns] = df_to_pred[numeric_columns].astype(np.int16)

    resp_df = pd.DataFrame(
        np.isin(
            model.predict_proba(df_to_pred)[:, 1],
            sorted(
                model.predict_proba(df_to_pred)[:, 1],
                reverse=True
                )[:limit]
            ),
        columns=["response"]
        )

    resp_df = resp_df.loc[resp_df.response, :][0:limit]

    return raw_post_text.loc[resp_df.index.to_list()].rename(
        columns={"post_id": "id"}
        ).to_dict("records")
