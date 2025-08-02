import logging

from pydantic import BaseModel
from contextlib import asynccontextmanager


from utils.model_utils import get_custom_recommendations
from movie_recommendation import main as train_model
#
import configparser
import os
config = configparser.ConfigParser()
config.read('conf/dbconn.properties')

username = config.get('DEFAULT', 'db.username')
password = config.get('DEFAULT', 'db.password')
# uvicorn controllers.primary_controller:app --reload
connection = "mongodb+srv://" + username + ":" + password + "@cluster0.whlb6ym.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ['http://localhost:3000',
           'http://localhost:5173']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    train_model()
    yield

class RecommendationRequest(BaseModel):
    user_id: int
    movie_id: int
    top_n: int = 10


class Rating(BaseModel):
    movieId: int
    rating: int

class NewUser(BaseModel):
    user_id: str
    ratings: list[Rating]

@app.post("/recommend")
def recommend(req: NewUser):
    logging.log(logging.INFO, msg=f"Received recommendation request for user_id={req.user_id} with ratings={req.ratings}")
    recs = get_custom_recommendations(req.user_id, req.ratings)
    return recs.to_dict(orient='records')
