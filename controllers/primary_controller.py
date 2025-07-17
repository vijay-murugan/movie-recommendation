from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

from utils.model_utils import get_custom_recommendations
from movie_recommendation import main as train_model


# uvicorn controllers.primary_controller:app --reload

class RecommendationRequest(BaseModel):
    user_id: int
    movie_id: int
    top_n: int = 10

@asynccontextmanager
async def lifespan(_app: FastAPI):
    train_model()  # Run your startup logic here
    yield  # App runs here

app = FastAPI(lifespan=lifespan)

class Rating(BaseModel):
    movieId: int
    rating: int

class NewUser(BaseModel):
    user_id: str
    ratings: list[Rating]

@app.post("/recommend")
def recommend(req: NewUser):
    # Call your hybrid recommendation function
    recs = get_custom_recommendations(req.user_id, req.ratings)
    return recs.to_dict(orient='records')

