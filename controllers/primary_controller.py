from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

from movie_recommendation import get_hybrid_recommendations
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

@app.post("/recommend")
def recommend(req: RecommendationRequest):
    # Call your hybrid recommendation function
    recs = get_hybrid_recommendations(req.user_id, req.movie_id, req.top_n)
    return {"recommendations": recs}
