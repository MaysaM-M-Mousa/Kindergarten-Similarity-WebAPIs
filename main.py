from fastapi import FastAPI
from model.logic import find_top_n_similar_with_weights
from pydantic import BaseModel, Field
from datetime import datetime, date

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello Maysam!"}


class Record(BaseModel):
    latitude: float
    longitude: float
    country: str
    city: str
    tuition: float


class Weights(BaseModel):
    country: int = Field(None, ge=0, le=5)
    city: int = Field(None, ge=0, le=5)
    tuition: int = Field(None, ge=0, le=5)
    location: int = Field(None, ge=0, le=5)
    start_date: int = Field(None, ge=0, le=5)
    registration_expiration: int = Field(None, ge=0, le=5)


class Input(BaseModel):
    data: Record
    weights_dict: Weights
    date: date


@app.post("/kindergartens/similarity")
async def perform_logic(record: Input):
    return {"result": find_top_n_similar_with_weights(record.data, record.weights_dict, datetime.combine(record.date, datetime.min.time()))}
