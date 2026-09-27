from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
import os

from backend.routers import cv
from backend.services.cv_inference import cv_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the CV model if checkpoint path is set
    cv_service.load_model()
    yield
    # Shutdown logic can go here

app = FastAPI(
    title="Disaster Relief Coordination API",
    description="Backend routing for NLP, Vision, and ML modules",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(cv.router)

@app.get("/")
def health_check():
    return {"status": "Active", "message": "Disaster Relief Platform Backend is running."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)