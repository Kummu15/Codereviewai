from fastapi import FastAPI
from app.api.routes import router
from app.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="CodeReview AI: automated code quality review service.",
    version=settings.PROJECT_VERSION,
)

app.include_router(router, prefix=settings.API_PREFIX)

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": settings.PROJECT_NAME}
