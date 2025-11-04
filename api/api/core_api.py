from fastapi import (
    APIRouter,
)


core_api = APIRouter()


@core_api.get("/health")
async def health_check():
    return {"status": "ok"}
