from fastapi import (
    APIRouter,
)
from utils.version import get_project_version

core_api = APIRouter()


@core_api.get("/health")
async def health_check():
    return {"status": "ok", "version": get_project_version()}
