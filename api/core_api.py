from fastapi import (
    APIRouter,
)
from my_utils.version import get_project_version

core_api = APIRouter()


@core_api.get("/health")
async def health_check():
    return {"status": "ok", "version": get_project_version()}
