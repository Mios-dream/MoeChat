from fastapi import APIRouter

from api.chat_api import chat_api
from api.asr_api import asr_api
from api.core_api import core_api
from api.tts_api import tts_api
from api.assistant_api import assistant_api

api_router = APIRouter(prefix="/api")
api_router.include_router(asr_api)
api_router.include_router(chat_api)
api_router.include_router(core_api)
api_router.include_router(tts_api)
api_router.include_router(assistant_api)
