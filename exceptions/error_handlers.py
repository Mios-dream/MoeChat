from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback

def setup_exception_handlers(app):
    # 全局异常处理
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content={
                "message": "服务器内部错误",
                "detail": str(exc),
                "trace": traceback.format_exc()
            },
        )

    # 捕获 HTTP 异常
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "message": exc.detail
            },
        )

    # 捕获请求验证错误
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        return JSONResponse(
            status_code=422,
            content={
                "message": "请求参数验证失败",
                "errors": exc.errors()
            },
        )