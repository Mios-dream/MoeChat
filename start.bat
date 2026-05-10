@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

echo ════════════════════════════════════════
echo  MoeChat 启动器
echo ════════════════════════════════════════

echo 正在检查环境...
uv version

if not %errorlevel% neq 0 (
    echo uv已经安装,正在启动...
) else (
    echo 正在安装uv...
    @REM set UV_INSTALLER_GHE_BASE_URL="https://ghproxy.cn/https://github.com"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

    if %errorlevel% neq 0 (
        echo 安装失败
        pause
        exit /b 1
    )
)

echo 正在同步依赖...
uv sync

echo.
echo ────────────────────────────────────────
echo  是否检查更新？
echo  [1] 检查更新并启动
echo  [2] 直接启动（跳过更新检查）
echo  [U] 仅检查更新（不启动）
echo.
set /p UPDATE_CHOICE="请选择 [1/2/U]: "

if /i "!UPDATE_CHOICE!"=="1" (
    echo.
    echo 正在检查更新...
    uv run python update.py
    echo.
    echo 启动程序...
    uv run main_web.py
) else if /i "!UPDATE_CHOICE!"=="U" (
    echo.
    echo 正在检查更新...
    uv run python update.py
    echo.
    echo 按任意键退出...
    pause
    exit /b 0
) else (
    echo.
    echo 直接启动程序...
    uv run main_web.py
)

pause