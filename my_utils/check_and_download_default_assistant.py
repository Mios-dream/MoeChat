import json
import shutil
import tempfile
import zipfile
import os
from urllib.request import urlopen, Request
from urllib.error import URLError

from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)

from Config import Config
from services.assistant_service import AssistantService
from my_utils.log import logger

# GitHub 仓库信息（与 update.py 保持一致）
_GITHUB_OWNER = "Mios-dream"
_GITHUB_REPO = "MoeChat"
_DEFAULT_ASSISTANT_NAME = "澪"
_DOWNLOAD_TIMEOUT = 120

assistant_service = AssistantService()


async def check_and_download_default_assistant():
    """
    检查初始助手是否存在，如果不存在则从 GitHub Releases 下载。
    不阻塞启动：下载失败时仅记录警告，允许用户手动配置。
    """
    assistant_info_path = os.path.join(
        Config.BASE_AGENTS_PATH, _DEFAULT_ASSISTANT_NAME, "info.yaml"
    )
    if os.path.isfile(assistant_info_path):
        return

    logger.info(
        f"未检测到初始助手 '{_DEFAULT_ASSISTANT_NAME}'，尝试从 GitHub Releases 下载..."
    )

    # 从 "assistant" 标签下载固定的 assistant.zip
    _ASSISTANT_TAG = "assistant"
    _ASSISTANT_ZIP_NAME = "assistant.zip"
    api_url = f"https://api.github.com/repos/{_GITHUB_OWNER}/{_GITHUB_REPO}/releases/tags/{_ASSISTANT_TAG}"
    req = Request(api_url)
    req.add_header("Accept", "application/vnd.github+json")

    try:
        with urlopen(req, timeout=15) as resp:
            release_data = json.loads(resp.read().decode("utf-8"))
    except (URLError, json.JSONDecodeError) as e:
        logger.warning(
            f"获取 GitHub Release (tag={_ASSISTANT_TAG}) 信息失败: {e}，跳过助手数据下载。"
        )
        return

    # 查找固定的 assistant.zip 资产
    assets = release_data.get("assets", [])
    assistant_asset = None
    for asset in assets:
        if asset.get("name", "") == _ASSISTANT_ZIP_NAME:
            assistant_asset = asset
            break

    if assistant_asset is None:
        logger.warning(
            f"未在 Release (tag={_ASSISTANT_TAG}) 中找到 {_ASSISTANT_ZIP_NAME}，请手动配置助手。"
        )
        return

    download_url = assistant_asset.get("browser_download_url", "")
    if not download_url:
        logger.warning("无法获取助手数据下载链接。")
        return

    asset_name = _ASSISTANT_ZIP_NAME
    asset_size = assistant_asset.get("size", 0)
    logger.info(
        f"找到助手数据包: {asset_name} ({asset_size / 1024:.0f} KB)，开始下载..."
    )

    # 下载到临时文件
    tmp_zip = os.path.join(tempfile.gettempdir(), f"moechat_assistant_{asset_name}")
    try:
        req = Request(download_url)
        req.add_header("Accept", "application/octet-stream")
        with urlopen(req, timeout=_DOWNLOAD_TIMEOUT) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
            chunk_size = 8192

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                transient=False,
            ) as progress:
                task = progress.add_task(
                    f"[cyan]下载 {asset_name}", total=total_size or None
                )
                with open(tmp_zip, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))

        logger.info(f"助手数据包下载完成 ({os.path.getsize(tmp_zip) / 1024:.0f} KB)。")
    except Exception as e:
        logger.warning(f"下载助手数据包失败: {e}，请手动配置助手。")
        return

    # 解压到 data/agents/
    agents_dir = Config.BASE_AGENTS_PATH
    os.makedirs(agents_dir, exist_ok=True)
    try:
        logger.info("正在解压助手数据包...")

        with zipfile.ZipFile(tmp_zip, "r") as zf:
            members = [m for m in zf.namelist() if not m.endswith("/")]
            total_files = len(members)

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                transient=False,
            ) as progress:
                task = progress.add_task("[green]解压", total=total_files)
                for member in members:
                    dest = os.path.join(agents_dir, member)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                        dst.flush()
                        os.fsync(dst.fileno())
                    progress.update(task, advance=1)

        logger.info(f"助手数据包解压完成 → {agents_dir}")
    except Exception as e:
        logger.warning(f"解压助手数据包失败: {e}，请手动配置助手。")
        # 清理可能的不完整目录
        default_dir = os.path.join(agents_dir, _DEFAULT_ASSISTANT_NAME)
        if os.path.isdir(default_dir):
            shutil.rmtree(default_dir, ignore_errors=True)
    finally:
        # 清理临时 zip
        try:
            os.remove(tmp_zip)
        except OSError:
            pass
