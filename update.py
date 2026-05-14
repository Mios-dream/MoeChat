"""
自动更新脚本
从 GitHub Releases 检查最新版本并自动下载更新

用法:
    python update.py                  # 检查并更新（交互模式）
    python update.py --check-only     # 仅检查，不下载
    python update.py --json           # 输出 JSON 格式（供前端调用）
    python update.py --check-only --json  # 仅检查 + JSON 输出
"""

import os
import sys
import json
import shutil
import zipfile
import tempfile
from urllib.request import urlopen, Request
from urllib.error import URLError

# ── 配置 ──────────────────────────────────────────────────
# GitHub 仓库信息（改为你的仓库）
GITHUB_OWNER = "Mios-dream"
GITHUB_REPO = "MoeChat"

# 排除不覆盖的文件/文件夹（用户数据、配置、模型等，更新时不覆盖）
EXCLUDE_PATTERNS = [
    "config.yaml",
    "config.yml",
    "data/",
    ".venv/",
    "venv/",
    "__pycache__/",
    "plugins/*/config.yml",
]

# 下载超时（秒）
DOWNLOAD_TIMEOUT = 120
# ──────────────────────────────────────────────────────────


def get_current_version():
    """
    从 pyproject.toml 获取当前版本
    Returns:
        str: 当前版本号
    """
    try:
        import tomllib

        pyproject_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "pyproject.toml"
        )
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)
        return config.get("project", {}).get("version", "0.0.0")
    except Exception:
        return "0.0.0"


def get_latest_release():
    """
    查询 GitHub 最新 Release 信息
    Returns:
        dict: Release 信息，包含 tag_name, html_url, assets 等
    """
    api_url = (
        f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
    )
    req = Request(api_url)
    req.add_header("Accept", "application/vnd.github+json")
    # 如果使用私有仓库，可以设置 Token
    # token = os.environ.get("GITHUB_TOKEN")
    # if token:
    #     req.add_header("Authorization", f"Bearer {token}")

    try:
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except URLError as e:
        print(f"❌ 检查更新失败（网络错误）: {e}")
        return None
    except json.JSONDecodeError:
        print("❌ 解析 GitHub 响应失败")
        return None


def compare_versions(v1: str, v2: str) -> int:
    """
    比较两个版本号
    Returns:
        int: 1 表示 v1 > v2, -1 表示 v1 < v2, 0 表示相等
    """

    def parse(v: str):
        # 去掉前缀 v
        v = v.lstrip("v")
        parts = []
        for p in v.split("."):
            num = ""
            for c in p:
                if c.isdigit():
                    num += c
                else:
                    break
            parts.append(int(num) if num else 0)
        # 补齐到 3 段
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts[:3])

    p1, p2 = parse(v1), parse(v2)
    if p1 > p2:
        return 1
    elif p1 < p2:
        return -1
    return 0


def find_zip_asset(release_data: dict):
    """
    从 Release assets 中找到 zip 文件
    """
    assets = release_data.get("assets", [])
    for asset in assets:
        name = asset.get("name", "")
        if name.endswith(".zip"):
            return asset
    return None


def should_exclude(relative_path: str) -> bool:
    """
    判断文件是否应该被排除（用户数据/配置等）
    """
    for pattern in EXCLUDE_PATTERNS:
        # 目录模式：data/ 匹配 data/ 开头的任何路径
        if pattern.endswith("/"):
            if relative_path.startswith(pattern) or relative_path == pattern.rstrip(
                "/"
            ):
                return True
        # 文件模式
        elif relative_path == pattern:
            return True
        # glob 模式
        elif "*" in pattern:
            import fnmatch

            if fnmatch.fnmatch(relative_path, pattern):
                return True
    return False


def download_with_progress(url: str, dest_path: str):
    """
    下载文件并显示进度
    """
    req = Request(url)
    req.add_header("Accept", "application/octet-stream")

    with urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 8192

        with open(dest_path, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = downloaded * 100 / total_size
                    bar_len = 30
                    filled = int(bar_len * downloaded / total_size)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(
                        f"\r  下载中: |{bar}| {percent:.1f}% ({downloaded // 1024}/{total_size // 1024} KB)",
                        end="",
                        flush=True,
                    )
        print()


def apply_update(zip_path: str, project_root: str):
    """
    解压更新包并覆盖文件
    """
    temp_extract = tempfile.mkdtemp(prefix="moechat_update_")

    try:
        print("📦 正在解压更新包...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_extract)

        # zip 内可能有一个顶层目录（MoeChat/），检测并处理
        extracted_items = os.listdir(temp_extract)
        source_dir = temp_extract
        if len(extracted_items) == 1 and os.path.isdir(
            os.path.join(temp_extract, extracted_items[0])
        ):
            source_dir = os.path.join(temp_extract, extracted_items[0])

        # 遍历源目录，逐个覆盖
        updated_files = 0
        skipped_files = 0
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                src_file = os.path.join(root, file)
                rel_path = os.path.relpath(src_file, source_dir)

                if should_exclude(rel_path):
                    skipped_files += 1
                    continue

                dst_file = os.path.join(project_root, rel_path)
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy2(src_file, dst_file)
                updated_files += 1

        print(f"✅ 已更新 {updated_files} 个文件，跳过 {skipped_files} 个用户/配置文件")
        return True

    except Exception as e:
        print(f"❌ 解压更新失败: {e}")
        return False
    finally:
        # 清理临时文件
        shutil.rmtree(temp_extract, ignore_errors=True)


def check_and_update(json_output: bool = False, check_only: bool = False):
    """
    主流程：检查更新 → 下载 → 应用

    Args:
        json_output: 是否输出 JSON 格式
        check_only: 仅检查，不下载更新

    Returns:
        bool: 是否执行了更新（check_only 时返回是否有更新）
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    current_ver = get_current_version()

    if not json_output:
        print(f"📋 MoeChat 自动更新工具")
        print(f"{'=' * 40}")
        print(f"当前版本: v{current_ver}")
        print(f"正在检查更新...")

    release = get_latest_release()
    if release is None:
        if json_output:
            print(
                json.dumps(
                    {"error": "无法获取最新版本信息", "current_version": current_ver},
                    ensure_ascii=False,
                )
            )
        else:
            print("⚠️  无法获取最新版本信息，请检查网络连接")
        return False

    latest_tag = release.get("tag_name", "v0.0.0")
    latest_ver = latest_tag.lstrip("v")
    release_url = release.get("html_url", "")
    release_notes = release.get("body", "")

    if not json_output:
        print(f"最新版本: {latest_tag}")
        print(f"Release:   {release_url}")
        print()

    if compare_versions(current_ver, latest_ver) >= 0:
        if json_output:
            print(
                json.dumps(
                    {
                        "update_available": False,
                        "current_version": current_ver,
                        "latest_version": latest_ver,
                    },
                    ensure_ascii=False,
                )
            )
        else:
            print("✅ 已是最新版本，无需更新")
        return False

    # 发现新版本
    if check_only:
        if json_output:
            print(
                json.dumps(
                    {
                        "update_available": True,
                        "current_version": current_ver,
                        "latest_version": latest_ver,
                        "release_url": release_url,
                        "release_notes": release_notes,
                    },
                    ensure_ascii=False,
                )
            )
        else:
            print(f"✨ 发现新版本 {latest_tag}")
        return True

    if not json_output:
        print(f"✨ 发现新版本 {latest_tag}，准备下载...")
        print()

    asset = find_zip_asset(release)
    if not asset:
        if json_output:
            print(
                json.dumps(
                    {
                        "error": "未找到 zip 包",
                        "current_version": current_ver,
                        "latest_version": latest_ver,
                    },
                    ensure_ascii=False,
                )
            )
        else:
            print("❌ 未在 Release 中找到 zip 包")
            print("提示: 推送标签后 GitHub Actions 会自动打包，请确认工作流已成功运行")
        return False

    download_url = asset.get("browser_download_url", "")
    if not download_url:
        if json_output:
            print(json.dumps({"error": "无法获取下载链接"}, ensure_ascii=False))
        else:
            print("❌ 无法获取下载链接")
        return False

    # 下载到临时文件
    tmp_zip = os.path.join(tempfile.gettempdir(), f"moechat_{latest_ver}.zip")
    try:
        if not json_output:
            print(f"⬇️  正在下载更新包...")
        download_with_progress(download_url, tmp_zip)
    except Exception as e:
        if json_output:
            print(json.dumps({"error": f"下载失败: {e}"}, ensure_ascii=False))
        else:
            print(f"❌ 下载失败: {e}")
        return False

    # 应用更新
    if not json_output:
        print()
    success = apply_update(tmp_zip, project_root)

    # 清理下载的 zip
    try:
        os.remove(tmp_zip)
    except OSError:
        pass

    if success:
        if json_output:
            print(
                json.dumps(
                    {
                        "success": True,
                        "message": f"已升级到 {latest_tag}",
                        "current_version": latest_ver,
                    },
                    ensure_ascii=False,
                )
            )
        else:
            print()
            print(f"🎉 更新完成！已升级到 {latest_tag}")
            print("💡 建议重启程序以加载更新")
        return True

    if json_output:
        print(json.dumps({"error": "解压更新失败"}, ensure_ascii=False))
    return False


def main():
    """入口函数"""
    # 解析命令行参数
    json_output = "--json" in sys.argv
    check_only = "--check-only" in sys.argv

    try:
        updated = check_and_update(json_output=json_output, check_only=check_only)
        if not updated:
            # 没有更新或更新失败
            pass
    except KeyboardInterrupt:
        if json_output:
            print(json.dumps({"error": "用户取消"}, ensure_ascii=False))
        else:
            print("\n⚠️  用户取消更新")
        sys.exit(1)
    except Exception as e:
        if json_output:
            print(json.dumps({"error": str(e)}, ensure_ascii=False))
        else:
            print(f"\n❌ 更新过程出现异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
