import os
import re
import shutil
import zipfile
import io
import tempfile
import yaml
from pathlib import Path
from models.dto.assistant_request import (
    AddAssistantRequest,
    AssistantAssetsCheckRequest,
    AssistantAssetsDownloadRequest,
    DeleteAssistantRequest,
    UpdateAssistantRequest,
    SwitchAssistantRequest,
)

from services.assistant_service import AssistantService
from my_utils.file_utils import get_latest_modification_time, get_subdirectory_mtimes
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
)
from Config import Config
from my_utils.log import logger

assistant_api = APIRouter()

assistant_service = AssistantService()


@assistant_api.get("/assistants")
async def get_assistants():
    """
    获取所有助手信息

    Returns:
        包含所有助手信息的列表
    """
    assistants = assistant_service.load_assistant_info()

    if assistants:
        return {
            "msg": "Load assistants success",
            "data": assistants,
            "count": len(assistants),
        }
    else:
        return {"msg": "No assistants found", "data": assistants, "count": 0}


@assistant_api.get("/assistant/current")
async def get_current_assistant():
    """
    获取当前选择的助手信息

    Returns:
        当前助手的信息或空对象（含 userState 嵌套）
    """
    try:
        # 获取当前助手名称
        current_assistant_name = assistant_service.get_current_assistant_name()

        if not current_assistant_name:
            return {"msg": "No current assistant selected", "data": None}

        # 获取当前助手的详细信息（含 userState）
        current_assistant = assistant_service.get_assistant_by_name(
            current_assistant_name
        )

        if current_assistant:
            return {
                "msg": "Get current assistant success",
                "data": current_assistant,
            }
        else:
            return {"msg": "Current assistant info not found", "data": None}
    except Exception as e:
        logger.error(f"获取当前助手信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取当前助手信息失败: {str(e)}")


@assistant_api.post("/assistant/switch")
async def switch_assistant(switch_request: SwitchAssistantRequest):
    """
    切换当前使用的助手

    Args:
        switch_request: 包含要切换的助手名称

    Returns:
        切换结果信息
    """
    try:
        if not switch_request.name:
            raise HTTPException(status_code=400, detail="Assistant name is required")
        if switch_request.name == assistant_service.get_current_assistant_name():
            # 获取助手详细信息返回给客户端
            assistant_info = assistant_service.get_assistant_by_name(
                switch_request.name
            )
            return {
                "msg": f"成功切换到助手 '{switch_request.name}'",
                "data": assistant_info,
            }
        # 调用服务层切换助手
        await assistant_service.set_assistant(switch_request.name)

        # 保存最后使用的助手，以便下次启动时自动加载
        assistant_service.save_last_used_agent()

        # 获取助手详细信息返回给客户端
        assistant_info = assistant_service.get_assistant_by_name(switch_request.name)

        if not assistant_info:
            raise HTTPException(
                status_code=404,
                detail=f"Assistant '{switch_request.name}' not found",
            )

        return {
            "msg": f"成功切换到助手 '{switch_request.name}'",
            "data": assistant_info,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"切换助手失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"切换助手失败: {str(e)}")


@assistant_api.post("/assistant/assets/check")
async def check_assets_update(
    assistant_assets_check: AssistantAssetsCheckRequest,
):
    """
    检查助手资源文件是否有更新

    Args:
        assistant_assets_check: 助手名称
        last_modified: 客户端保存的最后修改时间戳

    Returns:
        包含是否需要更新的信息，以及各子目录的修改时间
    """
    # 构建助手目录路径
    assistant_dir = os.path.join(Config.BASE_AGENTS_PATH, assistant_assets_check.name)
    assets_dir = os.path.join(assistant_dir, "assets")

    # 检查助手是否存在
    if not os.path.exists(assistant_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Assistant '{assistant_assets_check.name}' not found",
        )

    # 检查assets目录是否存在
    if not os.path.exists(assets_dir):
        return {
            "msg": "Assets directory not found",
            "needsUpdate": False,
            "assetsLastModified": 0,
            "assetTypes": {},
        }

    # 获取最新的修改时间
    latest_mtime = get_latest_modification_time(assets_dir)

    # 获取各子目录的修改时间
    subdirectory_mtimes = get_subdirectory_mtimes(assets_dir)

    # 比较修改时间，判断是否需要更新
    needs_update = latest_mtime > assistant_assets_check.lastModified

    return {
        "msg": "Check update success",
        "needsUpdate": needs_update,
        "assetsLastModified": latest_mtime,
        "assetTypes": subdirectory_mtimes,
    }


@assistant_api.post("/assistant/assets/download")
async def download_assets(assistant_assets_download: AssistantAssetsDownloadRequest):
    """
    下载助手资源文件（assets目录），打包为zip文件

    支持增量下载：通过 assetTypes 参数指定需要下载的资源类型（子目录名），
    只打包指定子目录下的文件。为空则下载全部资源。

    Args:
        assistant_name: 助手名称
        assetTypes: 需要下载的资源类型列表（子目录名），为空则下载全部

    Returns:
        zip格式的资源文件
    """
    # 构建助手目录路径
    assistant_dir = os.path.join(
        Config.BASE_AGENTS_PATH, assistant_assets_download.name
    )
    assets_dir = os.path.join(assistant_dir, "assets")

    # 检查助手是否存在
    if not os.path.exists(assistant_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Assistant '{ assistant_assets_download.name}' not found",
        )

    # 检查assets目录是否存在
    if not os.path.exists(assets_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Assets directory not found for assistant '{ assistant_assets_download.name}'",
        )

    # 获取需要下载的资源类型
    asset_types = assistant_assets_download.assetTypes
    # 是否为增量模式（指定了资源类型且不为空）
    is_incremental = len(asset_types) > 0

    # 创建内存中的zip文件
    zip_buffer = io.BytesIO()

    # 遍历assets目录，添加文件到zip
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        assets_path = Path(assets_dir)
        # 检查目录是否为空
        if not any(assets_path.iterdir()):
            # 如果目录为空
            raise HTTPException(
                status_code=404,
                detail=f"Assets is empty for assistant '{ assistant_assets_download.name}'",
            )

        if is_incremental:
            # 增量模式：只打包指定子目录的文件
            for asset_type in asset_types:
                # 安全检查：防止路径遍历
                if "/" in asset_type or "\\" in asset_type or ".." in asset_type:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid asset type: {asset_type}",
                    )

                type_dir = os.path.join(assets_dir, asset_type)
                if os.path.isdir(type_dir):
                    # 子目录存在，打包该目录下的所有文件
                    for root, _, files in os.walk(type_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # 计算相对于assets父目录的路径，保持目录结构
                            rel_path = os.path.relpath(
                                file_path, os.path.dirname(assets_dir)
                            )
                            zip_file.write(file_path, rel_path)
                elif asset_type == "other":
                    # "other" 表示根目录下直接存放的文件
                    for entry in os.listdir(assets_dir):
                        entry_path = os.path.join(assets_dir, entry)
                        if os.path.isfile(entry_path):
                            rel_path = os.path.relpath(
                                entry_path, os.path.dirname(assets_dir)
                            )
                            zip_file.write(entry_path, rel_path)
        else:
            # 全量模式：添加所有文件
            for root, _, files in os.walk(assets_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # 计算相对路径，保持目录结构
                    rel_path = os.path.relpath(file_path, os.path.dirname(assets_dir))
                    zip_file.write(file_path, rel_path)

    # 重置文件指针
    zip_buffer.seek(0)

    # 返回zip文件
    return Response(
        content=zip_buffer.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=assets.zip"},
    )


@assistant_api.post("/assistant/assets/upload")
async def upload_assets(
    name: str = Form(...),
    asset_types: str = Form(default=""),
    assets_zip: UploadFile = File(...),
):
    """
    上传助手资源文件（assets目录）

    支持增量上传：通过 asset_types 参数指定需要上传的资源类型（子目录名，逗号分隔），
    只更新指定子目录下的文件，保留其他目录不变。为空则全量覆盖。

    Args:
        name: 助手名称
        asset_types: 需要上传的资源类型（子目录名，逗号分隔），为空则全量覆盖
        assets_zip: 包含assets目录的zip文件

    Returns:
        上传成功的消息
    """
    # 验证助手名称合法性
    if not name or not re.match(r"^[a-zA-Z0-9_\u4e00-\u9fa5]+$", name):
        raise HTTPException(status_code=400, detail="Invalid assistant name format")

    # 防止路径遍历攻击
    safe_name = os.path.basename(name)
    if safe_name != name:
        raise HTTPException(status_code=400, detail="Invalid assistant name")

    # 构建助手目录路径
    assistant_dir = os.path.join(Config.BASE_AGENTS_PATH, safe_name)

    # 检查助手是否存在
    if not os.path.exists(assistant_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Assistant '{safe_name}' not found",
        )

    # 构建assets目录路径
    assets_dir = os.path.join(assistant_dir, "assets")

    # 解析 asset_types 参数：逗号分隔的字符串转为列表
    asset_types_list = [t.strip() for t in asset_types.split(",") if t.strip()] if asset_types else []
    # 是否为增量模式（指定了资源类型且不为空）
    is_incremental = len(asset_types_list) > 0

    # 读取上传的文件内容
    try:
        # 读取zip文件内容
        zip_content = await assets_zip.read()

        # 验证文件类型是否为zip
        if not zip_content.startswith(b"PK"):
            raise HTTPException(
                status_code=400, detail="Uploaded file is not a valid zip file"
            )

        if is_incremental:
            # 增量模式：只更新指定的子目录，保留其他目录
            # 验证 asset_types 中的路径安全性
            for asset_type in asset_types_list:
                if "/" in asset_type or "\\" in asset_type or ".." in asset_type:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid asset type: {asset_type}",
                    )

            # 确保assets目录存在
            os.makedirs(assets_dir, exist_ok=True)

            # 直接从zip文件中提取指定子目录的内容
            with zipfile.ZipFile(io.BytesIO(zip_content), "r") as zip_ref:
                zip_contents = zip_ref.namelist()

                # 检查是否包含assets目录前缀
                has_assets_dir = any(item.startswith("assets/") for item in zip_contents)

                # 提取文件到目标目录
                for item in zip_contents:
                    # 跳过目录项
                    if item.endswith("/"):
                        continue

                    # 获取相对于assets目录的路径
                    if has_assets_dir:
                        if not item.startswith("assets/"):
                            continue
                        relative_path = item[7:]  # 去掉 assets/ 前缀
                    else:
                        relative_path = item

                    # 检查该文件是否属于指定的资源类型
                    path_parts = relative_path.split("/", 1)
                    if len(path_parts) < 2:
                        # 文件直接在assets根目录下，检查是否有 "other" 类型
                        if "other" not in asset_types_list:
                            continue
                    else:
                        # 文件在子目录中，检查子目录名是否在指定类型中
                        if path_parts[0] not in asset_types_list:
                            continue

                    # 构建目标路径
                    target_path = os.path.join(assets_dir, relative_path)
                    # 确保目标文件的目录存在
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    # 提取文件
                    with (
                        zip_ref.open(item) as source,
                        open(target_path, "wb") as target,
                    ):
                        target.write(source.read())
        else:
            # 全量模式：删除原assets目录后重新创建
            if os.path.exists(assets_dir):
                shutil.rmtree(assets_dir)

            # 创建新的assets目录
            os.makedirs(assets_dir, exist_ok=True)

            # 直接从zip文件中提取assets目录内容
            with zipfile.ZipFile(io.BytesIO(zip_content), "r") as zip_ref:
                # 获取zip文件中的所有文件列表
                zip_contents = zip_ref.namelist()

                # 检查是否包含assets目录
                has_assets_dir = any(item.startswith("assets/") for item in zip_contents)

                # 提取文件到目标目录
                for item in zip_contents:
                    # 跳过目录项
                    if item.endswith("/"):
                        continue

                    if has_assets_dir:
                        # 如果zip中包含assets目录，需要去掉这个前缀
                        if item.startswith("assets/"):
                            # 去掉assets/前缀，保留剩余路径
                            target_path = os.path.join(assets_dir, item[7:])
                            # 确保目标文件的目录存在
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)
                            # 提取文件
                            with (
                                zip_ref.open(item) as source,
                                open(target_path, "wb") as target,
                            ):
                                target.write(source.read())
                    else:
                        # 如果zip中不包含assets目录，直接提取到assets目录
                        target_path = os.path.join(assets_dir, item)
                        # 确保目标文件的目录存在
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        # 提取文件
                        with (
                            zip_ref.open(item) as source,
                            open(target_path, "wb") as target,
                        ):
                            target.write(source.read())

        logger.info(f"Successfully uploaded assets for assistant: {safe_name}")

        return {
            "status": "success",
            "message": f"Assets uploaded successfully for assistant '{safe_name}'",
        }

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file format")
    except Exception as e:
        logger.error(f"Error uploading assets for assistant {safe_name}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to upload assets: {str(e)}"
        )


@assistant_api.post("/assistant/import-from-zip")
async def import_assistant_from_zip(
    assistant_zip: UploadFile = File(...),
):
    """
    上传完整助手 zip 包并导入为新助手。

    zip 包需要包含 info.yaml 或 info.yml，可位于根目录或单层助手目录中。
    """
    try:
        # 读取上传的zip文件内容
        zip_content = await assistant_zip.read()

        # 验证文件类型是否为zip
        if not zip_content.startswith(b"PK"):
            raise HTTPException(status_code=400, detail="上传文件不是有效的zip文件")

        # 使用临时目录解析zip内容
        with tempfile.TemporaryDirectory() as temp_dir:
            # 解压zip文件到临时目录
            with zipfile.ZipFile(io.BytesIO(zip_content), "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            # 查找 info.yaml 或 info.yml 文件
            info_file_path = None
            assistant_root_dir = None  # 记录助手的根目录（用于复制assets等）

            # 首先在根目录查找
            root_info_yaml = os.path.join(temp_dir, "info.yaml")
            root_info_yml = os.path.join(temp_dir, "info.yml")

            if os.path.isfile(root_info_yaml):
                info_file_path = root_info_yaml
                assistant_root_dir = temp_dir
            elif os.path.isfile(root_info_yml):
                info_file_path = root_info_yml
                assistant_root_dir = temp_dir
            else:
                # 在单层子目录中查找
                for dirname in os.listdir(temp_dir):
                    dir_path = os.path.join(temp_dir, dirname)
                    if not os.path.isdir(dir_path):
                        continue

                    sub_info_yaml = os.path.join(dir_path, "info.yaml")

                    if os.path.isfile(sub_info_yaml):
                        info_file_path = sub_info_yaml
                        assistant_root_dir = dir_path
                        break

            # 验证是否找到了info文件
            if not info_file_path:
                raise HTTPException(
                    status_code=400,
                    detail="zip文件中未找到 info.yaml",
                )

            # 读取并解析 info.yaml
            with open(info_file_path, "r", encoding="utf-8") as f:
                assistant_data = yaml.safe_load(f)

            # 获取助手名称
            assistant_name = assistant_data.get("name")
            target_assistant_dir = os.path.join(Config.BASE_AGENTS_PATH, assistant_name)

            # 检查助手是否已存在
            if os.path.exists(target_assistant_dir):
                raise HTTPException(
                    status_code=400,
                    detail=f"助手 '{assistant_name}' 已存在",
                )

            # 复制整个助手目录到目标位置
            if assistant_root_dir:
                # 复制整个助手目录
                shutil.copytree(assistant_root_dir, target_assistant_dir)

            logger.info(
                f"Successfully imported assistant '{assistant_name}' from zip file"
            )

            # 获取导入后的助手信息返回给客户端
            imported_assistant = assistant_service.get_assistant_by_name(assistant_name)

            return {
                "msg": f"成功导入助手 '{assistant_name}'",
                "data": imported_assistant,
            }

    except HTTPException:
        # 直接抛出已有的HTTPException
        raise
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="无效的zip文件格式")
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"info文件格式错误: {str(e)}")
    except Exception as e:
        logger.error(f"导入助手失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"导入助手失败: {str(e)}")


@assistant_api.post("/assistant/info/update")
async def update_assistant(update_request: UpdateAssistantRequest):
    """
    更新助手信息接口
    """
    try:
        # 调用服务层更新助手信息，返回含 userState 嵌套的字典
        updated_assistant = assistant_service.update_assistant_info(update_request)

        return {
            "msg": f"助手 '{update_request.name}' 信息更新成功",
            "data": updated_assistant,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"更新助手信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新助手信息失败: {str(e)}")


@assistant_api.post("/assistant/info/add")
async def add_assistant(add_request: AddAssistantRequest):
    """
    添加新助手接口
    """
    try:
        # 调用服务层添加助手，返回含 userState 嵌套的字典
        new_assistant = assistant_service.add_assistant(add_request)

        return {
            "msg": f"助手 '{add_request.name}' 添加成功",
            "data": new_assistant,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"添加助手失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"添加助手失败: {str(e)}")


@assistant_api.post("/assistant/info/delete")
async def delete_assistant(delete_request: DeleteAssistantRequest):
    """
    删除助手接口
    """
    try:
        # 调用服务层删除助手
        assistant_service.delete_assistant(delete_request.name)
        return {"msg": f"助手 '{delete_request.name}' 删除成功"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"删除助手失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除助手失败: {str(e)}")
