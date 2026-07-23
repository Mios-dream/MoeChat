import os
import shutil
import time
import yaml
from Config import Config
from my_utils import config_manager as CConfig
from services.tts_service import ttsService
from models.dto.request.assistant_request import (
    AddAssistantRequest,
    UpdateAssistantRequest,
)
from models.types.assistant_info import AssistantInfo
from models.types.user_state import UserStateInfo
from core.assistant import Assistant
from my_utils.file_utils import get_latest_modification_time
import my_utils.log as Log


class AssistantService:
    """
    助手服务类
    """

    _instance = None

    current_assistant: Assistant | None = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 初始化当前助手为None
        self.current_assistant: Assistant | None = None
        # 初始化当前助手名称为None
        self.current_assistant_name: str | None = None
        # 初始化助手信息缓存为空字典
        self.assistants_cache: dict[str, AssistantInfo] = {}
        # 初始化已加载助手为空字典
        self.loaded_agents: dict[str, Assistant] = {}

    # ==================== User State Helpers ====================

    def _get_user_state_path(self, assistant_name: str) -> str:
        return os.path.join(Config.BASE_AGENTS_PATH, assistant_name, "user_state.yaml")

    def _load_user_state(self, assistant_name: str) -> UserStateInfo:
        path = self._get_user_state_path(assistant_name)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                return UserStateInfo.from_dict(data)
        return UserStateInfo()

    def _save_user_state(self, assistant_name: str, state: UserStateInfo) -> None:
        path = self._get_user_state_path(assistant_name)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                state.model_dump(),
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
                indent=2,
            )

    def _build_response_dict(self, assistant_name: str) -> dict:
        """
        构建带有 userState 嵌套的响应字典
        """
        if assistant_name not in self.assistants_cache:
            self.get_assistant_by_name(assistant_name)
        assistant_info = self.assistants_cache.get(assistant_name)
        if assistant_info is None:
            raise FileNotFoundError(f"助手 '{assistant_name}' 信息未找到")

        user_state = self._load_user_state(assistant_name)
        assistant_dict = assistant_info.model_dump()
        assistant_dict["userState"] = user_state.model_dump()
        return assistant_dict

    # ==================== Core Methods ====================

    def load_assistant_info(self) -> list[dict]:
        """
        加载全部助手信息（含用户私有状态）
        返回列表，每个元素为包含 userState 的字典
        """
        assistants_path = Config.BASE_AGENTS_PATH
        assistants: list[dict] = []

        if not os.path.exists(assistants_path):
            Log.logger.warning(f"助手路径不存在: {assistants_path}")
            raise FileNotFoundError(f"助手路径不存在: {assistants_path}")

        for dirname in os.listdir(assistants_path):
            file_path = os.path.join(assistants_path, dirname, "info.yaml")
            if os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_data = yaml.safe_load(f)

                assistant_info = AssistantInfo.from_dict(raw_data)
                self.assistants_cache[assistant_info.name] = assistant_info

                # 加载用户私有状态
                user_state = self._load_user_state(dirname)

                # 重新计算 assetsLastModified（始终从文件系统获取）
                assets_dir = os.path.join(assistants_path, dirname, "assets")
                if os.path.exists(assets_dir):
                    user_state.assetsLastModified = get_latest_modification_time(
                        assets_dir
                    )
                else:
                    user_state.assetsLastModified = 0

                # 如果 user_state 的 assetsLastModified 与文件系统不同，保存更新
                self._save_user_state(dirname, user_state)

                response = assistant_info.model_dump()
                response["userState"] = user_state.model_dump()
                assistants.append(response)

        return assistants

    def update_assistant_info(self, update_request: UpdateAssistantRequest) -> dict:
        """
        更新助手信息
        同时更新 userState 中的 updatedAt 和 assetsLastModified
        返回带 userState 嵌套的完整字典
        """
        assistants_path = Config.BASE_AGENTS_PATH
        assistant_dir = os.path.join(assistants_path, update_request.name)
        info_file_path = os.path.join(assistant_dir, "info.yaml")

        if not os.path.exists(assistant_dir) or not os.path.isfile(info_file_path):
            raise FileNotFoundError(f"助手 '{update_request.name}' 不存在")

        # 读取现有共享信息
        with open(info_file_path, "r", encoding="utf-8") as f:
            existing_info = yaml.safe_load(f)

        # 更新信息（只更新非None字段）
        update_data = update_request.model_dump(exclude_unset=True)
        # 过滤掉可能残留的 user-private 字段
        update_data.pop("firstMeetTime", None)
        update_data.pop("love", None)
        update_data.pop("updatedAt", None)
        update_data.pop("assetsLastModified", None)
        existing_info.update(update_data)

        # 保存共享信息
        with open(info_file_path, "w", encoding="utf-8") as f:
            yaml.dump(
                existing_info,
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
                indent=2,
            )

        # 更新缓存
        assistant_info = AssistantInfo.from_dict(existing_info)
        self.assistants_cache[assistant_info.name] = assistant_info

        # 更新用户状态
        user_state = self._load_user_state(update_request.name)
        user_state.updatedAt = int(time.time())

        assets_dir = os.path.join(assistant_dir, "assets")
        if os.path.exists(assets_dir):
            user_state.assetsLastModified = get_latest_modification_time(assets_dir)
        else:
            user_state.assetsLastModified = 0

        self._save_user_state(update_request.name, user_state)

        # 如果当前正在使用的助手被更新，重新加载
        if self.current_assistant_name == update_request.name:
            self.reload_current_assistant()

        return self._build_response_dict(update_request.name)

    def add_assistant(self, add_request: AddAssistantRequest) -> dict:
        """
        添加新助手
        分别创建 info.yaml（共享信息）和 user_state.yaml（用户私有状态）
        返回带 userState 嵌套的完整字典
        """
        assistants_path = Config.BASE_AGENTS_PATH
        assistant_dir = os.path.join(assistants_path, add_request.name)
        info_file_path = os.path.join(assistant_dir, "info.yaml")

        if not add_request.name:
            raise ValueError("助手名称不能为空")

        if os.path.exists(assistant_dir):
            raise ValueError(f"助手 '{add_request.name}' 已存在")

        os.makedirs(assistant_dir, exist_ok=True)
        os.makedirs(os.path.join(assistant_dir, "assets"), exist_ok=True)
        os.makedirs(os.path.join(assistant_dir, "memory"), exist_ok=True)
        os.makedirs(os.path.join(assistant_dir, "data_base"), exist_ok=True)

        assistant_info = add_request.model_dump()

        # 保存共享信息到 info.yaml
        with open(info_file_path, "w", encoding="utf-8") as f:
            yaml.dump(
                assistant_info,
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
                indent=2,
            )

        # 保存用户私有状态到 user_state.yaml
        user_state = UserStateInfo(
            firstMeetTime=int(time.time()),
            love=0,
            updatedAt=int(time.time()),
            assetsLastModified=0,
        )
        self._save_user_state(add_request.name, user_state)

        # 更新缓存
        assistant_info_obj = AssistantInfo.from_dict(assistant_info)
        self.assistants_cache[assistant_info_obj.name] = assistant_info_obj

        return self._build_response_dict(add_request.name)

    def delete_assistant(self, assistant_name: str) -> None:
        """
        删除助手目录及其所有内容（包括 user_state.yaml）
        默认助手不可删除（同时是语音配置的回退源）
        """
        # 后端保护：默认助手禁止删除（前端已隐藏入口，此处兜底防止绕过）
        if assistant_name == Config.DEFAULT_ASSISTANT_NAME:
            raise ValueError(f"默认助手 '{assistant_name}' 不可删除")

        assistants_path = Config.BASE_AGENTS_PATH
        assistant_dir = os.path.join(assistants_path, assistant_name)

        if not os.path.exists(assistant_dir):
            raise FileNotFoundError(f"助手 '{assistant_name}' 不存在")

        if self.current_assistant_name == assistant_name:
            self.current_assistant = None
            self.current_assistant_name = None

        if assistant_name in self.assistants_cache:
            del self.assistants_cache[assistant_name]
        if assistant_name in self.loaded_agents:
            del self.loaded_agents[assistant_name]

        shutil.rmtree(assistant_dir)

    async def set_assistant(self, assistant_name: str) -> Assistant:
        """
        设置当前助手
        """
        assistant_info_path = os.path.join(
            Config.BASE_AGENTS_PATH, assistant_name, "info.yaml"
        )
        if not os.path.exists(assistant_info_path):
            raise FileNotFoundError(f"助手 '{assistant_name}' 不存在")

        if assistant_name in self.loaded_agents:
            self.current_assistant = self.loaded_agents[assistant_name]
            self.current_assistant_name = assistant_name
            try:
                await self._set_gsv_models(self.current_assistant)
            except Exception:
                Log.logger.error(
                    f"设置助手语音模型失败: {assistant_name},跳过设置模型",
                    exc_info=True,
                )
            Log.logger.info(f"已切换到助手: {assistant_name}")
            return self.current_assistant

        try:
            agent = Assistant(assistant_name)
            try:
                await self._set_gsv_models(agent)
            except Exception:
                Log.logger.error(
                    f"设置助手语音模型失败: {assistant_name},跳过设置模型",
                    exc_info=True,
                )
            self.current_assistant = agent
            self.current_assistant_name = assistant_name
            self.loaded_agents[assistant_name] = agent
            Log.logger.info(f"已加载并切换到助手: {assistant_name}")
            return agent
        except Exception as e:
            Log.logger.error(
                f"加载助手失败: {assistant_name}, 错误: {e}", exc_info=True
            )
            raise RuntimeError(f"加载助手 '{assistant_name}' 失败: {str(e)}")

    def get_current_assistant(self) -> Assistant | None:
        return self.current_assistant

    def get_current_assistant_name(self) -> str | None:
        return self.current_assistant_name

    def reload_current_assistant(self) -> None:
        """
        重新加载当前助手
        """
        if self.current_assistant_name:
            try:
                agent = Assistant(self.current_assistant_name)
                self.current_assistant = agent
                self.loaded_agents[self.current_assistant_name] = agent
                Log.logger.info(f"已重新加载助手: {self.current_assistant_name}")
            except Exception as e:
                Log.logger.error(
                    f"重新加载助手失败: {self.current_assistant_name}, 错误: {e}"
                )
                raise FileNotFoundError(
                    f"重新加载助手 '{self.current_assistant_name}' 失败: {str(e)}"
                )

    async def initialize_default_assistant(self) -> Assistant | None:
        """
        初始化默认助手
        """
        last_used_file = os.path.join(Config.BASE_DATA_PATH, "last_used_agent.txt")
        last_used = None

        if os.path.exists(last_used_file):
            try:
                with open(last_used_file, "r", encoding="utf-8") as f:
                    last_used = f.read().strip()
            except Exception as e:
                Log.logger.error(f"读取上次使用的助手失败: {e}")

        if last_used:
            try:
                return await self.set_assistant(last_used)
            except:
                Log.logger.info(f"无法加载上次使用的助手: {last_used}")

        try:
            return await self.set_assistant(Config.DEFAULT_ASSISTANT_NAME)
        except:
            Log.logger.info(f"无法加载默认助手'{Config.DEFAULT_ASSISTANT_NAME}'")
        return None

    def save_last_used_agent(self) -> None:
        if self.current_assistant_name:
            try:
                last_used_file = os.path.join(
                    Config.BASE_DATA_PATH, "last_used_agent.txt"
                )
                with open(last_used_file, "w", encoding="utf-8") as f:
                    f.write(self.current_assistant_name)
                Log.logger.info(f"已保存最后使用的助手: {self.current_assistant_name}")
            except Exception as e:
                Log.logger.error(f"保存最后使用的助手失败: {e}")

    def get_assistant_by_name(self, assistant_name: str) -> dict | None:
        """
        根据名称获取助手信息（含用户私有状态）
        返回带 userState 嵌套的字典，或 None
        """
        # 先从缓存中查找共享信息
        if assistant_name not in self.assistants_cache:
            file_path = os.path.join(
                Config.BASE_AGENTS_PATH, assistant_name, "info.yaml"
            )
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        raw_data = yaml.safe_load(f)
                    assistant_info = AssistantInfo.from_dict(raw_data)
                    self.assistants_cache[assistant_name] = assistant_info
                except Exception as e:
                    Log.logger.error(f"加载助手信息失败: {assistant_name}, 错误: {e}")
                    return None
            else:
                return None

        try:
            return self._build_response_dict(assistant_name)
        except FileNotFoundError:
            return None

    async def _set_gsv_models(self, agent: Assistant) -> None:
        """
        设置当前助手的语音模型路径。

        gsvSetting 中的路径使用规则：
        - 使用助手自有资源：仅存文件名（如 "gpt.ckpt"），按 <assets>/<sub_dir>/<文件名> 解析
        - 使用预设资源：存相对于 resources/presets/ 的完整路径（如 "default/assets/models/gpt.ckpt"），
          按 resources/presets/<路径> 解析；同时兼容前一规则（优先匹配助手自有目录）。
        """
        is_api = CConfig.config["TTS"]["mode"] == "api"
        gsv = agent.agent_config.gsvSetting
        current_asset_base = f"{Config.BASE_AGENTS_PATH}/{agent.agent_name}/assets"

        def _resolve(sub_dir: str, file_path: str) -> str:
            """
            按序查找资源文件：
            1. <助手 assets>/<sub_dir>/<file_path>     （助手自有资源）
            2. <presets 根目录>/<file_path>              （预设资源）
            """
            if not file_path:
                return ""

            path = os.path.join(current_asset_base, sub_dir, file_path)
            if os.path.isfile(path):
                return path

            path = os.path.join(Config.RESOURCES_PATH, file_path)
            if os.path.isfile(path):
                return path

            return ""

        if is_api:
            gpt_model_path = gsv.gptModelPath or ""
            sovits_model_path = gsv.sovitsModelPath or ""
            ref_audio_path = gsv.refAudioPath or ""
            spk_audio_path = ref_audio_path
            prompt_text = gsv.promptText or ""
            language = gsv.textLang or "zh"
        else:
            gpt_model_path = _resolve("models", gsv.gptModelPath)
            sovits_model_path = _resolve("models", gsv.sovitsModelPath)
            ref_audio_path = _resolve("audio", gsv.refAudioPath)
            spk_audio_path = ref_audio_path
            prompt_text = gsv.promptText or ""
            language = gsv.textLang or "zh"

        await ttsService.switch_tts_models(
            gpt_model_path=gpt_model_path,
            sovits_model_path=sovits_model_path,
            spk_audio_path=spk_audio_path,
            ref_audio_path=ref_audio_path,
            prompt_text=prompt_text,
            language=language,
        )
