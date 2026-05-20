import os
import shutil
import time
import yaml
from Config import Config
from my_utils import config_manager as CConfig
from services.tts_service import ttsService
from models.dto.assistant_request import AddAssistantRequest, UpdateAssistantRequest
from models.types.assistant_info import AssistantInfo
from models.types.user_state import UserStateInfo
from services.agent import Agent
from my_utils.file_utils import get_latest_modification_time
import my_utils.log as Log


class AssistantService:
    """
    助手服务类
    """

    _instance = None

    current_assistant: Agent | None = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 初始化当前助手为None
        self.current_assistant: Agent | None = None
        # 初始化当前助手名称为None
        self.current_assistant_name: str | None = None
        # 初始化助手信息缓存为空字典
        self.assistants_cache: dict[str, AssistantInfo] = {}
        # 初始化已加载助手为空字典
        self.loaded_agents: dict[str, Agent] = {}
        # 默认助手 gsvSetting 缓存（用于语音配置回退），首次访问时加载
        self._default_gsv_cache: dict | None = None

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

        # 默认助手的配置变更时，使 GSV 默认值缓存失效，确保下次回退读到最新值
        if update_request.name == Config.DEFAULT_ASSISTANT_NAME:
            self._default_gsv_cache = None

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

    async def set_assistant(self, assistant_name: str) -> Agent:
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
            agent = Agent(assistant_name)
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

    def get_current_assistant(self) -> Agent | None:
        return self.current_assistant

    def get_current_assistant_name(self) -> str | None:
        return self.current_assistant_name

    def reload_current_assistant(self) -> None:
        """
        重新加载当前助手
        """
        if self.current_assistant_name:
            try:
                agent = Agent(self.current_assistant_name)
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

    async def initialize_default_assistant(self) -> Agent | None:
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

    def _load_default_gsv_setting(self) -> dict:
        """
        读取默认助手的 gsvSetting，作为语音配置的回退源。
        结果缓存在实例上，避免重复读盘；默认助手不存在时返回空字典。
        """
        if self._default_gsv_cache is not None:
            return self._default_gsv_cache

        default_info_path = os.path.join(
            Config.BASE_AGENTS_PATH, Config.DEFAULT_ASSISTANT_NAME, "info.yaml"
        )
        if not os.path.isfile(default_info_path):
            Log.logger.warning(
                f"默认助手 '{Config.DEFAULT_ASSISTANT_NAME}' 不存在，无法提供 GSV 默认配置"
            )
            self._default_gsv_cache = {}
            return self._default_gsv_cache

        try:
            with open(default_info_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self._default_gsv_cache = data.get("gsvSetting", {}) or {}
        except Exception as e:
            Log.logger.error(f"读取默认助手 GSV 配置失败: {e}", exc_info=True)
            self._default_gsv_cache = {}

        return self._default_gsv_cache  # type: ignore

    async def _set_gsv_models(self, agent: Agent) -> None:
        """
        设置当前助手的语音模型路径。
        字段为空或本地资源文件不存在时，回退到默认助手的对应字段与资源。
        """
        is_api = CConfig.config["TTS"]["mode"] == "api"
        gsv = agent.agent_config.gsvSetting

        default_name = Config.DEFAULT_ASSISTANT_NAME
        default_asset_base = f"{Config.BASE_AGENTS_PATH}/{default_name}/assets"
        default_gsv = self._load_default_gsv_setting()

        current_asset_base = f"{Config.BASE_AGENTS_PATH}/{agent.agent_name}/assets"
        is_default_agent = agent.agent_name == default_name

        def resolve_local_file(
            sub_dir: str, file_name: str, default_file_name: str
        ) -> tuple[str, bool]:
            """
            返回 (最终绝对路径, 是否使用了默认助手资源)
            - 字段为空 → 使用默认助手资源
            - 当前助手目录下文件不存在 → 使用默认助手资源
            - 若当前助手本身就是默认助手，则不再回退（避免循环）
            """
            if not file_name:
                if not default_file_name:
                    return "", True
                return (
                    os.path.join(default_asset_base, sub_dir, default_file_name),
                    True,
                )

            candidate = os.path.join(current_asset_base, sub_dir, file_name)
            if not os.path.isfile(candidate):
                if is_default_agent or not default_file_name:
                    return candidate, False
                Log.logger.warning(
                    f"助手 [{agent.agent_name}] 资源不存在: {candidate}，回退至默认助手 [{default_name}]"
                )
                return (
                    os.path.join(default_asset_base, sub_dir, default_file_name),
                    True,
                )
            return candidate, False

        if is_api:
            # API 模式：原样透传，空值由默认助手字段补齐，远端负责解析
            gpt_model_path = gsv.gptModelPath or default_gsv.get("gptModelPath", "")
            sovits_model_path = gsv.sovitsModelPath or default_gsv.get(
                "sovitsModelPath", ""
            )
            ref_audio_path = gsv.refAudioPath or default_gsv.get("refAudioPath", "")
            spk_audio_path = ref_audio_path
            prompt_text = gsv.promptText or default_gsv.get("promptText", "")
            language = gsv.textLang or default_gsv.get("textLang", "zh")
        else:
            # 本地模式：拼绝对路径，缺失/无效时回退默认助手
            gpt_model_path, _ = resolve_local_file(
                "models", gsv.gptModelPath, default_gsv.get("gptModelPath", "")
            )
            sovits_model_path, _ = resolve_local_file(
                "models", gsv.sovitsModelPath, default_gsv.get("sovitsModelPath", "")
            )
            ref_audio_path, audio_use_default = resolve_local_file(
                "audio", gsv.refAudioPath, default_gsv.get("refAudioPath", "")
            )
            spk_audio_path = ref_audio_path

            # 参考音频与参考文本/语言必须配对：用默认音频时强制使用默认文本与语言
            if audio_use_default:
                prompt_text = default_gsv.get("promptText", "")
                language = default_gsv.get("textLang", "zh")
            else:
                prompt_text = gsv.promptText or default_gsv.get("promptText", "")
                language = (
                    gsv.textLang
                    if gsv.textLang in ("zh", "en", "ja")
                    else default_gsv.get("textLang", "zh")
                )

        await ttsService.switch_tts_models(
            gpt_model_path=gpt_model_path,
            sovits_model_path=sovits_model_path,
            spk_audio_path=spk_audio_path,
            ref_audio_path=ref_audio_path,
            prompt_text=prompt_text,
            language=language,
        )
