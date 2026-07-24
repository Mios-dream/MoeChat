"""
记忆工具：由助手自主调用，记录重要信息

助手通过调用此工具记录重要记忆，无需外部 LLM 提取。
工具执行在服务端，结果不转发给客户端（保持沉浸感）。
"""

from tool_system.core.base import ServerTool
from tool_system.core.registry import register_tool
from tool_system.core.enums import ExecutionDomain, ExecutionMode
from my_utils import log as Log


@register_tool(
    domain=ExecutionDomain.SERVER,
    mode=ExecutionMode.SYNC,
    timeout=5.0,
    tags=["memory"],
)
class RememberTool(ServerTool):
    """
    记住重要事情的工具

    当你遇到以下情况时应该使用此工具：
    1. 用户透露了重要的个人信息（关于用户的）
    2. 你和用户之间发生了有意义的互动（关于彼此的）
    3. 用户在对话中分享了有价值的知识（关于世界的）
    4. 你对自己的感受或认知发生了变化（关于自身的）

    注意事项：
    - 只记录真正重要的事情，不要事无巨细
    - 用你自己的视角来描述记忆内容
    - 不要记录已经在其他记忆中重复的内容
    """

    name = "remember"
    description = (
        "记住一段重要信息。当你觉得某件事值得长期记住时使用此工具。"
        "只有真正重要的事情才值得记录，不要事无巨细。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "记忆内容。用你自己的视角描述，"
                "如'用户提到他喜欢吃辣'或'今天和用户一起看了日落，感觉很温暖'",
            },
            "category": {
                "type": "string",
                "enum": ["about_user", "about_us", "about_world", "about_self"],
                "description": "记忆分类：\n"
                "- about_user：关于用户的信息（偏好、经历、习惯等）\n"
                "- about_us：关于你和用户的关系和共同经历\n"
                "- about_world：从对话中学到的新知识\n"
                "- about_self：你对自己认知的变化",
            },
            "importance": {
                "type": "number",
                "description": "这条记忆的重要程度（0.0~1.0）。"
                "非常重要的事情给接近 1.0，普通重要给 0.5~0.7，"
                "细节性的信息给 0.3~0.5。"
                "影响性格和关系的核心记忆应设为 0.9 以上并设为 core 类型。",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "memory_type": {
                "type": "string",
                "enum": ["core", "normal"],
                "description": "记忆类型：\n"
                "- core：永久记忆，永远不会被遗忘（用于极其重要的事）\n"
                "- normal：普通记忆，会随时间逐渐淡忘",
            },
        },
        "required": ["content", "category", "importance", "memory_type"],
    }

    # 全局记忆引擎引用，由外部注入
    _engine = None

    @classmethod
    def set_engine(cls, engine):
        """注入记忆引擎实例"""
        cls._engine = engine

    async def execute(self, **kwargs: str | float) -> str:
        """
        执行记忆记录

        Args:
            content: 记忆内容
            category: 记忆分类
            importance: 重要度（默认 0.5）
            memory_type: 记忆类型（默认 normal）

        Returns:
            JSON 格式结果
        """
        engine = self.__class__._engine
        if engine is None:
            return self.result_error("记忆引擎未就绪")

        raw_content = kwargs.get("content", "")
        content = str(raw_content).strip() if raw_content else ""
        category = str(kwargs.get("category", "about_user"))
        importance = float(kwargs.get("importance", 0.5))
        memory_type = str(kwargs.get("memory_type", "normal"))

        if not content:
            return self.result_error("记忆内容不能为空")

        mem_id = engine.add_memory(
            content=content.strip(),
            category=category,
            importance=importance,
            memory_type=memory_type,
        )

        if mem_id is None:
            return self.result_json({
                "status": "exists",
                "message": "已存在相似的记忆，无需重复记录",
                "memory_id": None,
            })

        return self.result_json({
            "status": "ok",
            "message": "已记住",
            "memory_id": mem_id,
        })


@register_tool(
    domain=ExecutionDomain.SERVER,
    mode=ExecutionMode.SYNC,
    timeout=5.0,
    tags=["memory"],
)
class RecallTool(ServerTool):
    """
    主动回忆记忆的工具

    当你想确认某件事、回忆相关细节时使用此工具进行检索。
    这让你可以主动查询自己的记忆，而不是被动等待记忆被注入。
    """

    name = "recall"
    description = (
        "主动回忆记忆。当你对某件事印象模糊、想确认细节、"
        "或需要回想相关信息时使用此工具进行检索。"
        "注意：不要对每句话都使用，只在真正需要回忆时使用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "你想回忆什么？描述你想查找的内容或关键词",
            },
            "top_k": {
                "type": "integer",
                "description": "返回几条相关记忆（默认 3）",
                "default": 3,
            },
        },
        "required": ["query"],
    }

    # 全局记忆引擎引用，由外部注入
    _engine = None

    @classmethod
    def set_engine(cls, engine):
        """注入记忆引擎实例"""
        cls._engine = engine

    async def execute(self, **kwargs: str | float) -> str:
        """
        执行记忆检索

        Args:
            query: 查询内容
            top_k: 返回条数

        Returns:
            JSON 格式结果（记忆列表）
        """
        engine = self.__class__._engine
        if engine is None:
            return self.result_error("记忆引擎未就绪")

        raw_query = kwargs.get("query", "")
        query = str(raw_query).strip() if raw_query else ""
        raw_top_k = kwargs.get("top_k", 3)
        top_k = int(raw_top_k) if raw_top_k else 3

        if not query:
            return self.result_error("查询内容不能为空")

        top_k = max(1, min(20, top_k))

        try:
            results = engine.search_raw(query, top_k=top_k, include_diaries=True)
        except Exception as e:
            Log.logger.warning(f"[recall] 检索失败: {e}")
            return self.result_error("检索失败")

        if not results:
            return self.result_json({
                "status": "empty",
                "message": "没有找到相关记忆",
                "results": [],
                "count": 0,
            })

        formatted = []
        for r in results:
            entry = {
                "content": r["content"],
                "score": r["score"],
            }
            if r.get("source") == "diary":
                entry["type"] = "日记"
                entry["date"] = r.get("day", "")
            else:
                cat_label = {
                    "about_user": "关于用户",
                    "about_us": "关于彼此",
                    "about_world": "关于世界",
                    "about_self": "关于自身",
                }.get(r["category"], r["category"])
                entry["type"] = f"记忆[{cat_label}]"
                if r["memory_type"] == "core":
                    entry["type"] = "核心" + entry["type"]
            formatted.append(entry)

        return self.result_json({
            "status": "ok",
            "message": f"找到 {len(formatted)} 条相关记忆",
            "results": formatted,
            "count": len(formatted),
        })
