"""
Financial Plugin Main Logic
财务插件主逻辑模块
"""

import json
import yaml
import os
import logging
from typing import Dict, Optional, Any
from datetime import datetime

from .api_client import FinancialAPIClient
from .state_manager import SessionStateManager


class FinancialPlugin:
    """财务插件主类"""

    def __init__(self, config_path=None):
        self.name = "financial"
        self.version = "1.0.0"
        self.enabled = False
        self.config = {}

        if config_path is None:
            current_file_path = os.path.abspath(__file__)
            plugin_dir = os.path.dirname(current_file_path)
            config_path = os.path.join(plugin_dir, "config.yml")

        # 组件
        self.api_client = None
        self.state_manager = None
        self.action_triggers: Set[str] = set()  # 存储所有action关键词

        # 日志
        # 日志
        self.logger = logging.getLogger(f"{__name__}.FinancialPlugin")

        # 加载配置
        self.config_path = config_path  # 使用我们刚刚计算出的绝对路径
        self.load_config()

    def load_config(self):
        """加载插件配置"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)
                    self.config = config_data.get("financial_plugin", {})
                    self.enabled = self.config.get("enabled", False)

                    self.logger.info(f"配置加载成功，插件启用状态: {self.enabled}")
            else:
                self.logger.error(f"配置文件不存在: {self.config_path}")
                self.enabled = False

        except Exception as e:
            self.logger.error(f"配置加载失败: {e}")
            self.enabled = False

    def initialize(self):
        """初始化插件"""
        if not self.enabled:
            self.logger.info("插件未启用，跳过初始化")
            return False

        try:
            # 初始化API客户端
            self.api_client = FinancialAPIClient(self.config)
            # 加载动作关键词触发器
            self._load_action_triggers()

            # 检查财务服务是否可用
            if not self.api_client.health_check():
                self.logger.error("财务API服务不可用")
                return False

            # 初始化状态管理器
            self.state_manager = SessionStateManager(self.config)

            self.logger.info(f"✅ 财务插件 v{self.version} 初始化成功")
            return True

        except Exception as e:
            self.logger.error(f"❌ 财务插件初始化失败: {e}")
            return False

        # 加载关键词

    def _load_action_triggers(self):
        """从balancer服务的keywords.json中加载动作关键词"""
        try:
            # 根据当前文件位置，构建keywords.json的路径
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            # Balancer服务位于同级的balancer文件夹内
            keywords_path = os.path.join(
                plugin_dir, "balancer", "config", "keywords.json"
            )

            if not os.path.exists(keywords_path):
                self.logger.error(f"关键词文件不存在: {keywords_path}")
                return

            with open(keywords_path, "r", encoding="utf-8") as f:
                keywords_data = json.load(f)

            actions = keywords_data.get("actions", {})
            for action_type in actions.values():
                for lang_keywords in action_type.values():
                    for keyword in lang_keywords:
                        self.action_triggers.add(
                            keyword.lower()
                        )  # 统一转为小写方便匹配

            self.logger.info(f"成功加载 {len(self.action_triggers)} 个财务动作触发词。")

        except Exception as e:
            self.logger.error(f"加载财务动作触发词失败: {e}")

    def process_message(
        self, user_message: str, session_id: str, context: Dict = None
    ) -> Dict[str, Any]:
        """处理用户消息的主函数"""
        if not self.enabled or not self.api_client or not self.state_manager:
            return self._no_financial_result()

        # 前置关键词过滤,检查消息中动作触发词, 无视大小写

        if not any(trigger in user_message.lower() for trigger in self.action_triggers):
            self.logger.debug("消息未命中任何财务动作关键词，跳过处理。")
            return self._no_financial_result()

        # 如果通过了上面的过滤，说明这很可能是一条财务消息，再执行后续的复杂逻辑
        self.logger.info("消息命中财务动作关键词，进入详细处理流程...")
        try:
            # ... (后续的 try...except... 逻辑完全保持不变) ...
            self.logger.debug(f"处理消息: {user_message[:50]}...")
            has_pending = self.state_manager.has_pending_transaction(session_id)
            if has_pending:
                return self._handle_pending_transaction(user_message, session_id)
            else:
                return self._handle_new_message(user_message, session_id)
        except Exception as e:
            self.logger.error(f"消息处理异常: {e}")
            return self._error_result(f"插件处理异常: {str(e)}")

    def _handle_new_message(self, user_message: str, session_id: str) -> Dict[str, Any]:
        """处理新消息"""
        # 调用财务API检测
        api_result = self.api_client.add_transaction(user_message)

        if api_result["success"]:
            # 完整信息，直接记录成功
            self.logger.info(f"交易记录成功: {api_result['data']['transaction_id']}")
            return self._success_result(api_result, "transaction_completed")

        else:
            # 检查是否是财务相关但信息不完整
            if self._is_financial_related(api_result):
                # 创建pending事务
                pending = self.state_manager.create_pending_transaction(
                    session_id, user_message, api_result
                )

                self.logger.info(f"创建pending事务，缺失信息: {pending.missing_info}")
                return self._incomplete_result(pending, "transaction_pending")
            else:
                # 非财务相关消息
                return self._no_financial_result()

    def _handle_pending_transaction(
        self, user_message: str, session_id: str
    ) -> Dict[str, Any]:
        """处理有pending事务的情况"""
        # 构造合并后的文本进行API调用
        pending = self.state_manager.get_latest_pending(session_id)
        if not pending:
            # pending已过期，按新消息处理
            return self._handle_new_message(user_message, session_id)

        # 创建合并文本
        combined_text = self._create_combined_text(pending, user_message)

        # 调用API
        api_result = self.api_client.add_transaction(combined_text)

        if api_result["success"]:
            # 信息完整，记录成功
            completed = self.state_manager.complete_pending_transaction(session_id)
            self.logger.info(f"Pending事务完成: {api_result['data']['transaction_id']}")
            return self._success_result(api_result, "transaction_completed", completed)

        else:
            # 仍然不完整，更新pending事务
            updated_pending = self.state_manager.update_pending_transaction(
                session_id, user_message, api_result
            )

            if updated_pending:
                self.logger.info(
                    f"更新pending事务，剩余缺失: {updated_pending.missing_info}"
                )
                return self._incomplete_result(
                    updated_pending, "transaction_pending_updated"
                )
            else:
                # pending已过期，按新消息处理
                return self._handle_new_message(user_message, session_id)

    def _create_combined_text(self, pending, new_message: str) -> str:
        """创建合并后的文本"""
        # 简单合并策略：将对话历史和新消息组合
        all_texts = pending.conversation_history + [new_message]
        combined = " ".join(all_texts)

        self.logger.debug(f"合并文本: {combined}")
        return combined

    def _is_financial_related(self, api_result: Dict) -> bool:
        """判断是否与财务相关"""
        # 如果API返回了部分信息或明确的missing_info，说明是财务相关
        if api_result.get("data") and api_result["data"].get("missing_info"):
            return True

        # 检查错误信息中是否提到财务相关内容
        message = api_result.get("message", "").lower()
        financial_indicators = ["missing_info", "缺少信息", "amount", "action"]

        return any(indicator in message for indicator in financial_indicators)

    def _success_result(
        self, api_result: Dict, action_type: str, completed_pending=None
    ) -> Dict[str, Any]:
        """生成成功结果"""
        transaction_data = api_result["data"]["transaction_data"]

        result = {
            "financial_detected": True,
            "status": "success",
            "action_type": action_type,
            "timestamp": datetime.now().isoformat(),
            "transaction_data": transaction_data,
            "llm_context": {
                "type": "financial_success",
                "message": "用户完成了一笔财务记录",
                "transaction_info": {
                    "action": transaction_data["action"],
                    "amount": transaction_data["amount"],
                    "category": transaction_data.get("category"),
                    "debit_account": transaction_data["debit_account"],
                    "credit_account": transaction_data["credit_account"],
                },
                "suggestion_for_llm": self._generate_success_suggestion(
                    transaction_data
                ),
            },
        }

        # 如果是从pending完成的，添加额外信息
        if completed_pending:
            result["completed_conversation"] = completed_pending.conversation_history
            result["llm_context"]["conversation_turns"] = len(
                completed_pending.conversation_history
            )

        return result

    def _incomplete_result(self, pending, action_type: str) -> Dict[str, Any]:
        """生成信息不完整的结果"""
        return {
            "financial_detected": True,
            "status": "incomplete",
            "action_type": action_type,
            "timestamp": datetime.now().isoformat(),
            "pending_transaction": pending.to_dict(),
            "llm_context": {
                "type": "financial_incomplete",
                "message": "检测到财务意图但信息不完整",
                "missing_info": pending.missing_info,
                "conversation_history": pending.conversation_history,
                "extracted_info": pending.extracted_info,
                "suggestion_for_llm": self._generate_incomplete_suggestion(pending),
            },
        }

    def _no_financial_result(self) -> Dict[str, Any]:
        """生成非财务相关结果"""
        return {
            "financial_detected": False,
            "status": "not_financial",
            "timestamp": datetime.now().isoformat(),
            "message": "未检测到财务相关内容",
        }

    def _error_result(self, error_message: str) -> Dict[str, Any]:
        """生成错误结果"""
        return {
            "financial_detected": False,
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "llm_context": {
                "type": "financial_error",
                "message": "财务插件处理出现错误",
                "suggestion_for_llm": "财务记录功能暂时不可用，可以正常对话",
            },
        }

    def _generate_success_suggestion(self, transaction_data: Dict) -> str:
        """生成成功时给LLM的建议"""
        action = transaction_data["action"]
        amount = transaction_data["amount"]
        category = transaction_data.get("category", "")

        if action == "expense":
            return f"用户刚记录了一笔{category}{amount}元的支出，可以确认记录并给予适当的反馈或建议"
        elif action == "income":
            return f"用户记录了{amount}元的收入，可以表示开心并鼓励用户"
        elif action == "loan_payment":
            return f"用户还款{amount}元，可以表扬用户的负责任行为"
        else:
            return f"用户完成了一笔{amount}元的财务记录，给予确认和反馈"

    def _generate_incomplete_suggestion(self, pending) -> str:
        """生成信息不完整时给LLM的建议"""
        missing = pending.missing_info

        if "amount" in missing:
            return "用户提到了消费行为但没说金额，询问具体花了多少钱"
        elif "payment_method" in missing:
            return "用户说了金额但没说用什么支付，询问支付方式（支付宝、微信、现金等）"
        elif "action" in missing:
            return "用户提到了金额但行为不明确，询问是买了什么还是其他消费"
        else:
            return f"用户的财务信息不完整，缺少：{', '.join(missing)}，引导用户补充"

    def cancel_pending_transaction(self, session_id: str) -> bool:
        """取消当前的pending事务"""
        if not self.state_manager:
            return False

        return self.state_manager.cancel_pending_transaction(session_id)

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """获取会话状态"""
        if not self.state_manager:
            return {"error": "状态管理器未初始化"}

        return self.state_manager.get_session_status(session_id)

    def cleanup(self):
        """清理插件资源"""
        self.logger.info("开始清理财务插件...")

        if self.state_manager:
            self.state_manager.cleanup()

        self.logger.info("🧹 财务插件资源清理完成")


# 插件实例（单例模式）
_plugin_instance = None


def get_plugin():
    """获取插件实例（单例模式）"""
    global _plugin_instance
    if _plugin_instance is None:
        _plugin_instance = FinancialPlugin()
    return _plugin_instance


def initialize_plugin():
    """初始化插件的便捷函数"""
    plugin = get_plugin()
    return plugin.initialize()


def process_message(
    user_message: str, session_id: str, context: Dict = None
) -> Dict[str, Any]:
    """处理消息的便捷函数"""
    plugin = get_plugin()
    return plugin.process_message(user_message, session_id, context)


# 用于MoeChat集成的主要接口
def financial_plugin_hook(
    user_message: str, session_id: str, context: Dict = None
) -> Dict[str, Any]:
    """MoeChat插件钩子函数"""
    return process_message(user_message, session_id, context)


# 测试代码
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=== 财务插件完整测试 ===")

    # 1. 初始化插件
    plugin = get_plugin()
    init_success = plugin.initialize()
    print(f"插件初始化: {'成功' if init_success else '失败'}")

    if not init_success:
        print("插件初始化失败，请检查配置和财务服务")
        exit(1)

    # 2. 模拟多轮对话
    session_id = "test_session_001"

    print("\n--- 第1轮：信息不完整 ---")
    result1 = plugin.process_message("我吃了外卖", session_id)
    print(f"检测到财务: {result1['financial_detected']}")
    print(f"状态: {result1['status']}")
    if result1["financial_detected"]:
        print(f"LLM建议: {result1['llm_context']['suggestion_for_llm']}")

    print("\n--- 第2轮：补充金额 ---")
    result2 = plugin.process_message("花了50块", session_id)
    print(f"检测到财务: {result2['financial_detected']}")
    print(f"状态: {result2['status']}")
    if result2["financial_detected"]:
        print(f"LLM建议: {result2['llm_context']['suggestion_for_llm']}")

    print("\n--- 第3轮：完成记录 ---")
    result3 = plugin.process_message("微信付的", session_id)
    print(f"检测到财务: {result3['financial_detected']}")
    print(f"状态: {result3['status']}")
    if result3["financial_detected"]:
        print(f"交易金额: {result3['transaction_data']['amount']}")
        print(f"LLM建议: {result3['llm_context']['suggestion_for_llm']}")

    print("\n--- 第4轮：非财务消息 ---")
    result4 = plugin.process_message("今天天气真好", session_id)
    print(f"检测到财务: {result4['financial_detected']}")
    print(f"状态: {result4['status']}")

    print("\n=== 测试完成 ===")

    # 清理
    plugin.cleanup()
