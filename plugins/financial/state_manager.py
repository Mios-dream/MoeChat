"""
Session State Manager
会话状态管理器，负责处理多轮对话中的财务信息拼接
"""

import time
import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging


class PendingTransaction:
    """待完成的交易记录"""

    def __init__(self, session_id: str, initial_text: str, extracted_info: Dict = None):
        self.session_id = session_id
        self.transaction_id = f"{session_id}_{int(time.time() * 1000)}"
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self.status = "pending"  # pending, completed, expired, cancelled

        # 原始信息
        self.initial_text = initial_text
        self.conversation_history = [initial_text]

        # 提取的信息
        self.extracted_info = extracted_info or {}
        self.missing_info = self.extracted_info.get("missing_info", [])

    def update_info(self, new_text: str, new_extracted: Dict):
        """更新交易信息"""
        self.last_updated = datetime.now()
        self.conversation_history.append(new_text)

        # 合并新信息到已有信息中
        if new_extracted.get("data"):
            for key, value in new_extracted["data"].items():
                if value is not None:  # 只更新有值的字段
                    self.extracted_info[key] = value

        # 更新缺失信息列表
        self.missing_info = new_extracted.get("missing_info", [])

    def is_complete(self) -> bool:
        """检查信息是否完整"""
        return len(self.missing_info) == 0

    def is_expired(self, timeout_seconds: int) -> bool:
        """检查是否超时"""
        return (datetime.now() - self.last_updated).total_seconds() > timeout_seconds

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "transaction_id": self.transaction_id,
            "session_id": self.session_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "initial_text": self.initial_text,
            "conversation_history": self.conversation_history,
            "extracted_info": self.extracted_info,
            "missing_info": self.missing_info,
            "is_complete": self.is_complete(),
        }


class SessionStateManager:
    """会话状态管理器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化状态管理器

        Args:
            config: 插件配置
        """
        session_config = config.get("session", {})
        self.timeout = session_config.get("timeout", 300)  # 5分钟超时
        self.max_pending = session_config.get("max_pending", 5)  # 最大pending数量

        # 存储所有session的pending事务
        self.pending_transactions: Dict[str, List[PendingTransaction]] = {}

        # 日志
        self.logger = logging.getLogger(f"{__name__}.SessionStateManager")
        self.logger.info("会话状态管理器初始化完成")

    def create_pending_transaction(
        self, session_id: str, text: str, api_result: Dict
    ) -> PendingTransaction:
        """
        创建待完成的交易记录

        Args:
            session_id: 会话ID
            text: 用户输入文本
            api_result: 财务API返回的结果

        Returns:
            PendingTransaction: 创建的待完成交易
        """
        # 清理过期的事务
        self._cleanup_expired_transactions(session_id)

        # 检查是否超过最大数量
        if self._get_pending_count(session_id) >= self.max_pending:
            self.logger.warning(f"会话 {session_id} 的pending事务数量已达上限")
            # 清除最旧的pending事务
            self._remove_oldest_pending(session_id)

        # 创建新的pending事务
        pending = PendingTransaction(session_id, text, api_result.get("data"))

        # 存储
        if session_id not in self.pending_transactions:
            self.pending_transactions[session_id] = []

        self.pending_transactions[session_id].append(pending)

        self.logger.info(f"创建pending事务: {pending.transaction_id}")
        self.logger.debug(f"缺失信息: {pending.missing_info}")

        return pending

    def update_pending_transaction(
        self, session_id: str, new_text: str, api_result: Dict
    ) -> Optional[PendingTransaction]:
        """
        更新最新的pending事务

        Args:
            session_id: 会话ID
            new_text: 新的用户输入
            api_result: API返回结果

        Returns:
            PendingTransaction or None: 更新的事务，如果没有pending事务则返回None
        """
        pending_list = self.pending_transactions.get(session_id, [])

        if not pending_list:
            return None

        # 获取最新的pending事务
        latest_pending = pending_list[-1]

        # 检查是否过期
        if latest_pending.is_expired(self.timeout):
            self.logger.info(f"Pending事务已过期: {latest_pending.transaction_id}")
            self._remove_pending_transaction(session_id, latest_pending.transaction_id)
            return None

        # 更新信息
        latest_pending.update_info(new_text, api_result)

        self.logger.info(f"更新pending事务: {latest_pending.transaction_id}")
        self.logger.debug(f"剩余缺失信息: {latest_pending.missing_info}")

        return latest_pending

    def complete_pending_transaction(
        self, session_id: str, transaction_id: str = None
    ) -> Optional[PendingTransaction]:
        """
        完成pending事务

        Args:
            session_id: 会话ID
            transaction_id: 事务ID，如果为None则完成最新的事务

        Returns:
            PendingTransaction or None: 完成的事务
        """
        pending_list = self.pending_transactions.get(session_id, [])

        if not pending_list:
            return None

        # 找到要完成的事务
        target_pending = None
        if transaction_id:
            for pending in pending_list:
                if pending.transaction_id == transaction_id:
                    target_pending = pending
                    break
        else:
            target_pending = pending_list[-1]  # 最新的事务

        if not target_pending:
            return None

        # 标记为完成并移除
        target_pending.status = "completed"
        self._remove_pending_transaction(session_id, target_pending.transaction_id)

        self.logger.info(f"完成pending事务: {target_pending.transaction_id}")

        return target_pending

    def cancel_pending_transaction(
        self, session_id: str, transaction_id: str = None
    ) -> bool:
        """
        取消pending事务

        Args:
            session_id: 会话ID
            transaction_id: 事务ID，如果为None则取消最新的事务

        Returns:
            bool: 是否成功取消
        """
        pending_list = self.pending_transactions.get(session_id, [])

        if not pending_list:
            return False

        # 找到要取消的事务
        target_id = transaction_id
        if not target_id:
            target_id = pending_list[-1].transaction_id

        success = self._remove_pending_transaction(session_id, target_id)

        if success:
            self.logger.info(f"取消pending事务: {target_id}")

        return success

    def get_latest_pending(self, session_id: str) -> Optional[PendingTransaction]:
        """
        获取最新的pending事务

        Args:
            session_id: 会话ID

        Returns:
            PendingTransaction or None: 最新的pending事务
        """
        pending_list = self.pending_transactions.get(session_id, [])

        if not pending_list:
            return None

        latest = pending_list[-1]

        # 检查是否过期
        if latest.is_expired(self.timeout):
            self.logger.info(f"最新pending事务已过期: {latest.transaction_id}")
            self._remove_pending_transaction(session_id, latest.transaction_id)
            return None

        return latest

    def has_pending_transaction(self, session_id: str) -> bool:
        """
        检查是否有pending事务

        Args:
            session_id: 会话ID

        Returns:
            bool: 是否有pending事务
        """
        # 先清理过期事务
        self._cleanup_expired_transactions(session_id)

        pending_list = self.pending_transactions.get(session_id, [])
        return len(pending_list) > 0

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """
        获取会话状态信息

        Args:
            session_id: 会话ID

        Returns:
            dict: 会话状态信息
        """
        # 清理过期事务
        self._cleanup_expired_transactions(session_id)

        pending_list = self.pending_transactions.get(session_id, [])

        return {
            "session_id": session_id,
            "pending_count": len(pending_list),
            "has_pending": len(pending_list) > 0,
            "pending_transactions": [p.to_dict() for p in pending_list],
            "last_activity": max([p.last_updated for p in pending_list]).isoformat()
            if pending_list
            else None,
        }

    def _get_pending_count(self, session_id: str) -> int:
        """获取pending事务数量"""
        return len(self.pending_transactions.get(session_id, []))

    def _remove_oldest_pending(self, session_id: str):
        """移除最旧的pending事务"""
        pending_list = self.pending_transactions.get(session_id, [])
        if pending_list:
            oldest = pending_list[0]
            oldest.status = "expired"
            pending_list.pop(0)
            self.logger.info(f"移除最旧的pending事务: {oldest.transaction_id}")

    def _remove_pending_transaction(self, session_id: str, transaction_id: str) -> bool:
        """移除指定的pending事务"""
        pending_list = self.pending_transactions.get(session_id, [])

        for i, pending in enumerate(pending_list):
            if pending.transaction_id == transaction_id:
                pending_list.pop(i)

                # 如果列表为空，删除整个session记录
                if not pending_list:
                    del self.pending_transactions[session_id]

                return True

        return False

    def _cleanup_expired_transactions(self, session_id: str):
        """清理过期的pending事务"""
        if session_id not in self.pending_transactions:
            return

        pending_list = self.pending_transactions[session_id]
        expired_count = 0

        # 反向遍历以避免索引问题
        for i in range(len(pending_list) - 1, -1, -1):
            pending = pending_list[i]
            if pending.is_expired(self.timeout):
                pending.status = "expired"
                pending_list.pop(i)
                expired_count += 1

        if expired_count > 0:
            self.logger.info(f"清理了 {expired_count} 个过期的pending事务")

        # 如果列表为空，删除整个session记录
        if not pending_list:
            del self.pending_transactions[session_id]

    def cleanup_all_expired(self):
        """清理所有过期的事务"""
        expired_sessions = []

        for session_id in list(self.pending_transactions.keys()):
            self._cleanup_expired_transactions(session_id)
            if session_id not in self.pending_transactions:
                expired_sessions.append(session_id)

        if expired_sessions:
            self.logger.info(f"清理了 {len(expired_sessions)} 个会话的所有过期事务")

    def cleanup(self):
        """清理资源"""
        self.logger.info("开始清理状态管理器...")

        total_pending = sum(
            len(pending_list) for pending_list in self.pending_transactions.values()
        )
        if total_pending > 0:
            self.logger.warning(f"仍有 {total_pending} 个pending事务未完成")

        self.pending_transactions.clear()
        self.logger.info("状态管理器清理完成")


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # 模拟配置
    test_config = {
        "session": {
            "timeout": 10,  # 测试用短超时时间
            "max_pending": 3,
        }
    }

    manager = SessionStateManager(test_config)

    print("=== 会话状态管理器测试 ===")

    # 模拟API返回结果
    incomplete_result = {
        "success": False,
        "data": {
            "action": None,
            "amount": None,
            "category": "餐饮费用",
            "description": "我吃了外卖",
        },
        "missing_info": ["action", "amount"],
    }

    partial_result = {
        "success": False,
        "data": {
            "action": "expense",
            "amount": 50.0,
            "category": "餐饮费用",
            "description": "花了50块",
        },
        "missing_info": ["payment_method"],
    }

    # 测试创建pending事务
    print("1. 创建pending事务")
    pending = manager.create_pending_transaction(
        "user123", "我吃了外卖", incomplete_result
    )
    print(f"   创建成功: {pending.transaction_id}")
    print(f"   缺失信息: {pending.missing_info}")

    # 测试更新pending事务
    print("\n2. 更新pending事务")
    updated = manager.update_pending_transaction("user123", "花了50块", partial_result)
    if updated:
        print(f"   更新成功: {updated.transaction_id}")
        print(f"   对话历史: {updated.conversation_history}")
        print(f"   剩余缺失: {updated.missing_info}")

    # 测试会话状态
    print("\n3. 查看会话状态")
    status = manager.get_session_status("user123")
    print(f"   Pending数量: {status['pending_count']}")
    print(f"   有pending: {status['has_pending']}")

    # 测试完成事务
    print("\n4. 完成事务")
    completed = manager.complete_pending_transaction("user123")
    if completed:
        print(f"   完成事务: {completed.transaction_id}")
        print(f"   对话历史: {completed.conversation_history}")

    print("\n=== 测试完成 ===")
