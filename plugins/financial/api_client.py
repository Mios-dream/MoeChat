"""
Financial API Client
财务系统API客户端，负责与财务记录系统通信
"""

import requests
import json
import time
from typing import Dict, Optional, Any
import logging


class FinancialAPIClient:
    """财务系统API客户端"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化API客户端

        Args:
            config: 插件配置字典
        """
        api_config = config.get("financial_api", {})
        self.base_url = api_config.get("endpoint", "http://localhost:5000")
        self.timeout = api_config.get("timeout", 10)
        self.retry_count = api_config.get("retry_count", 3)
        self.retry_delay = 1  # 重试延迟秒数

        # 设置日志
        self.logger = logging.getLogger(f"{__name__}.FinancialAPIClient")

        # API端点
        self.endpoints = {
            "transaction": f"{self.base_url}/api/transaction",
            "balances": f"{self.base_url}/api/balances",
            "search": f"{self.base_url}/api/transactions/search",
            "monthly_report": f"{self.base_url}/api/report/monthly",
        }

        self.logger.info(f"财务API客户端初始化完成，服务地址: {self.base_url}")

    def add_transaction(self, text: str) -> Dict[str, Any]:
        """
        添加交易记录

        Args:
            text: 用户输入的自然语言文本

        Returns:
            dict: API响应结果
                {
                    "success": bool,
                    "message": str,
                    "data": {...} or None,
                    "api_error": str or None
                }
        """
        payload = {"text": text}

        return self._make_request(
            method="POST",
            url=self.endpoints["transaction"],
            json_data=payload,
            operation_name="添加交易记录",
        )

    def get_balances(self) -> Dict[str, Any]:
        """
        获取所有账户余额

        Returns:
            dict: 余额信息
        """
        return self._make_request(
            method="GET", url=self.endpoints["balances"], operation_name="获取账户余额"
        )

    def search_transactions(
        self,
        keyword: str = None,
        limit: int = 10,
        start_date: str = None,
        end_date: str = None,
    ) -> Dict[str, Any]:
        """
        搜索交易记录

        Args:
            keyword: 搜索关键词
            limit: 最大返回数量
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            dict: 搜索结果
        """
        params = {}
        if keyword:
            params["keyword"] = keyword
        if limit:
            params["limit"] = limit
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        return self._make_request(
            method="GET",
            url=self.endpoints["search"],
            params=params,
            operation_name="搜索交易记录",
        )

    def get_monthly_report(self, year: int = None, month: int = None) -> Dict[str, Any]:
        """
        获取月度财务报表

        Args:
            year: 年份
            month: 月份

        Returns:
            dict: 月度报表数据
        """
        from datetime import datetime

        if not year:
            year = datetime.now().year
        if not month:
            month = datetime.now().month

        params = {"year": year, "month": month}

        return self._make_request(
            method="GET",
            url=self.endpoints["monthly_report"],
            params=params,
            operation_name="获取月度报表",
        )

    def health_check(self) -> bool:
        """
        检查财务API服务是否正常

        Returns:
            bool: 服务是否正常
        """
        try:
            response = requests.get(f"{self.base_url}/api/balances", timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"财务API健康检查失败: {e}")
            return False

    def _make_request(
        self,
        method: str,
        url: str,
        json_data: Dict = None,
        params: Dict = None,
        operation_name: str = "API调用",
    ) -> Dict[str, Any]:
        """
        执行HTTP请求（带重试机制）

        Args:
            method: HTTP方法
            url: 请求URL
            json_data: JSON数据（POST请求用）
            params: URL参数（GET请求用）
            operation_name: 操作名称（用于日志）

        Returns:
            dict: 标准化的响应结果
        """
        last_error = None

        for attempt in range(self.retry_count):
            try:
                self.logger.debug(
                    f"{operation_name} - 尝试 {attempt + 1}/{self.retry_count}"
                )

                # 发起请求
                if method.upper() == "POST":
                    response = requests.post(
                        url,
                        json=json_data,
                        timeout=self.timeout,
                        headers={"Content-Type": "application/json"},
                    )
                else:  # GET
                    response = requests.get(url, params=params, timeout=self.timeout)

                # 检查HTTP状态码
                if response.status_code == 200:
                    try:
                        result = response.json()
                        self.logger.debug(f"{operation_name} - 成功")
                        return result
                    except json.JSONDecodeError as e:
                        return self._error_response(f"JSON解析失败: {e}")

                elif response.status_code == 404:
                    return self._error_response(f"API端点不存在: {url}")

                elif response.status_code == 500:
                    # 500错误可能是临时的，继续重试
                    last_error = f"服务器内部错误 (HTTP 500)"
                    self.logger.warning(f"{operation_name} - {last_error}, 将重试")

                else:
                    return self._error_response(f"HTTP错误: {response.status_code}")

            except requests.exceptions.ConnectTimeout:
                last_error = "连接超时"
                self.logger.warning(f"{operation_name} - {last_error}")

            except requests.exceptions.ConnectionError:
                last_error = "无法连接到财务服务"
                self.logger.warning(f"{operation_name} - {last_error}")

            except requests.exceptions.Timeout:
                last_error = "请求超时"
                self.logger.warning(f"{operation_name} - {last_error}")

            except Exception as e:
                last_error = f"未知错误: {e}"
                self.logger.error(f"{operation_name} - {last_error}")

            # 如果不是最后一次尝试，等待后重试
            if attempt < self.retry_count - 1:
                time.sleep(self.retry_delay)

        # 所有重试都失败了
        self.logger.error(f"{operation_name} - 所有重试失败，最后错误: {last_error}")
        return self._error_response(f"API调用失败: {last_error}")

    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """
        生成标准错误响应

        Args:
            error_message: 错误信息

        Returns:
            dict: 标准错误响应格式
        """
        return {
            "success": False,
            "message": error_message,
            "data": None,
            "api_error": error_message,
        }


# 简单测试代码
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.DEBUG)

    # 模拟配置
    test_config = {
        "financial_api": {
            "endpoint": "http://localhost:5000",
            "timeout": 10,
            "retry_count": 3,
        }
    }

    # 创建客户端
    client = FinancialAPIClient(test_config)

    print("=== 财务API客户端测试 ===")

    # 1. 健康检查
    print(f"服务健康检查: {'✅ 正常' if client.health_check() else '❌ 异常'}")

    # 2. 测试添加交易
    print("\n测试添加交易:")
    result = client.add_transaction("买了咖啡25块用支付宝付的")
    print(f"结果: {result['success']}")
    print(f"消息: {result['message']}")
    if result.get("data"):
        print(f"交易ID: {result['data'].get('transaction_id')}")

    # 3. 测试查询余额
    print("\n测试查询余额:")
    balances = client.get_balances()
    if balances["success"]:
        print(f"账户数量: {len(balances['data']['balances'])}")
    else:
        print(f"查询失败: {balances['message']}")

    print("\n=== 测试完成 ===")
