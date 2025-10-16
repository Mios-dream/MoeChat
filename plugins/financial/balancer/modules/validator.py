import json
import os
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from decimal import Decimal, InvalidOperation


class TransactionValidator:
    def __init__(self):
        self.config_path = "config"
        self.accounts = self._load_config("accounts.json")
        self.categories = self._load_config("categories.json")

    def _load_config(self, filename: str) -> dict:
        """加载配置文件"""
        try:
            with open(
                os.path.join(self.config_path, filename), "r", encoding="utf-8"
            ) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def validate_transaction(self, transaction_data: dict) -> dict:
        """
        验证交易数据的完整性和合法性

        Args:
            transaction_data: 交易数据字典

        Returns:
            dict: 验证结果
                {
                    'is_valid': bool,
                    'errors': List[str],      # 错误信息
                    'warnings': List[str],    # 警告信息
                    'suggestions': List[str], # 建议信息
                    'cleaned_data': dict      # 清理后的数据
                }
        """
        errors = []
        warnings = []
        suggestions = []
        cleaned_data = transaction_data.copy()

        # 验证必填字段
        required_fields = [
            "action",
            "amount",
            "description",
            "debit_account",
            "credit_account",
        ]
        for field in required_fields:
            if not transaction_data.get(field):
                errors.append(f"缺少必填字段: {field}")

        # 如果有必填字段缺失，直接返回
        if errors:
            return {
                "is_valid": False,
                "errors": errors,
                "warnings": warnings,
                "suggestions": suggestions,
                "cleaned_data": cleaned_data,
            }

        # 验证动作类型
        action_result = self._validate_action(transaction_data.get("action"))
        if action_result["errors"]:
            errors.extend(action_result["errors"])
        if action_result["warnings"]:
            warnings.extend(action_result["warnings"])
        if action_result["cleaned_value"] is not None:
            cleaned_data["action"] = action_result["cleaned_value"]

        # 验证金额
        amount_result = self._validate_amount(transaction_data.get("amount"))
        if amount_result["errors"]:
            errors.extend(amount_result["errors"])
        if amount_result["warnings"]:
            warnings.extend(amount_result["warnings"])
        if amount_result["cleaned_value"] is not None:
            cleaned_data["amount"] = amount_result["cleaned_value"]

        # 验证描述
        description_result = self._validate_description(
            transaction_data.get("description")
        )
        if description_result["warnings"]:
            warnings.extend(description_result["warnings"])
        if description_result["cleaned_value"] is not None:
            cleaned_data["description"] = description_result["cleaned_value"]

        # 验证账户
        debit_result = self._validate_account(
            transaction_data.get("debit_account"), "debit"
        )
        credit_result = self._validate_account(
            transaction_data.get("credit_account"), "credit"
        )

        if debit_result["errors"]:
            errors.extend(debit_result["errors"])
        if credit_result["errors"]:
            errors.extend(credit_result["errors"])
        if debit_result["warnings"]:
            warnings.extend(debit_result["warnings"])
        if credit_result["warnings"]:
            warnings.extend(credit_result["warnings"])

        # 验证分类
        if transaction_data.get("category"):
            category_result = self._validate_category(
                transaction_data.get("category"), transaction_data.get("action")
            )
            if category_result["warnings"]:
                warnings.extend(category_result["warnings"])
            if category_result["suggestions"]:
                suggestions.extend(category_result["suggestions"])

        # 验证借贷逻辑
        logic_result = self._validate_accounting_logic(
            transaction_data.get("action"),
            transaction_data.get("debit_account"),
            transaction_data.get("credit_account"),
        )
        if logic_result["errors"]:
            errors.extend(logic_result["errors"])
        if logic_result["warnings"]:
            warnings.extend(logic_result["warnings"])
        if logic_result["suggestions"]:
            suggestions.extend(logic_result["suggestions"])

        # 验证置信度
        if transaction_data.get("confidence") is not None:
            confidence_result = self._validate_confidence(
                transaction_data.get("confidence")
            )
            if confidence_result["warnings"]:
                warnings.extend(confidence_result["warnings"])
            if confidence_result["cleaned_value"] is not None:
                cleaned_data["confidence"] = confidence_result["cleaned_value"]

        # 业务规则验证
        business_result = self._validate_business_rules(cleaned_data)
        if business_result["warnings"]:
            warnings.extend(business_result["warnings"])
        if business_result["suggestions"]:
            suggestions.extend(business_result["suggestions"])

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
            "cleaned_data": cleaned_data,
        }

    def _validate_action(self, action: str) -> dict:
        """验证动作类型"""
        errors = []
        warnings = []
        cleaned_value = None

        if not action:
            errors.append("动作类型不能为空")
            return {
                "errors": errors,
                "warnings": warnings,
                "cleaned_value": cleaned_value,
            }

        valid_actions = ["expense", "income", "transfer", "loan_payment"]
        if action not in valid_actions:
            errors.append(
                f"无效的动作类型: {action}，有效值为: {', '.join(valid_actions)}"
            )
        else:
            cleaned_value = action.lower()

        return {"errors": errors, "warnings": warnings, "cleaned_value": cleaned_value}

    def _validate_amount(self, amount) -> dict:
        """验证金额"""
        errors = []
        warnings = []
        cleaned_value = None

        if amount is None:
            errors.append("金额不能为空")
            return {
                "errors": errors,
                "warnings": warnings,
                "cleaned_value": cleaned_value,
            }

        try:
            # 转换为Decimal以避免浮点数精度问题
            amount_decimal = Decimal(str(amount))

            if amount_decimal <= 0:
                errors.append("金额必须大于0")
            elif amount_decimal > Decimal("999999999.99"):
                errors.append("金额过大，超过系统限制")
            else:
                # 保留两位小数
                cleaned_value = float(amount_decimal.quantize(Decimal("0.01")))

                # 金额合理性警告
                if cleaned_value > 100000:
                    warnings.append("金额较大，请确认是否正确")
                elif cleaned_value < 0.01:
                    warnings.append("金额过小，可能影响统计精度")

        except (InvalidOperation, ValueError, TypeError):
            errors.append("无效的金额格式")

        return {"errors": errors, "warnings": warnings, "cleaned_value": cleaned_value}

    def _validate_description(self, description: str) -> dict:
        """验证描述"""
        warnings = []
        cleaned_value = None

        if not description:
            return {"warnings": [], "cleaned_value": None}

        # 清理描述
        cleaned_desc = description.strip()

        if len(cleaned_desc) > 500:
            warnings.append("描述过长，已截断至500字符")
            cleaned_desc = cleaned_desc[:500]
        elif len(cleaned_desc) < 2:
            warnings.append("描述过短，建议提供更详细的信息")

        cleaned_value = cleaned_desc

        return {"warnings": warnings, "cleaned_value": cleaned_value}

    def _validate_account(self, account_name: str, position: str) -> dict:
        """验证账户名称"""
        errors = []
        warnings = []

        if not account_name:
            errors.append(f"{position}账户不能为空")
            return {"errors": errors, "warnings": warnings}

        # 检查账户是否在配置中存在
        all_accounts = []
        for account_type in [
            "asset_accounts",
            "liability_accounts",
            "equity_accounts",
            "revenue_accounts",
            "expense_accounts",
        ]:
            accounts = self.accounts.get(account_type, {})
            all_accounts.extend(accounts.keys())

        if account_name not in all_accounts:
            warnings.append(
                f"账户 '{account_name}' 不在预定义账户列表中，将作为新账户创建"
            )

        return {"errors": errors, "warnings": warnings}

    def _validate_category(self, category: str, action: str) -> dict:
        """验证分类"""
        warnings = []
        suggestions = []

        if not category:
            return {"warnings": warnings, "suggestions": suggestions}

        # 检查分类是否匹配动作类型
        if action == "expense":
            valid_categories = list(
                self.categories.get("expense_categories", {}).keys()
            )
            if category not in valid_categories:
                warnings.append(f"费用分类 '{category}' 不在预定义列表中")
                suggestions.append(f"建议使用: {', '.join(valid_categories[:5])}")

        elif action == "income":
            valid_categories = list(self.categories.get("income_categories", {}).keys())
            if category not in valid_categories:
                warnings.append(f"收入分类 '{category}' 不在预定义列表中")
                suggestions.append(f"建议使用: {', '.join(valid_categories)}")

        return {"warnings": warnings, "suggestions": suggestions}

    def _validate_accounting_logic(
        self, action: str, debit_account: str, credit_account: str
    ) -> dict:
        """验证会计逻辑"""
        errors = []
        warnings = []
        suggestions = []

        if not all([action, debit_account, credit_account]):
            return {"errors": errors, "warnings": warnings, "suggestions": suggestions}

        # 检查同一账户不能同时作为借贷方
        if debit_account == credit_account:
            errors.append("借方和贷方不能是同一账户")

        # 根据动作类型验证账户类型逻辑
        if action == "expense":
            # 支出：借方应该是费用账户，贷方应该是资产账户
            if not self._is_expense_account(debit_account):
                warnings.append(
                    f"支出交易的借方账户 '{debit_account}' 通常应该是费用类账户"
                )
            if not self._is_asset_account(credit_account):
                warnings.append(
                    f"支出交易的贷方账户 '{credit_account}' 通常应该是资产类账户"
                )

        elif action == "income":
            # 收入：借方应该是资产账户，贷方应该是收入账户
            if not self._is_asset_account(debit_account):
                warnings.append(
                    f"收入交易的借方账户 '{debit_account}' 通常应该是资产类账户"
                )
            if not self._is_revenue_account(credit_account):
                warnings.append(
                    f"收入交易的贷方账户 '{credit_account}' 通常应该是收入类账户"
                )

        elif action == "transfer":
            # 转账：两个账户都应该是资产账户
            if not self._is_asset_account(debit_account):
                warnings.append(
                    f"转账的目标账户 '{debit_account}' 通常应该是资产类账户"
                )
            if not self._is_asset_account(credit_account):
                warnings.append(
                    f"转账的来源账户 '{credit_account}' 通常应该是资产类账户"
                )

        elif action == "loan_payment":
            # 还款：借方应该是负债账户，贷方应该是资产账户
            if not self._is_liability_account(debit_account):
                warnings.append(
                    f"还款交易的借方账户 '{debit_account}' 通常应该是负债类账户"
                )
            if not self._is_asset_account(credit_account):
                warnings.append(
                    f"还款交易的贷方账户 '{credit_account}' 通常应该是资产类账户"
                )

        return {"errors": errors, "warnings": warnings, "suggestions": suggestions}

    def _validate_confidence(self, confidence) -> dict:
        """验证置信度"""
        warnings = []
        cleaned_value = None

        try:
            conf_float = float(confidence)
            if conf_float < 0:
                cleaned_value = 0.0
                warnings.append("置信度不能为负数，已调整为0")
            elif conf_float > 1:
                cleaned_value = 1.0
                warnings.append("置信度不能大于1，已调整为1")
            else:
                cleaned_value = round(conf_float, 2)

                if cleaned_value < 0.5:
                    warnings.append("置信度较低，建议人工确认")

        except (ValueError, TypeError):
            cleaned_value = 1.0
            warnings.append("无效的置信度格式，已设为默认值1.0")

        return {"warnings": warnings, "cleaned_value": cleaned_value}

    def _validate_business_rules(self, transaction_data: dict) -> dict:
        """验证业务规则"""
        warnings = []
        suggestions = []

        # 金额异常检测
        amount = transaction_data.get("amount", 0)
        category = transaction_data.get("category", "")

        # 基于分类的金额合理性检查
        if category == "餐饮费用" and amount > 1000:
            warnings.append("餐饮费用金额较大，请确认是否正确")
        elif category == "交通费用" and amount > 2000:
            warnings.append("交通费用金额较大，请确认是否为长途出行")
        elif category == "娱乐费用" and amount > 5000:
            warnings.append("娱乐费用金额较大，请确认是否正确")

        # 时间相关检查
        current_hour = datetime.now().hour
        if category == "餐饮费用":
            if 2 <= current_hour <= 6:
                suggestions.append("深夜餐饮消费，注意健康饮食")

        # 支付方式合理性检查
        payment_method = transaction_data.get("payment_method", "")
        if payment_method == "现金" and amount > 5000:
            warnings.append("大额现金支付，请注意安全")

        return {"warnings": warnings, "suggestions": suggestions}

    def _is_expense_account(self, account_name: str) -> bool:
        """判断是否为费用账户"""
        return (
            account_name in self.accounts.get("expense_accounts", {})
            or "费用" in account_name
            or "支出" in account_name
        )

    def _is_asset_account(self, account_name: str) -> bool:
        """判断是否为资产账户"""
        return account_name in self.accounts.get("asset_accounts", {})

    def _is_revenue_account(self, account_name: str) -> bool:
        """判断是否为收入账户"""
        return (
            account_name in self.accounts.get("revenue_accounts", {})
            or "收入" in account_name
        )

    def _is_liability_account(self, account_name: str) -> bool:
        """判断是否为负债账户"""
        return (
            account_name in self.accounts.get("liability_accounts", {})
            or "欠款" in account_name
            or "贷款" in account_name
        )


# 简单测试代码
if __name__ == "__main__":
    validator = TransactionValidator()

    # 测试正常交易
    test_transaction = {
        "action": "expense",
        "amount": 25.0,
        "description": "买了咖啡25块用支付宝付的",
        "category": "餐饮费用",
        "debit_account": "餐饮费用",
        "credit_account": "支付宝",
        "confidence": 0.9,
    }

    result = validator.validate_transaction(test_transaction)
    print("=== 正常交易验证 ===")
    print(f"验证结果: {'通过' if result['is_valid'] else '失败'}")
    print(f"错误: {result['errors']}")
    print(f"警告: {result['warnings']}")
    print(f"建议: {result['suggestions']}")

    # 测试异常交易
    test_invalid = {
        "action": "invalid_action",
        "amount": -25,
        "description": "",
        "category": "未知分类",
        "debit_account": "餐饮费用",
        "credit_account": "餐饮费用",  # 同一账户
        "confidence": 1.5,
    }

    result2 = validator.validate_transaction(test_invalid)
    print("\n=== 异常交易验证 ===")
    print(f"验证结果: {'通过' if result2['is_valid'] else '失败'}")
    print(f"错误: {result2['errors']}")
    print(f"警告: {result2['warnings']}")
    print(f"建议: {result2['suggestions']}")
