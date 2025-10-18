import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager


class DatabaseManager:
    def __init__(self, db_path: str = "data/transactions.db"):
        self.db_path = db_path
        self.config_path = "config"
        self.accounts_config = self._load_config("accounts.json")
        self._ensure_database_exists()
        self._init_tables()

    def _load_config(self, filename: str) -> dict:
        """加载配置文件"""
        try:
            with open(
                os.path.join(self.config_path, filename), "r", encoding="utf-8"
            ) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"配置文件 {filename} 未找到")
            return {}
        except json.JSONDecodeError:
            print(f"配置文件 {filename} 格式错误")
            return {}

    def _ensure_database_exists(self):
        """确保数据库文件和目录存在"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    @contextmanager
    def _get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 使查询结果可以像字典一样访问
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_tables(self):
        """初始化数据库表"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 交易主表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    description TEXT NOT NULL,
                    total_amount REAL NOT NULL,
                    action TEXT NOT NULL,
                    category TEXT,
                    payment_method TEXT,
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # 分录明细表（实现复式记账）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id INTEGER NOT NULL,
                    account_name TEXT NOT NULL,
                    account_type TEXT NOT NULL,
                    debit_amount REAL DEFAULT 0,
                    credit_amount REAL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (transaction_id) REFERENCES transactions (id) ON DELETE CASCADE
                )
            """)

            # 账户余额表（用于快速查询）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS account_balances (
                    account_name TEXT PRIMARY KEY,
                    account_type TEXT NOT NULL,
                    balance REAL NOT NULL DEFAULT 0,
                    last_updated TEXT NOT NULL
                )
            """)

            # 创建索引以提高查询性能
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_entries_account ON entries(account_name)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_entries_transaction ON entries(transaction_id)"
            )

            conn.commit()

    def save_transaction(self, transaction_data: dict) -> int:
        """
        保存交易记录

        Args:
            transaction_data: 交易数据字典
                {
                    'action': str,
                    'amount': float,
                    'description': str,
                    'category': str,
                    'debit_account': str,
                    'credit_account': str,
                    'confidence': float
                }

        Returns:
            int: 交易ID
        """
        current_time = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            try:
                # 插入交易主表
                cursor.execute(
                    """
                    INSERT INTO transactions 
                    (date, description, total_amount, action, category, payment_method, confidence, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        current_time.split("T")[0],  # 日期部分
                        transaction_data["description"],
                        transaction_data["amount"],
                        transaction_data["action"],
                        transaction_data.get("category"),
                        transaction_data.get("payment_method"),
                        transaction_data.get("confidence", 1.0),
                        current_time,
                        current_time,
                    ),
                )

                transaction_id = cursor.lastrowid

                # 插入借方分录
                debit_account = transaction_data["debit_account"]
                debit_type = self._get_account_type(debit_account)
                cursor.execute(
                    """
                    INSERT INTO entries 
                    (transaction_id, account_name, account_type, debit_amount, credit_amount, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        transaction_id,
                        debit_account,
                        debit_type,
                        transaction_data["amount"],
                        0,
                        current_time,
                    ),
                )

                # 插入贷方分录
                credit_account = transaction_data["credit_account"]
                credit_type = self._get_account_type(credit_account)
                cursor.execute(
                    """
                    INSERT INTO entries 
                    (transaction_id, account_name, account_type, credit_amount, debit_amount, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        transaction_id,
                        credit_account,
                        credit_type,
                        transaction_data["amount"],
                        0,
                        current_time,
                    ),
                )

                # 更新账户余额
                self._update_account_balance(
                    cursor, debit_account, debit_type, transaction_data["amount"], True
                )
                self._update_account_balance(
                    cursor,
                    credit_account,
                    credit_type,
                    transaction_data["amount"],
                    False,
                )

                conn.commit()
                return transaction_id

            except Exception as e:
                conn.rollback()
                raise Exception(f"保存交易失败: {str(e)}")

    def _get_account_type(self, account_name: str) -> str:
        """根据账户名称推断账户类型"""
        # 从配置文件中查找账户类型
        all_accounts = self.accounts_config or {}

        # 查找负债账户
        liability_accounts = all_accounts.get("liability_accounts", {})
        if account_name in liability_accounts:
            return "liability"

        # 查找资产账户
        asset_accounts = all_accounts.get("asset_accounts", {})
        if account_name in asset_accounts:
            return "asset"

        # 查找收入账户
        revenue_accounts = all_accounts.get("revenue_accounts", {})
        if account_name in revenue_accounts:
            return "revenue"

        # 查找费用账户
        expense_accounts = all_accounts.get("expense_accounts", {})
        if account_name in expense_accounts:
            return "expense"

        # 默认推断（保持原逻辑作为兜底）
        if "费用" in account_name or "支出" in account_name:
            return "expense"
        elif "收入" in account_name:
            return "revenue"
        elif "欠款" in account_name or "贷款" in account_name or "借款" in account_name:
            return "liability"
        elif "资本" in account_name or "权益" in account_name:
            return "equity"
        else:
            return "asset"  # 默认为资产类

    def _update_account_balance(
        self,
        cursor,
        account_name: str,
        account_type: str,
        amount: float,
        is_debit: bool,
    ):
        """更新账户余额"""
        current_time = datetime.now().isoformat()

        # 获取当前余额
        cursor.execute(
            "SELECT balance FROM account_balances WHERE account_name = ?",
            (account_name,),
        )
        result = cursor.fetchone()

        if result:
            current_balance = result[0]
        else:
            current_balance = 0
            # 插入新账户
            cursor.execute(
                """
                INSERT INTO account_balances (account_name, account_type, balance, last_updated)
                VALUES (?, ?, ?, ?)
            """,
                (account_name, account_type, 0, current_time),
            )

        # 计算新余额
        # 资产和费用类账户：借方增加，贷方减少
        # 负债、权益和收入类账户：借方减少，贷方增加
        if account_type in ["asset", "expense"]:
            new_balance = (
                current_balance + amount if is_debit else current_balance - amount
            )
        else:  # liability, equity, revenue
            new_balance = (
                current_balance - amount if is_debit else current_balance + amount
            )

        # 更新余额
        cursor.execute(
            """
            UPDATE account_balances 
            SET balance = ?, last_updated = ?
            WHERE account_name = ?
        """,
            (new_balance, current_time, account_name),
        )

    def get_account_balance(self, account_name: str) -> Optional[float]:
        """获取账户余额"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT balance FROM account_balances WHERE account_name = ?",
                (account_name,),
            )
            result = cursor.fetchone()
            return result[0] if result else None

    def get_all_balances(self) -> List[Dict]:
        """获取所有账户余额"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT account_name, account_type, balance, last_updated 
                FROM account_balances 
                ORDER BY account_type, account_name
            """)
            return [dict(row) for row in cursor.fetchall()]

    def search_transactions(
        self,
        start_date: str = None,
        end_date: str = None,
        account_name: str = None,
        category: str = None,
        keyword: str = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        搜索交易记录

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            account_name: 账户名称
            category: 分类
            keyword: 关键词（在描述中搜索）
            limit: 最大返回数量

        Returns:
            List[Dict]: 交易记录列表
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 构建查询条件
            conditions = []
            params = []

            if start_date:
                conditions.append("t.date >= ?")
                params.append(start_date)

            if end_date:
                conditions.append("t.date <= ?")
                params.append(end_date)

            if category:
                conditions.append("t.category = ?")
                params.append(category)

            if keyword:
                conditions.append("t.description LIKE ?")
                params.append(f"%{keyword}%")

            # 如果指定了账户名，需要连接entries表
            if account_name:
                query = """
                    SELECT DISTINCT t.*, e.account_name, e.debit_amount, e.credit_amount
                    FROM transactions t
                    JOIN entries e ON t.id = e.transaction_id
                    WHERE e.account_name = ?
                """
                params.insert(0, account_name)

                if conditions:
                    query += " AND " + " AND ".join(conditions)
            else:
                query = "SELECT * FROM transactions t"
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

            query += f" ORDER BY t.date DESC, t.created_at DESC LIMIT {limit}"

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_monthly_summary(self, year: int, month: int) -> Dict:
        """获取月度汇总"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            date_filter = f"{year:04d}-{month:02d}"

            # 总收入
            cursor.execute(
                """
                SELECT SUM(total_amount) FROM transactions 
                WHERE action = 'income' AND date LIKE ?
            """,
                (f"{date_filter}%",),
            )
            total_income = cursor.fetchone()[0] or 0

            # 总支出
            cursor.execute(
                """
                SELECT SUM(total_amount) FROM transactions 
                WHERE action = 'expense' AND date LIKE ?
            """,
                (f"{date_filter}%",),
            )
            total_expense = cursor.fetchone()[0] or 0

            # 按分类统计支出
            cursor.execute(
                """
                SELECT category, SUM(total_amount) FROM transactions 
                WHERE action = 'expense' AND date LIKE ?
                GROUP BY category
                ORDER BY SUM(total_amount) DESC
            """,
                (f"{date_filter}%",),
            )
            expense_by_category = dict(cursor.fetchall())

            return {
                "year": year,
                "month": month,
                "total_income": total_income,
                "total_expense": total_expense,
                "net_income": total_income - total_expense,
                "expense_by_category": expense_by_category,
            }

    def delete_transaction(self, transaction_id: int) -> bool:
        """删除交易记录"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            try:
                # 获取交易信息以便回滚账户余额
                cursor.execute(
                    """
                    SELECT e.account_name, e.account_type, e.debit_amount, e.credit_amount
                    FROM entries e WHERE e.transaction_id = ?
                """,
                    (transaction_id,),
                )
                entries = cursor.fetchall()

                # 回滚账户余额
                for entry in entries:
                    account_name = entry[0]
                    account_type = entry[1]
                    debit_amount = entry[2]
                    credit_amount = entry[3]

                    if debit_amount > 0:
                        self._update_account_balance(
                            cursor, account_name, account_type, debit_amount, False
                        )
                    if credit_amount > 0:
                        self._update_account_balance(
                            cursor, account_name, account_type, credit_amount, True
                        )

                # 删除交易记录（级联删除entries）
                cursor.execute(
                    "DELETE FROM transactions WHERE id = ?", (transaction_id,)
                )

                conn.commit()
                return cursor.rowcount > 0

            except Exception as e:
                conn.rollback()
                raise Exception(f"删除交易失败: {str(e)}")

    def validate_accounts_balance(self) -> Dict:
        """验证账户余额一致性"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 重新计算所有账户余额
            cursor.execute("""
                SELECT account_name, account_type,
                       SUM(debit_amount) as total_debit,
                       SUM(credit_amount) as total_credit
                FROM entries 
                GROUP BY account_name, account_type
            """)

            calculated_balances = {}
            for row in cursor.fetchall():
                account_name = row[0]
                account_type = row[1]
                total_debit = row[2] or 0
                total_credit = row[3] or 0

                # 计算正确的余额
                if account_type in ["asset", "expense"]:
                    balance = total_debit - total_credit
                else:  # liability, equity, revenue
                    balance = total_credit - total_debit

                calculated_balances[account_name] = balance

            # 获取当前存储的余额
            cursor.execute("SELECT account_name, balance FROM account_balances")
            stored_balances = dict(cursor.fetchall())

            # 比较差异
            discrepancies = {}
            for account_name, calculated in calculated_balances.items():
                stored = stored_balances.get(account_name, 0)
                if abs(calculated - stored) > 0.01:  # 允许0.01的浮点数误差
                    discrepancies[account_name] = {
                        "calculated": calculated,
                        "stored": stored,
                        "difference": calculated - stored,
                    }

            return {
                "total_accounts": len(calculated_balances),
                "discrepancies_count": len(discrepancies),
                "discrepancies": discrepancies,
            }


# 简单测试代码
if __name__ == "__main__":
    db = DatabaseManager()

    # 测试保存交易
    test_transaction = {
        "action": "expense",
        "amount": 25.0,
        "description": "买了咖啡25块用支付宝付的",
        "category": "餐饮费用",
        "debit_account": "餐饮费用",
        "credit_account": "支付宝",
        "confidence": 0.9,
    }

    transaction_id = db.save_transaction(test_transaction)
    print(f"保存交易成功，ID: {transaction_id}")

    # 测试查询余额
    balance = db.get_account_balance("支付宝")
    print(f"支付宝余额: {balance}")

    # 测试搜索交易
    transactions = db.search_transactions(keyword="咖啡")
    print(f"搜索结果: {len(transactions)} 条记录")
