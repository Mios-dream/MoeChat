"""
核心记忆迁移脚本

将旧版 core_mem.yml 数据迁移到新版 SQLite 格式

用法:
    python -m my_utils.core_mem_migrate <角色名>
示例:
    python -m my_utils.core_mem_migrate Chat酱
"""

import sys
import sqlite3
import yaml
import os
import shutil
from pathlib import Path


def migrate(agent_id: str):
    """迁移指定角色的核心记忆数据"""
    data_dir = f"./data/agents/{agent_id}"
    yaml_path = f"{data_dir}/core_mem.yml"
    db_path = f"{data_dir}/core_mem.db"
    index_path = f"{data_dir}/core_mem.index"

    # 检查YAML文件是否存在
    if not os.path.exists(yaml_path):
        print(f"错误: {yaml_path} 不存在")
        return False

    # 检查SQLite是否已存在
    if os.path.exists(db_path):
        print(f"警告: {db_path} 已存在，将跳过迁移")
        response = input("是否继续? (y/n): ")
        if response.lower() != "y":
            return False

    # 读取YAML数据
    print(f"读取 {yaml_path} ...")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        print("YAML文件为空，无需迁移")
        return False

    print(f"找到 {len(data)} 条记忆")

    # 初始化SQLite
    os.makedirs(data_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            uuid TEXT PRIMARY KEY,
            time TEXT NOT NULL,
            text TEXT NOT NULL
        )
    """
    )

    # 迁移数据
    migrated = 0
    skipped = 0
    for uuid, item in data.items():
        cursor.execute("SELECT uuid FROM memories WHERE uuid = ?", (uuid,))
        if cursor.fetchone():
            skipped += 1
            continue
        cursor.execute(
            "INSERT INTO memories (uuid, time, text) VALUES (?, ?, ?)",
            (uuid, item["time"], item["text"]),
        )
        migrated += 1

    conn.commit()
    conn.close()

    print(f"迁移完成: {migrated} 条新增, {skipped} 条已存在")

    # 备份YAML文件
    backup_path = f"{yaml_path}.bak"
    shutil.move(yaml_path, backup_path)
    print(f"已备份到: {backup_path}")

    # 删除旧索引文件（如果存在）
    if os.path.exists(index_path):
        os.remove(index_path)
        print(f"已删除旧索引文件: {index_path}")
        print("索引将在下次启动时自动重建")

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python -m my_utils.core_mem_migrate <角色名>")
        print("示例: python -m my_utils.core_mem_migrate Chat酱")
        sys.exit(1)

    agent_id = sys.argv[1]
    success = migrate(agent_id)
    sys.exit(0 if success else 1)
