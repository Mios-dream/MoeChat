import os
import time


def get_latest_modification_time(directory):
    """
    获取目录下所有文件的最新修改时间

    Args:
        directory: 目录路径

    Returns:
        最新的修改时间戳
    """
    latest_mtime = 0

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            mtime = os.path.getmtime(file_path)
            if mtime > latest_mtime:
                latest_mtime = mtime

    # 如果目录为空，返回当前时间作为基准
    # if latest_mtime == 0:
    #     latest_mtime = time.time()

    return int(latest_mtime)
