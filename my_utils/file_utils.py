import os


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


def get_subdirectory_mtimes(directory: str) -> dict[str, int]:
    """
    获取目录下各子目录的最新修改时间

    用于增量资源更新检查，前端可根据子目录名称筛选需要更新的资源类型。

    Args:
        directory: 资源根目录路径（如 assets/）

    Returns:
        dict[str, int]: 子目录名称 -> 最新修改时间戳
            例如: {"images": 1234567890, "models": 1234567891, "other": 1234567892}
            其中 "other" 表示根目录下直接存放的文件（不属于任何子目录）
    """
    result = {}
    root_files_mtime = 0

    if not os.path.exists(directory) or not os.path.isdir(directory):
        return result

    for entry in os.listdir(directory):
        entry_path = os.path.join(directory, entry)

        if os.path.isdir(entry_path):
            # 子目录：获取该目录下所有文件的最新修改时间
            dir_mtime = 0
            for root, _, files in os.walk(entry_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    mtime = os.path.getmtime(file_path)
                    if mtime > dir_mtime:
                        dir_mtime = mtime
            if dir_mtime > 0:
                result[entry] = int(dir_mtime)
        else:
            # 根目录下的文件：记录最新修改时间
            mtime = os.path.getmtime(entry_path)
            if mtime > root_files_mtime:
                root_files_mtime = mtime

    # 如果根目录下有直接存放的文件，归入 "other" 类别
    if root_files_mtime > 0:
        result["other"] = int(root_files_mtime)

    return result
