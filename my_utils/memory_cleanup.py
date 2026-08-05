"""
系统内存清理工具

提供跨平台的内存回收能力（GC、PyTorch CUDA 缓存、进程工作集/堆回收），
以及基于 asyncio 的后台周期清理任务，供服务启动后定时执行。

平台差异：
- Windows：ctypes 调用 psapi.EmptyWorkingSet 收拢工作集（冷页面换出到磁盘，
  降低任务管理器"内存"列，缺页时自动换回，不影响运行）
- Linux：ctypes 调用 glibc malloc_trim(0) 将已释放的堆内存交还内核
  （仅 glibc 有效，Alpine/musl 无此符号，调用失败不影响运行）
- 其他平台：仅执行 gc.collect + torch.cuda.empty_cache
"""

import asyncio
import gc
import platform

from my_utils.log import logger

# 周期清理默认间隔（秒）：20 分钟
DEFAULT_CLEANUP_INTERVAL_SECONDS = 1200


def _collect_garbage() -> None:
    """回收 Python 临时对象并释放 PyTorch CUDA 缓存（跨平台通用）"""
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


def _trim_working_set_windows() -> None:
    """Windows：收拢进程工作集，将冷页面换出到磁盘"""
    try:
        import ctypes

        ctypes.windll.psapi.EmptyWorkingSet(ctypes.c_void_p(-1))
        logger.info("已收拢进程工作集，冷页面已换出到磁盘")
    except Exception as e:
        logger.warning(f"[内存清理] 收拢工作集失败: {e}")


def _trim_heap_linux() -> None:
    """Linux：将已释放的堆内存交还内核（glibc malloc_trim）"""
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        logger.info("[内存清理] 已将空闲堆内存交还内核 (malloc_trim)")
    except Exception as e:
        logger.warning(f"[内存清理] malloc_trim 失败: {e}")


def cleanup() -> None:
    """执行一次跨平台内存清理。

    通用步骤（gc + CUDA 缓存）恒执行；工作集/堆回收按平台分支执行。
    本方法线程安全，可在任意线程调用（耗时毫秒级，可忽略）。
    """
    _collect_garbage()
    system = platform.system()
    if system == "Windows":
        _trim_working_set_windows()
    elif system == "Linux":
        _trim_heap_linux()


async def periodic_cleanup(
    interval_seconds: int = DEFAULT_CLEANUP_INTERVAL_SECONDS,
) -> None:
    """后台周期清理协程：按固定间隔循环执行 cleanup()。

    参数：
    - interval_seconds: 清理间隔（秒），默认 20 分钟
    """
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            cleanup()
        except Exception as e:
            logger.error(f"[内存清理] 周期清理异常: {e}")


def start_periodic_cleanup(
    interval_seconds: int = DEFAULT_CLEANUP_INTERVAL_SECONDS,
) -> asyncio.Task:
    """启动后台周期清理任务（须在运行中的事件循环内调用）。

    返回创建的 asyncio.Task，供生命周期管理在退出时取消。

    参数：
    - interval_seconds: 清理间隔（秒），默认 20 分钟
    """
    return asyncio.get_running_loop().create_task(
        periodic_cleanup(interval_seconds)
    )
