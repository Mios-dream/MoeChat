# geolocation.py

import httpx
import config


class GeolocationError(Exception):
    """自定义异常，用于表示地理定位过程中发生的错误。"""

    pass


def get_public_ip() -> str:
    """
    从配置的服务中获取本机的公网IP地址。

    Returns:
        str: 公网IP地址字符串。

    Raises:
        GeolocationError: 如果请求失败或无法解析IP。
    """
    print("正在获取公网IP地址...")
    try:
        response = httpx.get(config.IP_SERVICE_URL, timeout=config.TIMEOUT)
        # 检查请求是否成功 (状态码 200-299)
        response.raise_for_status()

        data = response.json()
        ip = data.get("ip")
        if not ip:
            raise GeolocationError("无法从API响应中解析IP地址。")

        print(f"成功获取IP地址: {ip}")
        return ip
    except httpx.HTTPError as e:
        raise GeolocationError(f"获取IP地址时网络错误: {e}") from e


def get_location_from_ip(ip: str) -> str:
    """
    使用IP地址从配置的服务中获取地理位置（城市）。

    Args:
        ip (str): 要查询的公网IP地址。

    Returns:
        str: 对应的城市名称。

    Raises:
        GeolocationError: 如果请求失败或无法解析地理位置。
    """
    print(f"正在根据IP {ip} 查询地理位置...")
    url = config.GEO_SERVICE_URL.format(ip=ip)

    try:
        response = httpx.get(url, timeout=config.TIMEOUT)
        response.raise_for_status()

        data = response.json()
        # 检查API返回的状态
        if data.get("status") != "success":
            error_message = data.get("message", "未知API错误")
            raise GeolocationError(f"地理位置API返回错误: {error_message}")

        city = data.get("city")
        if not city:
            raise GeolocationError("无法从API响应中解析城市信息。")

        print(f"成功定位到城市: {city}")
        return city
    except httpx.HTTPError as e:
        raise GeolocationError(f"查询地理位置时网络错误: {e}") from e


def get_location() -> str:
    """
    执行完整的地理定位流程。
    这是此模块对外暴露的主要功能。

    Returns:
        str: 当前位置的城市名称。

    Raises:
        SystemExit: 如果在定位过程中发生无法恢复的错误。
    """
    try:
        public_ip = get_public_ip()
        city = get_location_from_ip(public_ip)
        return city
    except GeolocationError as e:
        # 出现错误,终止程序。
        # 定位失败程序直接退出。
        raise SystemExit(f"定位失败: {e}")


# --- 用于独立测试的入口 ---
if __name__ == "__main__":
    print("--- 正在独立测试 geolocation 模块 ---")
    location = get_location()
    print(f"--- 测试成功！最终获取到的位置是: {location} ---")
