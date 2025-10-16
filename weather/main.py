# main.py

import sys
import asyncio
from datetime import date, timedelta
from typing import List, Tuple
import geolocation
import weather_fetcher
import parsers
import output
from parsers import WeatherDataPoint


def _parse_user_intent() -> Tuple[int, int]:
    """
    解析用户的命令行输入意图。

    Returns:
        Tuple[int, int]: 一个元组 (要获取的天数, 日期起始偏移量)。
                         例如，'明天' -> (1, 1); '未来一周' -> (7, 0)
    """
    if len(sys.argv) != 2:
        print("用法: python main.py [今天天气 | 明天 | 后天 | 未来一周]")
        raise SystemExit(1)

    command = sys.argv[1]

    intent_map = {
        "今天天气": (1, 0),
        "天气": (1, 0),
        "明天": (1, 1),
        "后天": (1, 2),
        "未来一周": (7, 0),
    }

    if command not in intent_map:
        print(f"错误: 不支持的命令 '{command}'")
        print("支持的命令: 今天天气, 明天, 后天, 未来一周")
        raise SystemExit(1)

    return intent_map[command]


def _filter_data_by_intent(
    all_data: List[WeatherDataPoint], days_to_fetch: int, start_offset: int
) -> List[WeatherDataPoint]:
    """根据用户的意图筛选出最终需要的数据。"""

    start_date = date.today() + timedelta(days=start_offset)
    end_date = start_date + timedelta(days=days_to_fetch - 1)

    # 集合
    target_dates = {start_date + timedelta(days=i) for i in range(days_to_fetch)}

    final_data = [dp for dp in all_data if dp.date in target_dates]

    return final_data


async def main():
    """程序的主入口异步函数。"""

    # 解析用户意图
    try:
        days_to_fetch, start_offset = _parse_user_intent()
    except SystemExit:
        return

    print("--- 天气查询程序已启动 ---")

    try:
        # 获取地理位置
        city = geolocation.get_location()

        # 并发获取天气HTML
        html_content, source = await weather_fetcher.fetch_weather_html(city)

        # 解析HTML为结构化数据
        all_weather_data = parsers.parse(html_content, source)

        if not all_weather_data:
            # --- 在这里添加以下三行 ---
            with open("failed_to_parse.html", "w", encoding="utf-8") as f:
                f.write(html_content)
            print(
                "\n[诊断信息] 解析失败，已将收到的HTML保存到 failed_to_parse.html 文件中，以便分析。"
            )
            # --------------------------
            raise SystemExit("解析失败，未能从HTML中提取到任何天气数据。")

        # 根据用户意图筛选数据
        final_data = _filter_data_by_intent(
            all_weather_data, days_to_fetch, start_offset
        )

        if not final_data:
            print("未能找到您所请求日期的天气信息。")
            return

        # 保存结果到JSON文件
        output.save_as_json(final_data)

    except SystemExit as e:
        print(f"\n程序中止: {e}")
    except Exception as e:
        # 捕获其他所有意外错误
        print(f"\n[X] 发生未知错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())
