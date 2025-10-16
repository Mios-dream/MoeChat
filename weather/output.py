# output.py

import json
from typing import List
from parsers import WeatherDataPoint


def save_as_json(
    data_points: List[WeatherDataPoint], filename: str = "weather_report.json"
):
    """
    将天气数据点列表保存为格式化的JSON文件。

    Args:
        data_points (List[WeatherDataPoint]): 要保存的数据列表。
        filename (str): 输出的文件名。
    """
    print(f"正在将 {len(data_points)} 条天气数据写入到 {filename}...")

    output_data = [dp.to_dict() for dp in data_points]

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"✓ 成功！报告已保存到 {filename}")
    except IOError as e:
        print(f"[X] 写入文件时发生错误: {e}")
