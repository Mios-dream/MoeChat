# parsers.py (终极诊断版 v2)

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List
from bs4 import BeautifulSoup
import re


@dataclass
class WeatherDataPoint:
    """我们的标准天气数据单元。"""

    date: date
    description: str
    temp_high: int
    temp_low: int
    source: str

    def to_dict(self):
        return {
            "date": self.date.isoformat(),
            "description": self.description,
            "temp_high_celsius": self.temp_high,
            "temp_low_celsius": self.temp_low,
            "source": self.source,
        }


def _clean_temp(temp_str: str) -> int:
    """使用正则表达式清理温度字符串。"""
    match = re.search(r"(-?\d+)", temp_str)
    if match:
        return int(match.group(1))
    raise ValueError(f"无法从 '{temp_str}' 中提取温度")


import hashlib  # 导入哈希库


def _get_base64_fingerprint(base64_str: str, length: int = 16) -> str:
    """从Base64图片数据中创建一个简短、唯一的“指纹”"""
    # 我们不使用完整的Base64字符串，因为它太长了
    # 我们只取数据部分，并用md5创建一个简短的哈希值作为指纹
    try:
        data = base64_str.split(",")[1]
        return hashlib.md5(data.encode()).hexdigest()[:length]
    except IndexError:
        return "invalid_base64"


def _parse_google(html_content: str, source: str) -> List[WeatherDataPoint]:
    """【最终胜利版】精确适配最终的HTML结构。"""
    soup = BeautifulSoup(html_content, "html.parser")
    data_points = []

    container = soup.select_one("#wob_wc")
    if not container:
        return []

    for day_element in container.select(".wob_df"):
        try:
            est_date = date.today() + timedelta(days=len(data_points))

            desc_tag = day_element.select_one("img.YQ4gaf")
            description = (
                desc_tag["alt"] if desc_tag and "alt" in desc_tag.attrs else "N/A"
            )

            temp_high_tag = day_element.select_one(".gNCp2e span.wob_t")
            temp_low_tag = day_element.select_one(".QrNVmd span.wob_t")

            if not temp_high_tag or not temp_low_tag:
                continue

            temp_high = _clean_temp(temp_high_tag.text)
            temp_low = _clean_temp(temp_low_tag.text)

            data_points.append(
                WeatherDataPoint(
                    date=est_date,
                    description=description,
                    temp_high=temp_high,
                    temp_low=temp_low,
                    source=source,
                )
            )
        except Exception:
            continue

    return data_points


def _parse_bing(html_content: str, source: str) -> List[WeatherDataPoint]:
    # (此部分逻辑保持不变)
    soup = BeautifulSoup(html_content, "html.parser")  # 也换成html.parser
    data_points = []
    container = soup.select_one('div[class^="wtr_foreGround"]')
    if not container:
        return []
    for day_element in container.select(".wtr_forecastDay"):
        try:
            est_date = date.today() + timedelta(days=len(data_points))
            desc_tag = day_element.select_one(".wtr_condi")
            description = (
                desc_tag["aria-label"]
                if desc_tag and "aria-label" in desc_tag.attrs
                else "N/A"
            )
            temp_high = _clean_temp(day_element.select_one(".wtr_hi").text)
            temp_low = _clean_temp(day_element.select_one(".wtr_lo").text)
            data_points.append(
                WeatherDataPoint(
                    date=est_date,
                    description=description,
                    temp_high=temp_high,
                    temp_low=temp_low,
                    source=source,
                )
            )
        except Exception:
            continue
    return data_points


def _parse_baidu(html_content: str, source: str) -> List[WeatherDataPoint]:
    # (此部分逻辑保持不变)
    soup = BeautifulSoup(html_content, "html.parser")  # 也换成html.parser
    data_points = []
    container = soup.select_one("#weather_list")
    if not container:
        return []
    for day_element in container.select("a.weather-week-item"):
        try:
            est_date = date.today() + timedelta(days=len(data_points))
            desc_tag = day_element.select_one(".weather-week-des")
            description = desc_tag.text.strip() if desc_tag else "N/A"
            temp_tags = day_element.select(".weather-week-temp span")
            if len(temp_tags) < 2:
                continue
            temp_high = _clean_temp(temp_tags[0].text)
            temp_low = _clean_temp(temp_tags[1].text)
            data_points.append(
                WeatherDataPoint(
                    date=est_date,
                    description=description,
                    temp_high=temp_high,
                    temp_low=temp_low,
                    source=source,
                )
            )
        except Exception:
            continue
    return data_points


# 调度中心 (保持不变)
_parser_map = {
    "google": _parse_google,
    "bing": _parse_bing,
    "baidu": _parse_baidu,
}


def parse(html_content: str, source: str) -> List[WeatherDataPoint]:
    """根据来源调用相应的HTML解析器。"""
    if source not in _parser_map:
        raise ValueError(f"未知的来源: '{source}'. 没有对应的解析器。")

    parser_func = _parser_map[source]
    print(f"使用 {source.capitalize()} 解析器...")
    return parser_func(html_content, source)
