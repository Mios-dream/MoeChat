# weather_fetcher.py (v2.0.0+ 终极潜行版)

import asyncio
from playwright.async_api import async_playwright, Browser

# --- 这是最关键的修正 ---
# 导入 Stealth 类，而不是 stealth_async 函数
from playwright_stealth import Stealth

# --------------------------
import config
from typing import Tuple


class WeatherFetchError(Exception):
    """自定义异常，表示所有天气数据源都获取失败。"""

    pass


async def _fetch_from_source(
    browser: Browser, source_name: str, url: str
) -> Tuple[str, str]:
    """使用Playwright和stealth插件获取HTML。"""
    print(f"-> [潜行模式] 正在打开 {source_name.capitalize()}...")

    # 我们不再需要手动应用stealth，新的方法会自动处理
    context = await browser.new_context()
    page = await context.new_page()

    try:
        await page.goto(
            url, timeout=config.TIMEOUT * 1000, wait_until="domcontentloaded"
        )

        if source_name == "google":
            try:
                accept_button = page.get_by_role("button", name="Alle akzeptieren")
                await accept_button.wait_for(timeout=5000)
                print("[交互] 检测到Cookie弹窗，正在点击'全部接受'...")
                await accept_button.click()
                await page.wait_for_timeout(1000)
            except Exception:
                print("[交互] 未检测到Cookie弹窗或已处理。")

        selector_map = {
            "google": "#wob_wc",
            "bing": 'div[class^="wtr_foreGround"]',
            "baidu": "#weather_list",
        }
        print(f"[交互] 正在等待天气控件 '{selector_map[source_name]}' 出现...")
        await page.wait_for_selector(selector_map[source_name], timeout=20000)

        html_content = await page.content()
        print(f"<- [潜行成功] 从 {source_name.capitalize()} 获取到渲染后的页面！")
        return html_content, source_name
    except Exception as e:
        screenshot_path = f"debug_screenshot_{source_name}.png"
        await page.screenshot(path=screenshot_path, full_page=True)
        print(
            f"\n[!!!] 在处理 {source_name.capitalize()} 时发生错误: {e.__class__.__name__}"
        )
        print(f"截图已保存到: {screenshot_path}")
        raise
    finally:
        await context.close()


async def fetch_weather_html(city: str) -> Tuple[str, str]:
    """使用Playwright并发获取。"""
    print(f"\n[v2.0.0+ 终极潜行模式] 开始并发获取 '{city}' 的天气信息...")

    launch_options = {
        "headless": True,
        "args": [
            "--window-size=1920,1080",
            "--disable-blink-features=AutomationControlled",
        ],
    }

    # --- 这是新版库的推荐用法 ---
    stealth_instance = Stealth()
    async with stealth_instance.use_async(async_playwright()) as p:
        # -----------------------------
        browser = await p.chromium.launch(**launch_options)
        try:
            tasks = []
            for source_name, url_template in config.SOURCES.items():
                url = url_template.format(city=city)
                task = asyncio.create_task(
                    _fetch_from_source(browser, source_name, url)
                )
                tasks.append(task)

            for future in asyncio.as_completed(tasks):
                try:
                    html_content, source_name = await future
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    return html_content, source_name
                except Exception:
                    pass
        finally:
            await browser.close()

    raise WeatherFetchError("所有潜行尝试均告失败。")
