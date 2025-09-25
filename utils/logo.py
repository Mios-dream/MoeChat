# LOGO 打印功能
from rich.console import Console
from rich.text import Text
import time


def print_moechat_logo(delay=0.2):
    console = Console()

    ascii_lines = [
        "███╗   ███╗ ██████╗ ███████╗ ██████╗██╗  ██╗ █████╗ ████████╗",
        "████╗ ████║██╔═══██╗██╔════╝██╔════╝██║  ██║██╔══██╗╚══██╔══╝",
        "██╔████╔██║██║   ██║█████╗  ██║     ███████║███████║   ██║   ",
        "██║╚██╔╝██║██║   ██║██╔══╝  ██║     ██╔══██║██╔══██║   ██║   ",
        "██║ ╚═╝ ██║╚██████╔╝███████╗╚██████╗██║  ██║██║  ██║   ██║   ",
        "╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ",
    ]

    gradient_colors = [
        "cyan",
        "bright_cyan",
        "bright_blue",
        "blue",
        "purple",
        "magenta",
    ]
    total_lines = len(ascii_lines)

    # 初始化行缓冲为空字符串
    rendered_lines = [""] * total_lines

    def render_and_flush():
        console.clear()
        for line in rendered_lines:
            console.print(line if line else " " * len(ascii_lines[0]))

    # Step 1: 打印奇数行（0,2,4...）
    for i in range(0, total_lines, 2):
        rendered_lines[i] = Text(
            ascii_lines[i], style=gradient_colors[i % len(gradient_colors)]
        )
        render_and_flush()
        time.sleep(delay)

    # Step 2: 打印偶数行（1,3,5...）
    for i in range(1, total_lines, 2):
        rendered_lines[i] = Text(
            ascii_lines[i], style=gradient_colors[i % len(gradient_colors)]
        )
        render_and_flush()
        time.sleep(delay)


if __name__ == "__main__":
    print_moechat_logo()
