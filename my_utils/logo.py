# LOGO 打印功能
import time
import sys


def print_moechat_logo(delay=0.2):
    ascii_lines = [
        "███╗   ███╗ ██████╗ ███████╗ ██████╗██╗  ██╗ █████╗ ████████╗",
        "████╗ ████║██╔═══██╗██╔════╝██╔════╝██║  ██║██╔══██╗╚══██╔══╝",
        "██╔████╔██║██║   ██║█████╗  ██║     ███████║███████║   ██║   ",
        "██║╚██╔╝██║██║   ██║██╔══╝  ██║     ██╔══██║██╔══██║   ██║   ",
        "██║ ╚═╝ ██║╚██████╔╝███████╗╚██████╗██║  ██║██║  ██║   ██║   ",
        "╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ",
    ]

    gradient_rgbs = [
        (254, 126, 169),
        (254, 152, 185),
        (254, 178, 201),
        (254, 204, 217),
        (255, 230, 233),
        (255, 240, 245),
    ]
    total_lines = len(ascii_lines)

    def colorize(text, r, g, b):
        """使用ANSI 24位真彩色转义码着色"""
        return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

    def clear_and_render(rendered_lines):
        """光标归位 + 清屏 + 重绘"""
        sys.stdout.write("\033[H\033[J")
        for line in rendered_lines:
            sys.stdout.write(line + "\n")
        sys.stdout.flush()

    # 初始化行缓冲为空字符串
    rendered_lines = [""] * total_lines

    # Step 1: 打印奇数行（0,2,4...）
    for i in range(0, total_lines, 2):
        r, g, b = gradient_rgbs[i]
        rendered_lines[i] = colorize(ascii_lines[i], r, g, b)
        clear_and_render(rendered_lines)
        time.sleep(delay)

    # Step 2: 打印偶数行（1,3,5...）
    for i in range(1, total_lines, 2):
        r, g, b = gradient_rgbs[i]
        rendered_lines[i] = colorize(ascii_lines[i], r, g, b)
        clear_and_render(rendered_lines)
        time.sleep(delay)


if __name__ == "__main__":
    print_moechat_logo()
