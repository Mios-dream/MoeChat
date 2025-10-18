import flet as ft

# Chat messages
chat_list = ft.ListView(
    controls=[],
    expand=True,
    spacing=10,
    auto_scroll=True,
)


class ChatMessage:
    def __init__(self, user_name: str, text: str, positon: str):
        if positon == "left":
            self.position1 = ft.MainAxisAlignment.START
            self.position2 = ft.CrossAxisAlignment.START
        else:
            self.position1 = ft.MainAxisAlignment.END
            self.position2 = ft.CrossAxisAlignment.END
        self.user_name = user_name
        self.text = text
        self.tou = ft.CircleAvatar(
            content=ft.Text(self.get_initials(user_name), size=25),
            color=ft.Colors.WHITE,
            bgcolor=self.get_avatar_color(user_name),
            min_radius=30,
        )
        self.msg_list = ft.Column(
            controls=[
                ft.Container(
                    height=5,
                    width=1,
                )
            ],
            alignment=ft.MainAxisAlignment.START,
            horizontal_alignment=self.position2,
            auto_scroll=True,
        )
        self.cont = ft.Row(
            data=self.user_name,
            # controls=[self.msg_list, self.tou],
            alignment=self.position1,
            vertical_alignment=ft.CrossAxisAlignment.START,
            expand=True,
        )
        if positon == "left":
            self.cont.controls = [self.tou, self.msg_list]
            print("left")
        else:
            self.cont.controls = [self.msg_list, self.tou]
            print("right")

    def get_initials(self, user_name: str):
        if user_name:
            return user_name[:1].capitalize()
        else:
            return "Unknown"  # or any default value you prefer

    def get_avatar_color(self, user_name: str):
        colors_lookup = [
            ft.Colors.AMBER,
            ft.Colors.BLUE,
            ft.Colors.BROWN,
            ft.Colors.CYAN,
            ft.Colors.GREEN,
            ft.Colors.INDIGO,
            ft.Colors.LIME,
            ft.Colors.ORANGE,
            ft.Colors.PINK,
            ft.Colors.PURPLE,
            ft.Colors.RED,
            ft.Colors.TEAL,
            ft.Colors.YELLOW,
        ]
        return colors_lookup[hash(user_name) % len(colors_lookup)]


def get_msg_box(msg: str):
    res = ft.Container(
        content=ft.Text(msg, size=20, text_align=ft.TextAlign.CENTER),
        padding=5,
        border_radius=10,
        bgcolor=ft.colors.BLUE_900,
        width=300,
    )
    return res
