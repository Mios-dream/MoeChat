from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader

templates = APIRouter()

env = Environment(loader=FileSystemLoader("web/resources/static"))


@templates.get("/", response_class=HTMLResponse)
async def home_page():
    template = env.get_template("moechat_iphone_client.html")
    return template.render()


@templates.get("/", response_class=HTMLResponse)
async def index_page():
    template = env.get_template("moechat_iphone_client.html")
    return template.render()
