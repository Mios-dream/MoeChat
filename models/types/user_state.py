import time
from pydantic import BaseModel, Field


class UserStateInfo(BaseModel):
    """
    助手用户私有状态模型
    存储每个用户与助手相关的私有动态数据（好感度、首次相遇时间等）
    与助手主体 info.yaml 分离存储于 user_state.yaml
    """

    firstMeetTime: int = Field(
        default_factory=lambda: int(time.time()), description="助手初次相遇时间"
    )
    love: int = Field(default=0, description="助手好感度")
    updatedAt: int = Field(
        default_factory=lambda: int(time.time()), description="用户状态更新时间"
    )
    assetsLastModified: int = Field(default=0, description="助手资产最后修改时间")

    @staticmethod
    def from_dict(data: dict) -> "UserStateInfo":
        return UserStateInfo(
            firstMeetTime=data.get("firstMeetTime", int(time.time())),
            love=data.get("love", 0),
            updatedAt=data.get("updatedAt", int(time.time())),
            assetsLastModified=data.get("assetsLastModified", 0),
        )
