import time
from pydantic import BaseModel, Field


class UserStateInfo(BaseModel):
    """
    助手用户私有状态模型
    存储每个用户与助手相关的私有动态数据（好感度、信任度、羁绊、首次相遇时间等）
    与助手主体 info.yaml 分离存储于 user_state.yaml
    """

    firstMeetTime: int = Field(
        default_factory=lambda: int(time.time()), description="助手初次相遇时间"
    )
    love: int = Field(default=0, description="好感度 -50~100，决定关系阶段，双向波动")
    trust: int = Field(default=50, description="信任度 -50~100，下降快恢复慢")
    bond: int = Field(default=0, description="羁绊值 0~∞，只增不减，体现陪伴深度")
    updatedAt: int = Field(
        default_factory=lambda: int(time.time()), description="用户状态更新时间"
    )
    assetsLastModified: int = Field(default=0, description="助手资产最后修改时间")

    @staticmethod
    def from_dict(data: dict) -> "UserStateInfo":
        return UserStateInfo(
            firstMeetTime=data.get("firstMeetTime", int(time.time())),
            love=data.get("love", 0),
            trust=data.get("trust", 50),
            bond=data.get("bond", 0),
            updatedAt=data.get("updatedAt", int(time.time())),
            assetsLastModified=data.get("assetsLastModified", 0),
        )
