# 交互消息生成接口请求模型
from pydantic import BaseModel
from typing import Optional


class MouseEventStatus(BaseModel):
    idleDurationMs: int = 0
    isIdle: bool = False
    timestamp: int = 0


class AppEventStatus(BaseModel):
    appName: str = ""
    previousAppName: str | None = None
    title: str = ""
    category: str = ""
    continuousMs: int = 0
    timestamp: int = 0


class SystemPowerStatus(BaseModel):
    state: str = ""  # 'charging' | 'battery'
    timestamp: int = 0


class BatteryStatus(BaseModel):
    percent: int = 0
    isCharging: bool = False
    isLow: bool = False
    threshold: int = 0
    timestamp: int = 0


class TaskEventStatus(BaseModel):
    taskName: str = ""
    success: bool = False
    timestamp: int = 0


class SystemEventStatus(BaseModel):
    eventName: str = ""
    description: str = ""
    timestamp: int = 0


class InteractionContext(BaseModel):
    lastInteraction: int | None = None
    isBusy: bool | None = None
    isInConversation: bool | None = None
    lastEventTime: int | None = None
    lastEventType: str | None = None
    lastMessage: str | None = None
    mouseEventStatus: MouseEventStatus | None = None
    appEventStatus: AppEventStatus | None = None
    systemPowerStatus: SystemPowerStatus | None = None
    batteryStatus: BatteryStatus | None = None
    taskEventStatus: TaskEventStatus | None = None
    systemEventStatus: SystemEventStatus | None = None


class InteractionMessageRequest(BaseModel):
    event_type: str
    scene: str
    context: InteractionContext = InteractionContext()
    generation_motion: bool = False
    include_history: bool = False
    history_limit: int = 10
