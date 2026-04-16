<div align="center">
  <img src="./assets/tips.png" alt="MoeChat Banner" width="100%" />
</div>

<div align="center">
[![Version](https://img.shields.io/badge/version-1.0.0-2ea44f?style=for-the-badge)](./pyproject.toml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](./LICENSE)
</div>

<div align="center">
[![QQ群](https://custom-icon-badges.demolab.com/badge/QQ群-967981851-00BFFF?style=for-the-badge&logo=tencent-qq)](https://qm.qq.com/q/6pfdCFxJcc)
[![BiliBili](https://custom-icon-badges.demolab.com/badge/BiliBili-芙兰蠢兔-FF69B4?style=for-the-badge&logo=bilibili)](https://space.bilibili.com/3156308)
</div>


<div align="center">
  <img src="https://readme-typing-svg.demolab.com?font=ZCOOL+KuaiLe&size=40&pause=1200&color=fe7ea9&center=true&vCenter=true&repeat=true&width=900&lines=MoeChat+%7C+%E5%9F%BA%E4%BA%8E+GPT-SoVITS+%E7%9A%84%E8%AF%AD%E9%9F%B3%E4%BA%A4%E4%BA%92%E7%B3%BB%E7%BB%9F;%E7%94%A8%E8%AF%AD%E9%9F%B3%E5%92%8C+AI+%E8%A7%92%E8%89%B2%E8%87%AA%E7%84%B6%E5%AF%B9%E8%AF%9D%E3%80%81%E6%B2%89%E6%B5%B8%E6%89%AE%E6%BC%94;%E6%94%AF%E6%8C%81+ASR+%2B+LLM+%2B+TTS+%2B+WakeWord+%2B+Live2D" alt="typing" />
</div>

# 🌸MoeChat

一个面向角色陪伴与语音交互场景的 AI 系统，专为自然对话和沉浸式角色扮演而设计。该仓库为上游仓库的不同分支版，专注于桌面助手服务。

核心能力覆盖：

- 流式对话（SSE）
- 语音识别（ASR，REST + WebSocket）
- 语音合成（GPT-SoVITS，本地/接口双模式）
- 唤醒词检测（WakeWord，WebSocket）
- 助手资产管理（角色配置、语音模型、资源热更新）
- 情绪与动作生成（用于 Live2D 表情动作驱动）

</details>

## ✨ 项目亮点

### 🎯 项目简介

MoeChat 是一个围绕「语音陪伴 + 角色扮演 + 桌面助手」构建的后端核心系统，提供从语音输入到语音输出的完整闭环：

- 语音输入（ASR）
- 智能理解（LLM / ChatLLM / SLM）
- 语音输出（GPT-SoVITS）
- 角色驱动（情绪识别与动作生成）
- 本项目使用 GPT-SoVITS 作为 TTS 模块。
- 集成 ASR 接口，使用 funasr 作为语音识别模块基础。
- Moechat 支持所有 openai 规范的大语言模型接口。
- 先进的，拟人化的情绪算法，查看我们的情感系统构建[说明文档](./doc/emotion_readme.md)

### ✨ 核心特性

| 模块         | 能力              | 说明                                                         |
| ------------ | ----------------- | ------------------------------------------------------------ |
| 对话引擎     | SSE 流式输出      | 支持边生成边播报，降低首字延迟。Linux 环境下首 Token 延迟基本能做到 1.5s 以内。Windows 环境下延迟在 2.1s 左右。 |
| 语音识别     | REST + WebSocket  | 同时支持非流式和实时流式识别。                               |
| 语音合成     | 本地 / API 双模式 | 可按部署条件切换，兼顾质量与性能。根据情绪选择对应的参考音频。 |
| 唤醒词       | WakeWord          | 支持实时检测和关键词触发                                     |
| 助手系统     | 多助手管理        | 支持切换、增删改、资源同步                                   |
| 表情动作     | Emotion + Motion  | 支持驱动 Live2D 表情与动作                                   |
| 长期记忆查询 | LongMemory        | Moechat 项目拥有全站**最快**、**最精准**的长期记忆查询可根据如“昨天”、“上周”这样的模糊的时间范围精确查询记忆在 11800h CPU 的笔记本上测试，查询总耗时仅为 80ms 左右。 |

## 📂 项目结构

```text
MoeChat/
├── api/                # FastAPI 接口层（ASR、Chat、TTS、配置、助手等）
├── core/               # 对话核心流程与情绪/动作逻辑
├── services/           # 业务服务层（ASR、TTS、WakeWord、Assistant）
├── models/             # DTO 与类型定义
├── my_utils/           # 工具模块（配置、日志、请求封装等）
├── router/             # 路由注册
├── data/               # 模型、助手数据、运行时数据
├── assets/             # 文档图片等静态资源
├── config.example.yaml # 配置模板
├── config.yaml         # 实际运行配置（本地）
└── main_web.py         # 服务启动入口
```

## 🚀快速开始

### 推荐使用整合包

> 整合包包含完整环境、核心服务和客户端，对于普通用户使用，适合快速体验。

- 百度网盘：https://pan.baidu.com/s/5h_xqAGOZWkn4Y5dMSXk4Vg
- QQ 群：967981851（群内同步版本说明）



### 环境要求

- Windows / Linux
- Python 3.11+
- 推荐使用 uv 进行依赖管理
- 若使用本地 TTS / GPU 推理，请确保驱动与 CUDA 环境可用

### 配置项目

1. 复制配置模板

```bash
copy config.example.yaml config.yaml
```

2. 修改 `config.yaml` 中的 API Key、模型地址和运行参数。
3. 修改`Config.py`中的模型路径配置

### 启动服务

```bash
# 安装/同步依赖
uv sync

# 下载模型资源
uv run download.py

# 启动服务
uv run main_web.py
```

## 📋配置说明

默认配置文件：`config.yaml`

```yaml
SV:
  enable: false
  master_audio: test.wav
  thr:

LLM:
  api: https://dashscope.aliyuncs.com/compatible-mode/v1
  key: your-api-key
  model: qwen3.5-plus
  extra_config:
    enable_thinking: false

ChatLLM:
  api: https://dashscope.aliyuncs.com/compatible-mode/v1
  key: your-api-key
  model: qwen-flash-character
  extra_config:
    enable_thinking: false

SLM:
  api: http://localhost:11434/v1/chat/completions
  key: your-api-key
  model: qwen3:0.6b
  extra_config:
    temperature: 0.6
    stream: false

TTS:
  mode: local
  gptsovits_lite:
    use_bert: true
    use_flash_attn: false
  gptsovits:
    api: http://127.0.0.1:9880/tts

WakeWord:
  enable: false
  provider: cpu
  keywords_score: 1.0
  keywords_threshold: 0.25
```

关键字段建议：

- `TTS.mode`: `local` 走本地 GPT-SoVITS，`api` 走远端接口。
- `LLM/ChatLLM`: 推荐将聊天模型与工具模型分离，减少成本并提升响应速度。
- `SLM.extra_config.temperature`: 建议 `0.3~0.8`。
- `WakeWord.keywords_threshold`: 较小更敏感，建议先从 `0.25` 附近调试。



## 🎨客户端接入

官方客户端（Windows）：

- 项目地址：<https://github.com/Mios-dream/Meochat-APP>

客户端能力包含：

- Live2D 模型展示
- 桌面助手交互
- 助手配置管理
- 资源同步（assets 检查、下载、上传）

> [!WARNING]
> 当前官方客户端仅支持 Windows。

## 🤔常见问题

### Q1: 服务启动失败或模型加载失败

- 确认 `config.yaml` 路径和字段正确。
- 检查模型目录是否存在且完整，使用脚本下载模型，然后放入data/models目录下。
- 本地 TTS 模式下，确认 GPT-SoVITS 模型和参考音频路径有效。



## 📋开发计划

- [x] 基于 AI 情绪和动作驱动 Live2D
- [ ] 基于 AI 情绪和动作驱动 3D 模型
- [ ] 更完善的插件市场与能力扩展



## 🙏 致谢

- [SoulLink_Live2D](https://github.com/nanlingyin/SoulLink_Live2D)：为模型自动动作生成提供思路。
- [GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite)：为语音合成提供本地低成本方案

<div align="center">
  <h3>感谢一路以来支持 MoeChat 的朋友们</h3>
  <a href="https://github.com/Mios-dream/MioRobot/contributors" target="_blank">
    <img src="https://contrib.rocks/image?repo=Mios-dream/MoeChat" alt="contributors" />
  </a>
</div>
