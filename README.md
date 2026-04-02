<p align="left"><img src="/doc/screen/logo1.png" alt="logo1" style="zoom:20%;" /></p>

![banner](/doc/screen/banner.png)

[![BaiduPan](https://custom-icon-badges.demolab.com/badge/BaiduPan-Link-4169E1?style=flat&logo=baidunetdisk)](https://pan.baidu.com/share/init?surl=mf6hHJt8hVW3G2Yp2gC3Sw&pwd=2333)
[![QQ Group](https://custom-icon-badges.demolab.com/badge/QQ_Group-967981851-00BFFF?style=flat&logo=tencent-qq)](https://qm.qq.com/q/6pfdCFxJcc)
[![BiliBili](https://custom-icon-badges.demolab.com/badge/BiliBili-MoeChat-FF69B4?style=flat&logo=bilibili)](https://space.bilibili.com/3156308)
[![Discord](https://custom-icon-badges.demolab.com/badge/Discord-Moechat-FF5024?style=flat&logo=Discord)](https://discord.gg/2JJ6J2T9P7)
[![Mega](https://custom-icon-badges.demolab.com/badge/Mega-Moechat-FF5024?style=flat&logo=mega&logoColor=red)](https://mega.nz/folder/LsZFEBAZ#mmz75Q--hKL6KG9jRNIj1g)

<a href="/README.md">English</a> |
<a href="/doc/README_zh.md">Chinese</a>

# Voice Interaction System Powered by GPT-SoVITS

## Overview

A powerful voice interaction system for natural AI conversations and immersive roleplay.

## Features

- Uses GPT-SoVITS as the TTS module.
- Integrates ASR interfaces with FunASR as the speech recognition backbone.
- Supports all OpenAI-compatible LLM APIs.
- First-token latency is typically under 1.5s on Linux and around 2.1s on Windows.
- Includes highly optimized long-term memory retrieval with accurate fuzzy time queries such as "yesterday" and "last week". On an Intel i7-11800H laptop, query time is around 80ms.
- Selects reference audio dynamically according to emotion.

## Test Platform

### Server

- OS: Manjaro
- CPU: AMD Ryzen 9 5950X
- GPU: NVIDIA RTX 3080 Ti

### Client

- Raspberry Pi 5

### Benchmark

![](/doc/screen/img.png)

## Changelog

### 2025.10.08

- Added context-aware meme sending.

  <p align="left"><img src="/doc/screen/sample2.png" alt="sample2" style="zoom: 33%;" /></p>

- Added a lightweight financial system based on double-entry bookkeeping.

  <p align="left"><img src="/doc/screen/sample_booking_en.png" alt="sample_booking_en" style="zoom: 50%;" /></p>

### 2025.06.29

- Introduced a brand-new emotion system.
- Added a lightweight web UI with keyword-triggered floating effects and visual interactions.

  <div style="text-align: left;"><img src="/doc/screen/sample1.png" alt="sample1" style="zoom: 55%;" /></div>

### 2025.06.11

- Added Character Template support for creating characters with built-in prompt templates.
- Added Journal System (long-term memory): supports accurate time-range retrieval such as "what did we chat about yesterday" and "where did we go last week".
- Added Core Memory: stores important user facts, memories, and preferences.

  These features require Character Template to be enabled.

- Decoupled from the original GPT-SoVITS codebase and switched to API-based integration.

### 2025.05.13

- Added speaker recognition.
- Added emotion-tag-based reference audio selection.
- Fixed several bugs.

## Integrated Package Usage

> The integrated package includes the full runtime environment, GPT-SoVITS, and client.

Download links:

- BaiduPan: [Download](https://pan.baidu.com/share/init?surl=mf6hHJt8hVW3G2Yp2gC3Sw&pwd=2333)
- Backup (123Pan): [Download](https://www.123865.com/s/kxlvjv-0Jayv)
- QQ Group: [967981851](https://qm.qq.com/q/6pfdCFxJcc)

### Start Core Services

```bash
# Start GPT-SoVITS server
cd GPT-SoVITS-v2pro-20250604-nvidia50
runtime\python.exe api_v2.py

# Start MoeChat server (in integrated package root)
uv sync
uv run main_web.py
```

## Client

Thanks to SanSan for providing client support for MoeChat.

> The current official client supports Windows only.

The client provides Live2D display, desktop companion features, and configuration management.

Client repository: [Meochat-APP](https://github.com/Mios-dream/Meochat-APP)

Screenshots:

![](/doc/screen/app_screen_1.png)

![](/doc/screen/app_screen_2.png)

![](/doc/screen/app_screen_3.png)

![](/doc/screen/app_screen_4.png)

![](/doc/screen/app_screen_5.png)

## Configuration

The integrated package uses `config.yaml`.

```yaml
SV: # verify_speaker
  enable: false
  master_audio: test.wav # WAV file containing your voice (3s-5s recommended)
  thr: # Threshold. Lower means more sensitive. Suggested range: 0.5-0.8

LLM: # General-purpose LLM for non-chat tasks
  api: https://dashscope.aliyuncs.com/compatible-mode/v1
  key: your-api-key-here
  model: qwen3.5-flash-2026-02-23
  extra_config:
    enable_thinking: false
    # Extra parameters, e.g. temperature: 0.7

ChatLLM: # Dedicated chat model
  api: https://dashscope.aliyuncs.com/compatible-mode/v1
  key: your-api-key-here
  model: qwen-flash-character
  extra_config:
    enable_thinking: false
    # Extra parameters, e.g. temperature: 0.7

SLM: # Small model for VAD rewrite and intent tasks
  api: http://localhost:11434/v1
  key:
  model: qwen3:0.6b
  extra_config:
    stream: false

TTS: # Text-to-Speech module
  mode: local # api / local。
  gptsovits_lite:
    use_bert: true # use Bert model, default true, may cause poor voice quality for other languages
    use_flash_attn: false # use Flash Attention, default false, need to install Flash Attention library
  gptsovits:
    api: http://127.0.0.1:9880/tts
```

## API

### ASR API

Endpoint: `/api/asr`

```python
# JSON request
# Audio format: WAV, 16kHz, int16, mono, 20ms frame size
# Encode audio as URL-safe Base64 and put it in `data`
{
  "data": str  # base64 audio data
}
```

### Chat API

Endpoint: `/api/chat`

Parameters:

- `msg`: chat context
- `generation_motion`: whether to generate Live2D motions (uses more tokens and latency)

```python
# SSE streaming endpoint
{
  "msg": [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hello! How can I help you?"},
    {"role": "user", "content": "What is 1+1?"}
  ],
  "generation_motion": true
}
```

Server response examples:

```python
{
  "type": "text",
  "sentence_id": 1,
  "message": "...",
  "timestamp_ms": 1774771748616,
  "done": false
}
{
  "type": "audio",
  "sentence_id": 1,
  "message": "...",
  "file": "base64_audio",
  "timestamp_ms": 1774771750827,
  "done": false
}
{
  "type": "motion_frame",
  "sentence_id": 1,
  "source_text": "...",
  "motions": [
    {
      "duration": 1200,
      "action": "...",
      "parameters": {
        "ParamEyeLOpen": 1.65
      }
    }
  ],
  "timestamp_ms": 1774771755186,
  "done": false
}
{
  "type": "done",
  "timestamp_ms": 1774771755186,
  "total_sentences": 1,
  "full_text": "...",
  "done": true
}
```

## Roadmap

- [x] English README
- [ ] Improve and optimize web response performance
- [ ] Add Live2D-widget support on web
- [ ] LLM self-awareness and digital life features
- [ ] Introduce arousal parameters based on traditional and Basson models
- [ ] Add 3D model support and full projection in client
- [x] Use AI emotion/action to control Live2D expression and motion
- [ ] Use AI emotion/action to control 3D expression and motion

## Special Thanks

- [SoulLink_Live2D](https://github.com/nanlingyin/SoulLink_Live2D): inspiration for automatic motion generation.

<div align="center">
  <h3>--------------------Thanks to everyone who has supported this project--------------------</h3>
</div>
<a href="https://github.com/Mios-dream/MioRobot/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=Mios-dream/MoeChat" />
</a>

## License

Program Name: MoeChat
Copyright (C) 2025 芙兰蠢兔、Tenzray

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
