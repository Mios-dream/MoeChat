# 角色模板

import os
from utils import long_mem, data_base, prompt, core_mem, log as Log
from utils import config as CConfig
import time
from threading import Thread, Lock
import requests
import jionlp
import ast

# from ruamel.yaml import YAML
# from ruamel.yaml.scalarstring import PreservedScalarString
import re
import json
import yaml
from ruamel.yaml import YAML


class Agent:
    def update_config(self, agent_id: str):
        # 读取配置文件
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)
        with open(f"./data/agents/{agent_id}/agent_config.yaml", "r", encoding="utf-8") as f:
            self.agent_config = yaml.load(f)
        self.agent_id = agent_id

        # 载入配置
        '''
        agent独立配置文件
        '''
        self.char = self.agent_config["Agent"]["char"]
        self.user = self.agent_config["Agent"]["user"]
        self.char_settings = self.agent_config["Agent"]["char_settings"]
        self.char_personalities = self.agent_config["Agent"]["char_personalities"]
        self.message_example = self.agent_config["Agent"]["message_example"]
        self.mask = self.agent_config["Agent"]["mask"]

        self.is_data_base = CConfig.config["Agent"]["lore_books"]
        self.data_base_thresholds = CConfig.config["Agent"]["books_thresholds"]
        self.data_base_depth = CConfig.config["Agent"]["scan_depth"]

        '''
        全局设置
        '''
        self.is_long_mem = CConfig.config["Agent"]["long_memory"]
        self.is_check_memorys = CConfig.config["Agent"]["is_check_memorys"]
        self.mem_thresholds = CConfig.config["Agent"]["mem_thresholds"]

        self.is_core_mem = CConfig.config["Agent"]["is_core_mem"]

        self.llm_config = CConfig.config["LLM2"]

        # 载入提示词
        self.prompt = []
        self.prompt = """"""
        # self.prompt.append({"role": "system", "content": f"当前系统时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}"})
        # tt = '''6. 注意输出文字的时候，将口语内容使用""符号包裹起来，并且优先输出口语内容，其他文字使用()符号包裹。'''
        self.long_mem_prompt = prompt.long_mem_prompt
        self.data_base_prompt = prompt.data_base_prompt
        self.core_mem_prompt = prompt.core_mem_prompt
        if self.char_settings:
            self.system_prompt = prompt.system_prompt.replace(
                "{{char}}", self.char
            ).replace("{{user}}", self.user)
            self.char_setting_prompt = (
                prompt.char_setting_prompt.replace(
                    "{{char_setting_prompt}}", self.char_settings
                )
                .replace("{{char}}", self.char)
                .replace("{{user}}", self.user)
            )
            # self.prompt.append({"role": "system", "content": self.system_prompt})
            # self.prompt.append({"role": "system", "content": self.char_setting_prompt})
            self.prompt += self.system_prompt + "\n\n"
            self.prompt += self.char_setting_prompt + "\n\n"
        if self.char_personalities:
            self.char_Personalities_prompt = (
                prompt.char_Personalities_prompt.replace(
                    "{{char_Personalities_prompt}}", self.char_personalities
                )
                .replace("{{char}}", self.char)
                .replace("{{user}}", self.user)
            )
            # self.prompt.append({"role": "system", "content": self.char_Personalities_prompt})
            self.prompt += self.char_Personalities_prompt + "\n\n"
        if self.mask:
            self.mask_prompt = (
                prompt.mask_prompt.replace("{{mask_prompt}}", self.mask)
                .replace("{{char}}", self.char)
                .replace("{{user}}", self.user)
            )
            # self.prompt.append({"role": "system", "content": self.mask_prompt})
            self.prompt += self.mask_prompt + "\n\n"
        if self.message_example:
            self.message_example_prompt = (
                prompt.message_example_prompt.replace(
                    "{{message_example}}", self.message_example
                )
                .replace("{{user}}", self.user)
                .replace("{{char}}", self.char)
            )
            # self.prompt.append({"role": "system", "content": self.message_example_prompt})
            self.prompt += self.message_example_prompt + "\n\n"
        if self.agent_config["Agent"]["prompt"]:
            # self.prompt.append({"role":  "system", "content": self.agent_config["Agent"]["prompt"]})
            self.prompt += self.agent_config["Agent"]["prompt"] + "\n\n"

    def __init__(self, agent_id: str):
        self.lock = Lock()
        self.update_config(agent_id)
        # self.char = config["char"]
        # self.user = config["user"]
        # self.char_settings = config["char_settings"]
        # self.char_personalities = config["char_personalities"]
        # self.message_example = config["message_example"]
        # self.mask = config["mask"]

        # self.is_data_base = config["is_data_base"]
        # self.data_base_thresholds = config["data_base_thresholds"]
        # self.data_base_depth = config["data_base_depth"]

        # self.is_long_mem = config["is_long_mem"]
        # self.is_check_memorys = config["is_check_memorys"]
        # self.mem_thresholds = config["mem_thresholds"]

        # self.is_core_mem = config["is_core_mem"]
        # self.llm_config = config["llm"]

        # 创建上下文
        self.msg_data = []

        # 上下文缓存
        self.msg_data_tmp = []
        try:
            with open(
                f"./data/agents/{self.agent_id}/history.yaml", "r", encoding="utf-8"
            ) as f:
                msg_list = yaml.safe_load(f)
                self.msg_data = msg_list[-CConfig.config["Agent"]["context_length"] :]
                Log.logger.info(f"当前上下文长度：{len(msg_list)}")
        except:
            pass
        if self.agent_config["Agent"]["start_with"] and len(self.msg_data) == 0:
            for i in range(len(self.agent_config["Agent"]["start_with"])):
                role = "assistant"
                if i % 2 == 0:
                    role = "user"
                self.msg_data_tmp.append(
                    {"role": role, "content": self.agent_config["Agent"]["start_with"][i]}
                )

        # 载入提示词
        # self.prompt = []
        # # self.prompt.append({"role": "system", "content": f"当前系统时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}"})
        # # tt = '''6. 注意输出文字的时候，将口语内容使用""符号包裹起来，并且优先输出口语内容，其他文字使用()符号包裹。'''
        # self.long_mem_prompt = prompt.long_mem_prompt
        # self.data_base_prompt = prompt.data_base_prompt
        # self.core_mem_prompt = prompt.core_mem_prompt
        # if self.char_settings:
        #     self.system_prompt = prompt.system_prompt.replace("{{char}}", self.char).replace("{{user}}", self.user)
        #     self.char_setting_prompt = prompt.char_setting_prompt.replace("{{char_setting_prompt}}", self.char_settings).replace("{{char}}", self.char).replace("{{user}}", self.user)
        #     self.prompt.append({"role": "system", "content": self.system_prompt})
        #     self.prompt.append({"role": "system", "content": self.char_setting_prompt})
        # if self.char_personalities:
        #     self.char_Personalities_prompt = prompt.char_Personalities_prompt.replace("{{char_Personalities_prompt}}", self.char_personalities).replace("{{char}}", self.char).replace("{{user}}", self.user)
        #     self.prompt.append({"role": "system", "content": self.char_Personalities_prompt})
        #     # self.prompt += self.char_Personalities_prompt + "\n\n"
        # if self.mask:
        #     self.mask_prompt = prompt.mask_prompt.replace("{{mask_prompt}}", self.mask).replace("{{char}}", self.char).replace("{{user}}", self.user)
        #     self.prompt.append({"role": "system", "content": self.mask_prompt})
        #     # self.prompt += self.mask_prompt + "\n\n"
        # if self.message_example:
        #     self.message_example_prompt = prompt.message_example_prompt.replace("{{message_example}}", self.message_example).replace("{{user}}", self.user).replace("{{char}}", self.char)
        #     self.prompt.append({"role": "system", "content": self.message_example_prompt})
        #     # self.prompt += self.message_example_prompt + "\n\n"
        # if config["prompt"]:
        #     self.prompt.append({"role":  "system", "content": config["prompt"]})
        #     # self.prompt += config["prompt"]

        # 创建系统时间戳
        self.tt = int(time.time())

        # 创建数据存储文件夹
        os.path.exists(f"./data/agents/{self.agent_id}/memorys") or os.makedirs(
            f"./data/agents/{self.agent_id}/memorys"
        )
        os.path.exists(f"./data/agents/{self.agent_id}/data_base") or os.makedirs(
            f"./data/agents/{self.agent_id}/data_base"
        )

        # 加载角色记忆
        # if self.is_long_mem:
        self.Memorys = long_mem.Memorys()

        # 加载核心记忆
        # if self.is_core_mem:
        self.Core_mem = core_mem.Core_Mem()

        # 载入知识库
        # if self.is_data_base:
        self.DataBase = data_base.DataBase()

    # 知识库内容检索
    def get_data(self, msg: str, res_msg: list) -> str:
        msg_list = jionlp.split_sentence(msg, criterion="fine")
        res_ = self.DataBase.search(msg_list)
        if res_ != "":
            res_msg.append(res_)

    # 提取、插入核心记忆
    def insert_core_mem(self, msg2: str, msg3: str, msg1: str):
        mmsg = prompt.get_core_mem.replace(
            "{{memories}}", json.dumps(self.Core_mem.mems[-100:], ensure_ascii=False)
        )
        if self.msg_data[-1]["role"] != "assistant":
            return
        re_msg = "对话内容：助手：" + msg1 + "\n用户：" + msg2 + "\n助手：" + msg3
        key = self.llm_config["key"]
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        data = {
            "model": self.llm_config["model"],
            "messages": [
                {"role": "system", "content": mmsg},
                {"role": "user", "content": re_msg},
            ],
        }
        try:
            res = requests.post(
                self.llm_config["api"], json=data, headers=headers, timeout=15
            )
            res_msg = res.json()["choices"][0]["message"]["content"]
            mem_list = ast.literal_eval(
                jionlp.extract_parentheses(res_msg, "[]")[0]
                .replace(" ", "")
                .replace("\n", "")
            )
            if len(mem_list) > 0:
                self.Core_mem.add_memory(mem_list)
        except:
            return

    # 获取发送到大模型的上下文
    def get_msg_data(self, msg: str) -> list:
        # index = len(self.msg_data) - 1
        # g_t = Thread(target=self.insert_core_mem, args=(msg, index,))
        # g_t.daemon = True
        # g_t.start()

        ttt = int(time.time())
        self.tt = ttt
        t_n = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ttt))
        # self.prompt[0] = {"role": "system", "content": f"当前现实世界时间：{t_n}"}
        # self.tmp_mem = f"时间：{t_n}\n{self.user}：{msg.strip()}\n"
        t_list = []
        data_base = []
        mem_msg = []
        res_msg = []
        core_mem = []
        res_msg.append({"role": "system", "content": self.prompt})

        # 检索世界书
        if self.is_data_base:
            tt = Thread(
                target=self.get_data,
                args=(
                    msg,
                    data_base,
                ),
            )
            tt.daemon = True
            t_list.append(tt)
            tt.start()

        # 搜索记忆
        if self.is_long_mem:
            tt = Thread(target=self.Memorys.get_memorys, args=(msg, mem_msg, t_n))
            tt.daemon = True
            t_list.append(tt)
            tt.start()

        # 搜索核心记忆
        if self.is_core_mem:
            tt = Thread(
                target=self.Core_mem.find_mem,
                args=(
                    msg,
                    core_mem,
                ),
            )
            tt.daemon = True
            t_list.append(tt)
            tt.start()

        # 等待查询结果
        for tt in t_list:
            tt.join()

        # 合并上下文、世界书、记忆信息
        tmp_msg = """"""
        if self.is_data_base and data_base:
            # self.msg_data.append({"role": "system", "content": self.data_base_prompt.replace("{{data_base}}", data_base[0]).replace("{{user}}", self.user).replace("{{char}}", self.char)})
            # self.msg_data_tmp.append({"role": "system", "content": self.data_base_prompt.replace("{{data_base}}", data_base[0]).replace("{{user}}", self.user).replace("{{char}}", self.char)})
            tmp_msg += (
                self.data_base_prompt.replace("{{data_base}}", data_base[0])
                .replace("{{user}}", self.user)
                .replace("{{char}}", self.char)
            )
        if self.is_core_mem and core_mem:
            # self.msg_data.append({"role": "system", "content": self.core_mem_prompt.replace("{{core_mem}}", core_mem[0]).replace("{{user}}", self.user).replace("{{char}}", self.char)})
            # self.msg_data_tmp.append({"role": "system", "content": self.core_mem_prompt.replace("{{core_mem}}", core_mem[0]).replace("{{user}}", self.user).replace("{{char}}", self.char)})
            tmp_msg += (
                self.core_mem_prompt.replace("{{core_mem}}", core_mem[0])
                .replace("{{user}}", self.user)
                .replace("{{char}}", self.char)
            )
        if self.is_long_mem and mem_msg:
            # self.msg_data.append({"role": "system", "content": self.long_mem_prompt.replace("{{memories}}", mem_msg[0]).replace("{{user}}", self.user).replace("{{char}}", self.char)})
            # self.msg_data_tmp.append({"role": "system", "content": self.long_mem_prompt.replace("{{memories}}", mem_msg[0]).replace("{{user}}", self.user).replace("{{char}}", self.char)})
            tmp_msg += (
                self.long_mem_prompt.replace("{{memories}}", mem_msg[0])
                .replace("{{user}}", self.user)
                .replace("{{char}}", self.char)
            )
        # self.msg_data_tmp.append({"role": "system", "content": f"当前现实世界时间：{t_n}；一定要基于现实世界时间做出适宜的回复。"})

        # 合并上下文、世界书、记忆信息
        tmp_msg += f"""
<当前时间>{t_n}</当前时间>
<用户对话内容或动作>
{msg}
</用户对话内容或动作>
"""
        # self.msg_data_tmp.append({"role": "user", "content": tmp_msg})
        self.msg_data_tmp = [{"role": "user", "content": tmp_msg}]
        # self.msg_data_tmp = tmp_msg_data
        return res_msg + self.msg_data + self.msg_data_tmp

    # 刷新上下文内容
    def add_msg(self, msg: str):
        self.msg_data_tmp.append({"role": "assistant", "content": msg})
        msg_data_tmp = self.msg_data_tmp.copy()
        m1 = msg_data_tmp[-2]["content"]

        try:
            ttt1 = Thread(
                target=self.insert_core_mem,
                args=(
                    m1,
                    self.msg_data_tmp[-1]["content"],
                    self.msg_data[-1]["content"],
                ),
            )
            ttt1.daemon = True
            ttt1.start()
        except Exception as e:
            Log.logger.error(f"核心记忆插入失败：{self.msg_data_tmp}，错误：{e}")

        ttt2 = Thread(
            target=self.Memorys.add_memory1,
            args=(msg_data_tmp, self.tt, self.llm_config),
        )
        ttt2.daemon = True
        ttt2.start()

        self.msg_data += self.msg_data_tmp
        self.msg_data = self.msg_data[-CConfig.config["Agent"]["context_length"] :]
        # write_data = {
        #     "messages": self.msg_data[-60:]
        # }

        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)  # 设置缩进格式
        yaml.default_flow_style = False  # 禁用流式风格（更易读）
        yaml.allow_unicode = True  # 允许 unicode 字符（如中文）
        with open(
            f"./data/agents/{self.agent_id}/history.yaml", "a", encoding="utf-8"
        ) as f:
            yaml.dump(self.msg_data_tmp, f)
            # for mm in self.msg_data_tmp:
            #     role = mm["role"]
            #     content = mm["content"]
            #     f.write(f"【{role}】：{content}\n")
            # f.write("\n")
        self.msg_data_tmp = []
