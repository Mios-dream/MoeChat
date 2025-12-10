import os
import yaml
import jionlp as jio
import time
from utils import embedding, prompt
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString
import numpy as np
import pickle
import requests
import jionlp
from bisect import bisect_left, bisect_right
from utils import log as Log
from models.types.assistant_info import AssistantInfo


class Memory:

    def __init__(self, agent_config: AssistantInfo):
        # 初始化角色配置
        self.agent_id = agent_config.name
        self.char = agent_config.name
        self.user = agent_config.user
        self.thresholds = agent_config.settings.longMemoryThreshold
        self.enableLongMemorySearchEnhance = (
            agent_config.settings.enableLongMemorySearchEnhance
        )
        # 初始化记忆数据结构
        self.memories_key = []  # 记录所有记忆的key，秒级整形时间戳。
        self.memories_data = {}  # 记录所有记忆的文本数据。
        self.vectors = []  # 记录文本tag向量
        # self.tags = []
        # self.date_time = []

        # self.user_vectors = np.ndarray()
        # self.char_vectors = np.ndarray()

        # 加载记忆
        msg_vectors = []
        path = f"./data/agents/{self.agent_id}/memory"
        for root, dirs, files in os.walk(path):
            # print(files)
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    if file_path.find(".yaml") == -1:
                        continue
                    msgs = []
                    tag = []
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                        for key in data:
                            self.memories_data[key] = (
                                str(data[key]["msg"])
                                .replace("{{user}}", self.user)
                                .replace("{{char}}", self.char)
                            )
                            tag.append(data[key]["text_tag"])
                            m_list = data[key]["msg"].split("\n")
                            self.memories_key.append(key)
                            # self.date_time.append(m_list[0])
                            msgs.append(f"{m_list[1]}{m_list[2]}")
                    if os.path.exists(file_path.replace(".yaml", ".pkl")):
                        with open(file_path.replace(".yaml", ".pkl"), "rb") as f:
                            tmp_data = pickle.load(f)
                            self.vectors += tmp_data
                        Log.logger.info(f"加载记忆【{file}】")
                    else:
                        Log.logger.info(f"向量化记忆【{file}】")
                        t_v = embedding.t2vect(tag)
                        msg_vectors.append(t_v)
                        with open(file_path.replace(".yaml", ".pkl"), "wb") as f:
                            pickle.dump(t_v, f)
                        Log.logger.info(f"向量化完成【{file}】")
                except:
                    Log.logger.error(f"【{file_path}】记忆加载失败...")
                    continue
            # print(f"[错误]【{path}】")
        # self.char_vectors = np.concatenate(char_v)
        # self.user_vectors = np.concatenate(user_v)
        # if msg_vectors:
        #     self.vectors = np.concatenate(msg_vectors)
        Log.logger.info(
            f"共加载{len(self.memories_key)}条记忆...{len(self.vectors)}条记忆向量"
        )

        # 建立、加载索引数据库
        # if os.path.exists(self.char_file_path) and os.path.exists(self.user_file_path):
        #     with open()

    def find_range_indices(self, low, high) -> list | None:
        start_idx = bisect_left(self.memories_key, low)  # 找到第一个 >= low 的索引
        end_idx = bisect_right(self.memories_key, high)  # 找到最后一个 <= high 的索引
        if end_idx == 0 or start_idx >= len(
            self.memories_key
        ):  # 如果没有找到匹配的元素
            return None
        return [start_idx, end_idx - 1]

    # 获取与文本相关的记忆
    def get_memories(self, msg: str, res_msg: list, t_n: str = "时间"):
        """
        获取与文本相关的记忆
        Args:
            msg (str): 输入的消息
            res_msg (list): 输出的消息列表
            t_n (str, optional): 时间实体名称. Defaults to "时间".
        """
        if not len(self.memories_key) > 0:
            return
        # t = time.time()
        time_span_list = []

        # 提取文本中的时间实体
        res = jio.ner.extract_time(
            f"[{t_n}]{msg}", time_base=time.time(), with_parsing=False
        )

        # 获取与文本关联的时间范围信息
        if len(res) > 1:
            for t in res[1:]:
                try:
                    res_t = jio.parse_time(t["text"], time_base=res[0]["text"])
                    time_st1 = int(
                        time.mktime(
                            time.strptime(res_t["time"][0], "%Y-%m-%d %H:%M:%S")
                        )
                    )
                    time_st2 = int(
                        time.mktime(
                            time.strptime(res_t["time"][1], "%Y-%m-%d %H:%M:%S")
                        )
                    )
                    time_span_list.append(time_st1)
                    time_span_list.append(time_st2)
                except:
                    Log.logger.error(f"获取时间区间失败")
        if not time_span_list:
            return

        # 提取键
        # key_list = []
        res_index = self.find_range_indices(time_span_list[0], time_span_list[1])
        if not res_index:
            return
        # 将时间范围内的记忆添加到结果中
        if self.enableLongMemorySearchEnhance:
            Log.logger.info(f"深度检索记忆，检索阈值{self.thresholds}")
            q_v = embedding.t2vect([msg])[0]
            tmp_msg = ""
            for index in range(res_index[0] + 1, res_index[1] + 1):
                rr = np.dot(self.vectors[index], q_v)
                if rr >= self.thresholds:
                    tmp_msg += str(self.memories_data[self.memories_key[index]])
                    tmp_msg += "\n"
            if len(tmp_msg) > 0:
                res_msg.append(tmp_msg)
            # mem_list = []
            # for key in key_list:
            #     mem_list.append(str(self.memorys_data[key]))
            # res_mem = embedding.test(msg, mem_list, self.thresholds)
            # if res_mem:
            #     res_msg.append(res_mem)
        else:
            tmp_mem = ""
            for index in range(res_index[0] + 1, res_index[1] + 1):
                tmp_mem += str(self.memories_data[self.memories_key[index]])
                tmp_mem += "\n"
            if len(tmp_mem) > 0:
                res_msg.append(tmp_mem)

    # 写入记忆
    def save_memory(self, m_data: dict):
        t_n = int(m_data["t_n"])
        self.memories_key.append(t_n)
        self.memories_data[t_n] = m_data["msg"]
        tag_vector = embedding.t2vect([m_data["text_tag"]])[0]
        self.vectors.append(tag_vector)
        time_st = time.localtime(t_n)
        file_name = f"{time_st.tm_year}-{time_st.tm_mon}-{time_st.tm_mday}.yaml"
        file_pkl = f"{time_st.tm_year}-{time_st.tm_mon}-{time_st.tm_mday}.pkl"
        data = {
            t_n: {
                "text_tag": m_data["text_tag"],
                "msg": LiteralScalarString(m_data["msg"]),
            }
        }
        Yaml = YAML()
        Yaml.preserve_quotes = True
        Yaml.width = 4096

        with open(
            f"./data/agents/{self.agent_id}/memory/{file_name}", "a", encoding="utf-8"
        ) as f:
            Yaml.dump(data, f)
        day_time = t_n - (t_n - time.timezone) % 86400
        index = bisect_left(self.memories_key, day_time)
        v_list = self.vectors[index:]
        with open(f"./data/agents/{self.agent_id}/memory/{file_pkl}", "wb") as f:
            pickle.dump(v_list, f)

    # 提取记忆摘要，记录长期记忆
    def add_memory(self, data: list, t_n: int, llm_config: dict):
        summary_memory_prompt = prompt.get_mem_tag_prompt
        res_msg = "用户：" + data[-2]["content"]
        res_body = {
            "model": llm_config["model"],
            "messages": [
                {"role": "system", "content": summary_memory_prompt},
                {"role": "user", "content": res_msg},
            ],
        }
        key = llm_config["key"]
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        res_tag = ""
        try:
            res = requests.post(
                llm_config["api"], json=res_body, headers=headers, timeout=15
            )
            res = res.json()["choices"][0]["message"]["content"]
            res = jionlp.remove_html_tag(res).replace(" ", "").replace("\n", "")
            Log.logger.info(f"记录日记结果【{res}】")
            if res.find("日常闲聊") == -1:
                res_tag = res
            else:
                res_tag = "日常闲聊"
        except:
            Log.logger.error("错误获取聊天信息失败！")
            res_tag = "日常闲聊"
        t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_n))
        m1 = data[-2]["content"]
        m2 = data[-1]["content"]
        c1 = "{{user}}"
        c2 = "{{char}}"
        m_data = {
            "t_n": t_n,
            "text_tag": res_tag,
            "msg": f"时间：{t_str}\n{c1}：{m1}\n{c2}：{m2}",
        }
        self.save_memory(m_data)
