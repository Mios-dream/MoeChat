import os
import yaml
import hashlib
import pickle
import numpy as np
import faiss
from utils import embedding
from utils import log as Log
from models.types.assistant_info import AssistantInfo

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def _sum_md5(file_path: str):
    """
    计算文件的MD5值
    """
    with open(file_path, "rb") as file:
        md5_obj = hashlib.md5()
        while True:
            data = file.read(4096)  # 每次读取4096字节
            if not data:
                break
            md5_obj.update(data)
    md5_value = md5_obj.hexdigest()
    return md5_value


class DataBase:

    def __init__(self, agent_config: AssistantInfo):
        # 初始化数据库配置
        self.agent_id = agent_config.name
        self.thresholds = float(agent_config.settings.loreBooksThreshold)
        self.top_k = int(agent_config.settings.loreBooksDepth)
        self.path = f"./data/agents/{self.agent_id}/data_base"
        self.databases = []

        # 如果不存在创建数据库目录
        os.makedirs(self.path + "/tmp/labels", exist_ok=True)

        base_list = {}
        # 如果存在标签文件，加载标签
        if os.path.exists(self.path + "/tmp/label.yaml"):
            with open(self.path + "/tmp/label.yaml", "r", encoding="utf-8") as f:
                if base_list:
                    base_list = yaml.safe_load(f)
        # 搜索世界书目录下的所有文件
        file_list = os.listdir(self.path)
        books = []
        books_path = []
        # 遍历目录下所有文件，筛选出世界书文件
        for file in file_list:
            file_path = os.path.join(self.path, file)
            if os.path.isfile(file_path):
                f_md5 = _sum_md5(file_path)
                if file not in base_list or base_list[file] != f_md5:
                    books.append(file)
                    books_path.append(file_path)

        # 向量化世界书内容，并单独保存缓存文件
        for index in range(len(books)):
            with open(books_path[index], "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                try:
                    tmp1 = []
                    tmp2 = []
                    for data_key in data:
                        tmp1.append(data_key)
                        tmp2.append(data[data_key])
                    vect_list = embedding.t2vect(tmp1)
                    res_data = {"vect": vect_list, "text": tmp2}
                    pick_data = pickle.dumps(res_data)
                    with open(f"{self.path}/tmp/labels/{books[index]}.pkl", "wb") as f2:
                        f2.write(pick_data)
                    Log.logger.info(
                        f"成功向量化【{books[index]}】世界书，共加载{len(tmp1)}条数据。"
                    )
                    base_list[books[index]] = _sum_md5(books_path[index])
                except:
                    Log.logger.error(f"[错误]世界书[{books[index]}]加载错误")
                    base_list[books[index]] = _sum_md5(books_path[index])

        # 加载向量，建立数据库、索引，并保存，更新总表内容
        file_list = os.listdir(f"{self.path}/tmp/labels")
        tmp_list = []
        for file in file_list:
            if not os.path.isfile(f"{self.path}/tmp/labels/{file}"):
                continue
            with open(f"{self.path}/tmp/labels/{file}", "rb") as f:
                tmp_data = pickle.load(f)
            tmp_list.append(tmp_data["vect"])
            self.databases += tmp_data["text"]
            Log.logger.info(
                f"成功加载【{file}】世界书，共加载{len(tmp_data['vect'])}条数据。"
            )
        with open(f"{self.path}/tmp/label.yaml", "w") as f:
            yaml.safe_dump(base_list, f)
        if len(tmp_list) > 0:
            self.vects = np.concatenate(tmp_list)
        else:
            v = embedding.t2vect(["填充用废物"])
            tmp_list.append(v)
            self.vects = np.concatenate(tmp_list)
            self.databases.append("这是一条没有用的知识")
        # 建立索引
        self.index = faiss.IndexFlatIP(len(self.vects[0]))

        self.index.add(self.vects)

    # 查询接口
    def search(self, text: list[str]) -> str:
        """
        查询数据库中与输入文本最相关的内容
        """
        # 储存返回结果
        msg = ""
        # 向量化查询内容
        vect = embedding.t2vect(text)
        # 查询
        D, I = self.index.search(vect, self.top_k)
        # 返回结果
        for index in range(len(D)):
            for i2 in range(len(D[index])):
                if D[index][i2] >= self.thresholds:
                    msg += self.databases[I[index][i2]] + "\n\n"
        return msg
