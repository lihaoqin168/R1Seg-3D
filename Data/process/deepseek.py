import requests
import json
from typing import List, Tuple, Optional
import os
from tqdm import tqdm
# from examples.vllm_wrapper import vLLMWrapper
import re
import csv
import json
from transformers import PreTrainedTokenizer, GenerationConfig, StoppingCriteriaList
from typing import Optional, Callable, List, Tuple, Union
import copy
import torch
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList
from packaging import version

DEEPSEEK_API_URL = "http://117.145.189.131:48081/api/generate"
HISTORY_FORMATTER = lambda role, content: {"role": role, "content": content}

class DeepSeekChatClient:
    def __init__(
        self,
        api_key: str = 'chjjw',
        model: str = "deepseek-r1-70b",
        system_prompt: str = "你是一个专业的AI助手"
    ):
        """
        :param api_key: DeepSeek API密钥
        :param model: 模型标识，可选值如 deepseek-r1-70b-chat
        :param system_prompt: 系统级角色设定
        """
        self.headers = {
            "X-API-Key": "chjjw",  
            "Content-Type": "application/json"
        }
        self.model = model
        self.system_prompt = system_prompt

    def _format_messages(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None
    ) -> List[dict]:
        """将对话历史转换为DeepSeek API要求的消息格式"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if history:
            for h in history:
                messages.append(HISTORY_FORMATTER("user", h))
                messages.append(HISTORY_FORMATTER("assistant", h))
        
        messages.append({"role": "user", "content": query})
        # print(messages)
        return messages

    def chat(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        执行对话请求
        :param query: 当前用户输入
        :param history: 历史对话列表，格式 [(用户输入, 助手回复), ...]
        :return: 模型生成的回复文本
        """
        data = {
            "model": "deepseek-r1:70b",
            "messages": self._format_messages(query, history),
            "prompt": query, 
            "stream": False
        }

        response = requests.post(
            DEEPSEEK_API_URL,
            headers=self.headers,
            data=json.dumps(data)
        )
        if response.status_code == 200:
            # Print the response
            return response.json()['response'].strip()
            # print("Response:", response.json())
        else:
            print("Failed to get response. Status code:", response.status_code)

# 初始化客户端
client = DeepSeekChatClient(
    api_key="chjjw",
    model="deepseek-r1:70b",
    system_prompt="您是一名医疗人工智能视觉助手，可以分析单张 CT 图像。您会收到 CT 图像的文件名和医疗诊断报告。任务是利用提供的 CT 图像和报告信息，就该图像创建 9 个合理的问题。每个问题对应 4 个选项，这些问题来自以下 5 个方面：1). 平面（轴位、矢状位、冠状位）；2). CT相位（非对比、对比、动脉相、门静脉相、静脉相、延迟相、实质相、肾皮质相、双相、肾排泄相、动静脉混合相、髓核相等）或窗口（骨、肺、窗口等）；3). 器官；4). 异常类型或描述；5). 异常位置；"
)
# # 定义对话历史
# history = [
#     ("什么是CT扫描?", 
#      "CT(Computed Tomography)扫描是利用X射线束对人体进行断层扫描...")
# ]

# # 执行对话
# try:
#     response = client.chat(
#         query="腹部疼痛",
#         history=history,
#         temperature=0.3,
#         max_tokens=512
#     )
#     print("DeepSeek回复:", response)
# except Exception as e:
#     print("请求出错：", str(e))


#=================================================记载数据======================================================
file_path = "/home/machao/M3D-main/M3D_Cap.json"
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

group_id = 0
group_num = 2
train_data = data.get('train', [])
total_items = len(train_data)
chunk_size = total_items // group_num

split_train_data = [train_data[i:i + chunk_size] for i in range(0, total_items, chunk_size)]

data_list = split_train_data[group_id]
data_len = len(data_list)
print("data_len: ",data_len)

# vqa_data_name = "M3D_VQA_" + str(group_id) + ".csv"
path = 'test.csv'
with open(path, "a", newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(
        ["Image Path", "Text", "Question Type", "Question", "Choice A", "Choice B", "Choice C", "Choice D", "Answer", "Answer Choice"])

image_file = "M3D_Cap_npy/ct_quizze/012518/Axial_C__portal_venous_phase.npy"

with open("M3D_Cap_npy/ct_quizze/012518/text.txt", "r") as f:
    text = f.read()

user = f"""
        Image: {image_file}
        Report: {text}

        Desired format:
        1). Planes
        Question-1: ...? Choice: A. ... B. ... C. ... D. ... Answer: A. ...\n

        2). CT phase
        Question-2: ...? Choice: A. ... B. ... C. ... D. ... Answer: A. ...\n

        3). Organ
        Question-3: ...? Choice: A. ... B. ... C. ... D. ... Answer: A. ...\n

        4). Abnormality type or description
        Question-4: ...? Choice: A. ... B. ... C. ... D. ... Answer: A. ...\n

        Question-5: ...? Choice: A. ... B. ... C. ... D. ... Answer: A. ...\n

        Question-6: ...? Choice: A. ... B. ... C. ... D. ... Answer: A. ...\n

        5). Abnormality position
        Question-7: ...? Choice: A. ... B. ... C. ... D. ... Answer: A. ...\n

        Question-8: ...? Choice: A. ... B. ... C. ... D. ... Answer: A. ...\n

        Question-9: ...? Choice: A. ... B. ... C. ... D. ... Answer: A. ...\n

        Make the correct answers randomly distributed among the four choices.
        If there is a true or false question, please ensure that the proportion of yes and no is equivalent. For example, Is ... ? Are ... ?, Do ... ?, Does ... ?, Did ... ?, Can ... ?.
        Please do NOT ask directly what organs or abnormalities are visible in the image, as the answers are not unique. It would be best to use specific descriptions in your questions to ensure that other people can get an accurate answer even without providing choices.

        Please be careful not to mention the file name and report. Always ask questions and answer as if directly looking at the image.
        """
system = """
You are a medical AI visual assistant that can analyze a single CT image. You receive the file name of the CT image and the medical diagnosis report. The report describes multiple abnormal lesions in the image.
The task is to use the provided CT image and report information to create plausible 9 questions about the image.
Each question corresponds to four options, and these questions come from the following 5 aspects:
1). Planes (axial, sagittal, coronal);
2). CT phase (non-contrast, contrast, arterial phase, portal venous phase, venous phase, delayed phase, parenchymal phase, renal cortical phase, dual phase, renal excretory phase, mixed arteriovenous, myelography, etc.) or window ( bone, lung, window, etc.);
3). Organ;
4). Abnormality type or description;
5). Abnormality position;
"""
response = client.chat(query=user, history=None,
                            system=system)
print(response)
questions = re.findall(r'Question-(\d+): (.*?)(?: Choice: (.*?))? Answer: ([A-D])\. (.*?)\n', response)

with open(path, "a", newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)

    for q in questions:
        question_num, question, choices, an_choice, answer = q
        choices = re.findall(r"([A-D])\. (.+?)(?=(?: [A-D]\.|$))", choices)
        choices_dict = {choice[0]: choice[1] for choice in choices}

        for option in ['A', 'B', 'C', 'D']:
            if option not in choices_dict:
                choices_dict[option] = 'NA'

        if int(question_num) < 4:
            question_type = question_num
        elif int(question_num) < 7:
            question_type = str(4)
        else:
            question_type = str(5)

        csvwriter.writerow(
            [data["image"], text, question_type, question, choices_dict['A'], choices_dict['B'], choices_dict['C'], choices_dict['D'],
                answer, an_choice])
    # except:
    #     print("Error in " + "id:" + str(i) + " " + data["image"])
