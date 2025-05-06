import os
import torch
from PIL import Image
from utils import *
from openai import OpenAI


class LLM_API_Captioner:
    def __init__(self, llm_name):
        
        llm_dict = {
            "QwenMax": {
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_key": "your own api key",
                "model": "qwen-max",
            },
            "DeepSeekV3": {
                "base_url": "https://api.deepseek.com",
                "api_key": "your own api key",
                "model": "deepseek-chat",
            },
            "Kimi": {
                "base_url": "https://api.moonshot.cn/v1",
                "api_key": "your own api key",
                "model": "moonshot-v1-8k",
            }
        }
        self.client = OpenAI(
            base_url=llm_dict[llm_name]["base_url"], 
            api_key=llm_dict[llm_name]["api_key"],
        )
        self.model = llm_dict[llm_name]["model"]

    def complete_chat(self,  prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt
        )
        return response.choices[0].message.content


def chat_with_Llama(cur_prompt, tokenizer, model):

    if isinstance(cur_prompt, str):
        cur_prompt = [cur_prompt]
    
    answers = []

    for i in range(len(cur_prompt)):
        if i == 0:
            chat = [{"role": "user", "content": cur_prompt[0]}]
        else:
            chat.append({"role": "assistant", "content": response})
            chat.append({"role": "user", "content": cur_prompt[i]})

        input_ids = tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        # print(input_ids)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids["input_ids"],
                max_new_tokens=4096,#8192,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=input_ids["attention_mask"]
            )
        response = outputs[0][input_ids["input_ids"].shape[-1] :]
        response = tokenizer.decode(response, skip_special_tokens=True)
        answers.append(response)
    return answers


def chat_with_api(cur_prompt, model):
    if isinstance(cur_prompt, str):
        cur_prompt = [cur_prompt]
    
    answers = []
    for i in range(len(cur_prompt)):
        if i == 0:
            chat = [{"role": "user", "content": cur_prompt[0]}]
        else:
            chat.append({"role": "assistant", "content": response})
            chat.append({"role": "user", "content": cur_prompt[i]})

        response = model.complete_chat(chat)
        answers.append(response)
    return answers