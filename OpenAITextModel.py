from typing import *
from openai import OpenAI
import cv2
import os

GPT_MODEL_LIST = [
    "gpt-3.5-turbo",
    "gpt-4-turbo",
    "gpt-4-vision-preview",
]

class OpenAIModel:
    def __init__(self,
                 openai_api_token: str,
                 gpt_model: str,
                 max_tokens: int,
                 ):
        assert gpt_model in GPT_MODEL_LIST, "Wrong GPT Model. Please Check https://platform.openai.com/docs/models/continuous-model-upgrades"
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", f"{openai_api_token}"))

        self.gpt_model = gpt_model
        self.max_tokens = max_tokens

    def preprocess(self, prompt, video: Optional[cv2.VideoCapture]) -> list:
        return [prompt]

    def postprocess(self, result: str):
        return result

    def generate(self,
                 content: list) -> str:
        params = {
            "model": self.gpt_model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                },
            ],
            "max_tokens": self.max_tokens,
        }

        result = self.client.chat.completions.create(**params)
        return result.choices[0].message.content

    def __call__(self,
                 prompt: dict,
                 video: Optional[cv2.VideoCapture] = None):
        preprocessed_prompt = self.preprocess(prompt, video)
        result = self.generate(preprocessed_prompt)
        postprocessed_result = self.postprocess(result)
        return postprocessed_result
