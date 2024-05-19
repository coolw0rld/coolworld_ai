###############################################################################
# Language Model
#
# Language Voting Model을 이용하여 프롬프트를 생성, 전송, 답 후처리 후 반환까지 하는 모델입니다.
#
# 해당 모델을 절대 단독으로 인스턴스화하지 말고, 
# 상속 후 생성자, preprocess, postprocess 메소드를 오버라이드하여 이용하십시오.
# generate 메소드와 __call__ 메소드는 절대 오버라이드하지 마십시오.
###############################################################################
from typing import *

import openai
from openai import OpenAI
from anthropic import Anthropic
import cv2
import os
import Config
from LanguageVotingModel import LanguageVotingModel
import httpx
import base64


class LanguageModel:
    def __init__(self,
                 gpt_model: str,
                 claude_model: str,
                 max_tokens: int,
                 language_voting_model: LanguageVotingModel
                 ):
        assert gpt_model in Config.GPT_MODEL_LIST, Config.GPT_WRONG_MODEL_ERROR_MESSAGE
        assert claude_model in Config.CLAUDE_MODEL_LIST, Config.CLAUDE_WRONG_MODEL_ERROR_MESSAGE

        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", f"{Config.OPENAI_TOKEN}"))
        self.anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", f"{Config.ANTHROPIC_TOKEN}"))

        self.gpt_model = gpt_model
        self.claude_model = claude_model

        self.max_tokens = max_tokens

        self.languageVotingModel = language_voting_model

    def preprocess(self, prompt, system: str = "", imageURL: str = None) -> list:
        return [prompt]

    def postprocess(self, result: str):
        return result

    def generate(self,
                 content: dict) -> str:
        gpt_params = {
            "model": self.gpt_model,
            "messages": [
                {
                    "role": "system",
                    "content": content["system"],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": content["user"],
                        }
                    ],
                },
            ],
            "max_tokens": self.max_tokens,
        }


        claude_params = {
            "model": self.claude_model,
            "system":content["system"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": content["user"]
                        }
                    ],
                },
            ],
            "max_tokens": self.max_tokens,
        }

        if len(content["image"]) > 0:
            for c in content["image"]:
                gpt_params["messages"][1]["content"].append({"type":"image_url", "image_url":{"url":c}})

                image_data = base64.b64encode(httpx.get(c).content).decode("utf-8")
                claude_params["messages"][0]["content"].append({"type":"image", "source":{"type":"base64", "media_type":"image/jpeg", "data":image_data}})


        gpt_result = self.openai_client.chat.completions.create(**gpt_params).choices[0].message.content
        claude_result = self.anthropic_client.messages.create(**claude_params).content[0].text

        return self.languageVotingModel.voting(gpt_result, claude_result)

    def __call__(self,
                 prompt: dict,
                 imageURL: str = None):
        preprocessed_prompt = self.preprocess(prompt, imageURL)
        result = self.generate(preprocessed_prompt)
        postprocessed_result = self.postprocess(result)
        return postprocessed_result
