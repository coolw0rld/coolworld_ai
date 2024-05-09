from typing import *
from OpenAITextModel import OpenAIModel
from copy import deepcopy

"""
미션 생성 모델
"""


class MissionGenerateModel(OpenAIModel):
    def __init__(self,
                 path_of_form: str,
                 openai_api_token: str = "",
                 gpt_model: str = "gpt-4-turbo",
                 max_tokens: int = 200,
                 form_encoding="UTF8"
                 ):
        super().__init__(openai_api_token, gpt_model, max_tokens)

        with open(path_of_form, "r", encoding=form_encoding) as f:
            self.form = f.read()

    def preprocess(self, prompt: dict, video=None) -> list:
        final_prompt = deepcopy(self.form)
        for key, answer in prompt.items():
            final_prompt.replace(f"<{key}>", str(answer))
        return [final_prompt]

    def postprocess(self, result: str, **kwargs) -> str:
        return result
