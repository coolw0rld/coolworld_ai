###############################################################################
# Mission Clear Check Model
# (Language Model의 자식 클래스입니다.)
#
# 미션 생성 모델입니다.
###############################################################################
from typing import *
from LanguageModel import LanguageModel
from copy import deepcopy
from LanguageVotingModel import LanguageVotingModel


class MissionGenerateModel(LanguageModel):
    def __init__(self,
                 path_of_form: str,
                 language_voting_model: LanguageVotingModel,
                 gpt_model: str = "gpt-4-turbo",
                 claude_model: str = "claude-3-haiku-20240307",
                 max_tokens: int = 200,
                 form_encoding="UTF8"
                 ):
        super().__init__(gpt_model, claude_model, max_tokens, language_voting_model)

        with open(path_of_form, "r", encoding=form_encoding) as f:
            self.form = f.read()

    def preprocess(self, prompt: dict, imageUrl: str = None, system: str = "") -> list:
        final_prompt = deepcopy(self.form)
        for key, answer in prompt.items():
            final_prompt = final_prompt.replace(f"<{key}>", str(answer))
        return {"system":system, "user":final_prompt, "image":[]}

    def postprocess(self, result: str, **kwargs) -> str:
        return result
