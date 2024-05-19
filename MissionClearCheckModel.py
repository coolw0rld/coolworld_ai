###############################################################################
# Mission Clear Check Model
# (Language Model의 자식 클래스입니다.)
#
###############################################################################
from typing import *
from LanguageModel import LanguageModel
from copy import deepcopy
import cv2
import base64
from LanguageVotingModel import LanguageVotingModel


class MissionClearCheckImageModel(LanguageModel):
    def __init__(self,
                 path_of_form: str,
                 language_voting_model: LanguageVotingModel,
                 gpt_model: str = "gpt-4-vision-preview",
                 claude_model: str = "claude-3-haiku-20240307",
                 max_tokens: int = 200,
                 temperature: float = 0.0,
                 form_encoding="UTF8",
                 ):
        super().__init__(gpt_model, claude_model, max_tokens, temperature, language_voting_model)

        with open(path_of_form, "r", encoding=form_encoding) as f:
            self.form = f.read()

    def preprocess(self, prompt, imageURL: str, system="이미지들로 보았을 때 이 미션이 성공했을 것 같은지 예 아니오로만 대답해줘.") -> list:
        final_prompt = deepcopy(self.form)
        for key, answer in prompt.items():
            final_prompt = final_prompt.replace(f"<{key}>", str(answer))

        return {"system":system, "user":final_prompt, "image":[imageURL]}

    def postprocess(self, result: str) -> bool:
        if "예" in result:
            return True
        elif "아니오" in result:
            return False
        else:
            # result를 바탕으로 긍부정 분류 모델 꽂아넣어 그거 결과로
            return True  # 이건 임시
