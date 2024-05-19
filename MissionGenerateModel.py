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


SURVEY_MATCHER = {
    0:{0:"항상", 1:"가끔", 2:"안"},
    1:{0:"0번", 1:"1번", 2:"2번 이상"},
    2:{0:"항상", 1:"가끔", 2:"안"},
    3:{0:"10분 이하", 1:"10분에서 20분", 2:"20분 이상"},
    4:{0:"자주", 1:"가끔", 2:"안"},
}

class MissionGenerateModel(LanguageModel):
    def __init__(self,
                 path_of_form: str,
                 language_voting_model: LanguageVotingModel,
                 gpt_model: str = "gpt-4-turbo",
                 claude_model: str = "claude-3-opus-20240229",
                 max_tokens: int = 200,
                 form_encoding="UTF8"
                 ):
        super().__init__(gpt_model, claude_model, max_tokens, language_voting_model)
        self.form_encoding = form_encoding

        with open(path_of_form, "r", encoding=form_encoding) as f:
            self.form = f.read()

    def preprocess(self, prompt: list, imageUrl: str = None, system: str = "") -> list:
        final_prompt = deepcopy(self.form)

        for survey_no, ans in enumerate(prompt):
            final_prompt = final_prompt.replace(f"<ans{survey_no}>", SURVEY_MATCHER[survey_no][ans])

        if system=="":
            with open("./PromptForm/mission_generate_system_prompt.txt", "r", encoding=self.form_encoding) as f:
                system = f.read()

        print(final_prompt)

        return {"system":system, "user":final_prompt, "image":[]}

    def postprocess(self, result: str, **kwargs) -> str:
        return result
