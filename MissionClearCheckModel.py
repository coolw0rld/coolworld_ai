from typing import *
from OpenAITextModel import OpenAIModel
from copy import deepcopy
import cv2
import base64

"""
미션 클리어 판단 모델
"""


class MissionClearCheckVideoModel(OpenAIModel):
    def __init__(self,
                 path_of_form: str,
                 openai_api_token: str = "",
                 gpt_model: str = "gpt-4-vision-preview",
                 max_tokens: int = 200,
                 form_encoding="UTF8",
                 max_frame: int = 50,
                 image_size: int = 768,
                 ):
        super().__init__(openai_api_token, gpt_model, max_tokens)

        self.max_frame = max_frame
        self.image_size = image_size
        with open(path_of_form, "r", encoding=form_encoding) as f:
            self.form = f.read()

    def preprocess(self, prompt: dict, video: cv2.VideoCapture) -> list:
        final_prompt = deepcopy(self.form)
        for key, answer in prompt.items():
            final_prompt.replace(f"<{key}>", str(answer))

        base64Frames = []
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        video.release()

        return [final_prompt, *map(lambda x: {"image": x, "resize": self.image_size}, base64Frames[0::self.max_frame])]

    def postprocess(self, result: str) -> bool:
        if result == "예":
            return True
        elif result == "아니오":
            return False
        else:
            # result를 바탕으로 긍부정 분류 모델 꽂아넣어 그거 결과로
            return True  # 이건 임시


class MissionClearCheckImageModel(OpenAIModel):
    def __init__(self,
                 path_of_form: str,
                 openai_api_token: str = "",
                 gpt_model: str = "gpt-4-vision-preview",
                 max_tokens: int = 200,
                 form_encoding="UTF8",
                 image_size: int = 768,
                 ):
        super().__init__(openai_api_token, gpt_model, max_tokens)

        self.image_size = image_size
        with open(path_of_form, "r", encoding=form_encoding) as f:
            self.form = f.read()

    def preprocess(self, prompt: dict, image: bytes) -> list:
        final_prompt = deepcopy(self.form)
        for key, answer in prompt.items():
            final_prompt.replace(f"<{key}>", str(answer))

        base64_str = base64.b64encode(image)
        imgdata = base64.b64decode(base64_str)

        return [final_prompt, [{"image": imgdata, "resize": self.image_size}]]

    def postprocess(self, result: str) -> bool:
        if result == "예":
            return True
        elif result == "아니오":
            return False
        else:
            # result를 바탕으로 긍부정 분류 모델 꽂아넣어 그거 결과로
            return True  # 이건 임시
