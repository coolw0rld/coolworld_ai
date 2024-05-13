from MissionClearCheckModel import MissionClearCheckImageModel, MissionClearCheckVideoModel
from MissionGenerateModel import MissionGenerateModel
import cv2

TOKEN = ""

if __name__=="__main__":
    generate_model = MissionGenerateModel("PromptForm/generate_preprocess_form.txt", openai_api_token=TOKEN)
    clear_check_model_video = MissionClearCheckVideoModel("PromptForm/clear_check_form.txt", openai_api_token=TOKEN)
    clear_check_model_image = MissionClearCheckImageModel("PromptForm/clear_check_form.txt", openai_api_token=TOKEN)

    survey_result = {
        "num_use_car": 3,
        "category_energy": "전기",
        "num_eat_meat": 10,
        "buy_day": 20,
        "num_of_trash": 2,
        "day_of_recycle": 1,
        "amount_of_water": 5,
        "num_of_travel": 0,
    }

    mission = generate_model(survey_result)
    print(f"미션 : {mission}")

    video = cv2.VideoCapture("test/sample_video.mp4")
    result = clear_check_model_video({"mission": mission}, video=video)
    print(f"{'클리어 성공' if result else '클리어 실패'}")

    f = open('./test/input.jpg', 'rb')
    result = clear_check_model_image({"mission": mission}, image=f.read())
    f.close()
    print(f"{'클리어 성공' if result else '클리어 실패'}")