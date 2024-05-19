from MissionClearCheckModel import MissionClearCheckImageModel
from MissionGenerateModel import MissionGenerateModel
from LanguageVotingModel import LanguageVotingModel
import cv2
import Config

if __name__=="__main__":
    language_voting_model = LanguageVotingModel()

    generate_model = MissionGenerateModel("PromptForm/generate_preprocess_form.txt", language_voting_model=language_voting_model)
    clear_check_model_image = MissionClearCheckImageModel("PromptForm/clear_check_form.txt", language_voting_model=language_voting_model)

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

    # f = open('./test/input.jpg', 'rb')
    # result = clear_check_model_image({"mission": mission}, "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg")
    # f.close()
    # print(f"{'클리어 성공' if result else '클리어 실패'}")
