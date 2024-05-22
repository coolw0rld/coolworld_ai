from MissionClearCheckModel import MissionClearCheckImageModel
from MissionGenerateModel import MissionGenerateModel
from LanguageVotingModel import LanguageVotingModel
from MissionGenerateModel import SURVEY_MATCHER
import warnings
warnings.filterwarnings('ignore')

language_voting_model = LanguageVotingModel()
generate_model = MissionGenerateModel("PromptForm/generate_preprocess_form.txt",
                                      language_voting_model=language_voting_model)
clear_check_model_image = MissionClearCheckImageModel("PromptForm/clear_check_form.txt",
                                                      language_voting_model=language_voting_model)

ans = [
    [2, 0, 1, 0, 0],
    [0, 2, 0, 0, 1],
    [0, 0, 2, 1, 0],
    [0, 1, 0, 2, 0],
    [1, 0, 0, 0, 2]
]

for a in ans:
    print(generate_model(a))
    print(generate_model(a))