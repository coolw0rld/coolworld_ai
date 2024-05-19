from flask import Flask, request, jsonify, Response
from flask_restx import Api, Resource
from MissionClearCheckModel import MissionClearCheckImageModel
from MissionGenerateModel import MissionGenerateModel
from LanguageVotingModel import LanguageVotingModel
import json

app = Flask(__name__)
api = Api(app)

headers = {
    "Content-Type": "application/json"
}

@app.route("/mission_create", methods=['POST'])
def mission_create():
    json_data = request.get_json(force=True)
    survey_answer = list(json_data["answer"])

    return json.loads(generate_model(survey_answer))

@app.route("/check_clear", methods=['POST'])
def check_clear():
    json_data = request.get_json(force=True)
    mission = str(json_data["mission"])
    image_url = str(json_data["image"])
    result = clear_check_model_image({"mission": mission}, image_url)

    return json.dumps({"result":result})


if __name__ == "__main__":
    language_voting_model = LanguageVotingModel()
    generate_model = MissionGenerateModel("PromptForm/generate_preprocess_form.txt", language_voting_model=language_voting_model)
    clear_check_model_image = MissionClearCheckImageModel("PromptForm/clear_check_form.txt", language_voting_model=language_voting_model)

    app.run(debug=True, host='127.0.0.1', port=8080)