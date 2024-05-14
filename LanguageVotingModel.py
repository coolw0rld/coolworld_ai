###############################################################################
# Language Voting Model
#
# Junyou Li, Qin Zhang, Yangbin Yu, Qiang Fu, & Deheng Ye. (2024). More Agents Is All You Need.
# [https://arxiv.org/abs/2402.05120]
#
# 여러개의 LM으로 Voting을 하는 모델입니다. 자세한 정보는 위의 논문을 참고하세요.
###############################################################################
from typing import *
from transformers import AutoTokenizer
import Config


class LanguageVotingModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)

    def BLEU_Score(self, s_gen: str, s_ref: str) -> float:
        t_gen = self.tokenizer(s_gen)['input_ids']
        t_ref = self.tokenizer(s_ref)['input_ids']

        common = len([t for t in t_ref if t in t_gen])
        total = len(t_gen)

        return common / total

    def voting(self, *args) -> str:
        assert len(args) >= 2, Config.VOTING_MODEL_LESS_SENTENCE_ERROR_MESSAGE

        #print(args)

        score = {s: 0.0 for s in args}

        for s_gen in args:
            for s_ref in args:
                if s_gen != s_ref:
                    score[s_gen] += self.BLEU_Score(s_gen, s_ref)
        #print(score)
        best_answer = list({k: v for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)}.keys())[0]

        return best_answer

    def __call__(self, *args):
        return self.voting(*args)
