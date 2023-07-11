import os
import pytest
import json
import numpy as np

from guardrails.embedding import OpenAIEmbedding

COUNTRIES = [
    "China",
    "The United States",
    "Canada",
    "India",
    "Sweden",
    "Egypt",
    "Nigeria",
    "Brazil",
    "The United Kingdom"
]

PROMPT_FORMATS = [
    "What is the capital of {}",
    "What is {}'s capital",
    "The capital of {}",
    "{}'s capital"
]

def _get_prompt(country, prompt_idx):
    return PROMPT_FORMATS[prompt_idx].format(country)

def _get_all_prompts():
    prompts = []
    for i, prompt_fmt in enumerate(PROMPT_FORMATS):
        prompts += [_get_prompt(country, i) for country in COUNTRIES]
    return prompts

def _cosine_sim(emb_a, emb_b):
    return np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))

@pytest.fixture()
def set_up():
    return {
        key: json.loads(open('./test_assets/cache/{}.json'.format(key), 'r').read())
        for key in ['embeddings', 'responses']
    }

@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") is None, reason="openai api key not set"
)
class TestCache:

    def test_foo(self, set_up):
        d = set_up
        print('asdfasdf', d['responses'])