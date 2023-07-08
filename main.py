import argparse
import numpy as np
import threading
import logging

from flask import Flask, request
import openai

TEXT_SIMILARITY_MODELS = ["ada", "babbage", "curie", "davinci"]

DEFAULT_CACHE_SIZE = 512 * 512
DEFAULT_TEXT_SIMILARITY_MODEL = "ada"
DEFAULT_SIMILARITY_THRESHOLD = 0.95

def _cache_size_type(x):
    y = int(x)
    if y < 1:
        raise argparse.ArgumentTypeError("Cache size must be greater than 0")
    return y

def _similarity_threshold_type(x):
    y = float(x)
    if y < 0 or y > 1:
        raise argparse.ArgumentTypeError("Similarity threshold must be in the range [0, 1]")
    return y

class Node(object):

    def __init__(self, prompt, embedding, response):
        self.prompt = prompt
        self.embedding = embedding
        self.response = response
        self.left = None
        self.right = None

class Cache(object):

    def __init__(self,
                 size,
                 embedding_engine="text-similarity-{}-001".format(DEFAULT_TEXT_SIMILARITY_MODEL),
                 threshold=DEFAULT_SIMILARITY_THRESHOLD,
                 llm_model="gpt-3.5-turbo"):
        self.lock = threading.Lock()
        self.threshold = threshold
        self.size = size
        self.embedding_engine = embedding_engine
        self.llm_model = llm_model
        self.cache = {}
        self.head = None
        self.tail = None

    def process_prompt(self, prompt):

        def _fetch_from_cache(cached_prompt, log=False):
            if log == True:
                logging.info("Fetching response for \"{}\" from cache".format(cached_prompt))

            node = self.cache[cached_prompt]
            if node == self.head:
                return node.response
            if node == self.tail:
                if node != self.head:
                    self.tail = node.right
                    self.tail.left = None
                    node.right = None
                    node.left = self.head
                    self.head.right = node
                    self.head = node
                return node.response
            node.left.right = node.right
            node.right.left = node.left
            node.right = None
            node.left = self.head
            self.head.right = node
            self.head = node
            return node.response

        def _find_closest_match(embedding):
            if len(self.cache) == 0:
                return None

            embedding_norm = np.linalg.norm(embedding)
            vals = list(self.cache.values())
            cache_embeddings = np.array([val.embedding for val in vals])
            numerator = np.dot(cache_embeddings, embedding)
            denominator = np.linalg.norm(cache_embeddings, axis=1) * embedding_norm
            similarities = numerator / denominator
            max_sim_idx = np.argmax(similarities)
            maximum_similarity = similarities[max_sim_idx]
            if maximum_similarity > self.threshold:
                sim_prompt = vals[max_sim_idx].prompt
                logging.info("Cached prompt \"{}\" has similarity {} with \"{}\"".format(
                    sim_prompt, maximum_similarity, prompt
                ))
                return _fetch_from_cache(sim_prompt)
            return None

        def _add_prompt_to_cache(embedding, response):
            logging.info("Adding prompt \"{}\" and response \"{}\" to cache".format(
                prompt, response
            ))
            new_node = Node(prompt, embedding, response)
            self.cache[prompt] = new_node
            if self.head is None:
                self.head = new_node
                self.tail = new_node
                return
            new_node.left = self.head
            self.head.right = new_node
            self.head = new_node
            if len(self.cache) > self.size:
                del_node = self.tail
                logging.info("Removing prompt \"{}\" and response \"{}\" from cache".format(
                    del_node.prompt, del_node.response
                ))
                del self.cache[del_node.prompt]
                self.tail = del_node.right
                self.tail.left = None

        with self.lock:
            logging.info("Processing prompt \"{}\"".format(prompt))
            if prompt in self.cache:
                return _fetch_from_cache(prompt, log=True)

            emb_response = openai.Embedding.create(input=prompt, engine=self.embedding_engine)
            embedding = np.array(emb_response['data'][0]['embedding'])
            closest_match = _find_closest_match(embedding)
            if closest_match is not None:
                return closest_match

            messages = [{'role': 'system', 'content': prompt}]
            logging.info("Calling LLM with prompt \"{}\"".format(prompt))
            chat = openai.ChatCompletion.create(model=self.llm_model, messages=messages)
            chat_response = chat.choices[0].message.content
            _add_prompt_to_cache(embedding, chat_response)
            logging.info("Returning response \"{}\" for prompt \"{}\"".format(chat_response, prompt))
            return chat_response

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Guardrails embedding API coding challenge')
    parser.add_argument('--api_key_file',
                        required=True,
                        type=argparse.FileType('r'),
                        help="Path of a file consisting only of an OpenAI API key")
    parser.add_argument('--text_similarity_model',
                        type=str,
                        default=DEFAULT_TEXT_SIMILARITY_MODEL,
                        choices=TEXT_SIMILARITY_MODELS)
    parser.add_argument('--text_similarity_threshold',
                        type=_similarity_threshold_type,
                        default=DEFAULT_SIMILARITY_THRESHOLD,
                        help="Similarity threshold above which cached values will be returned")
    parser.add_argument('--cache_size',
                        type=_cache_size_type,
                        default=DEFAULT_CACHE_SIZE,
                        help="Size of the cache (must be greater than 0)")
    args = parser.parse_args()

    openai.api_key = args.api_key_file.read().strip()
    embedding_engine = "text-similarity-{}-001".format(args.text_similarity_model)
    threshold = args.text_similarity_threshold
    cache_size = args.cache_size

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    config_dict = {
        'Embedding engine': embedding_engine,
        'Similarity threshold': threshold,
        'Cache size': cache_size,
        'API key file': args.api_key_file.name
    }
    logging.info('Starting API ({})'.format(str(config_dict)))

    app = Flask(__name__)
    cache = Cache(cache_size, embedding_engine, threshold)

    @app.route('/')
    def process_prompt():
        return cache.process_prompt(request.args.get('prompt'))

    app.run()