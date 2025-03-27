import time
import math
import re
import logging
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import networkx as nx
import nltk
import requests
from django.http import JsonResponse
from django.db.models import Count, F, Q
from nltk import word_tokenize
from nltk.corpus import stopwords
from django.db import transaction
from queue import Queue

# journal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gutendex_import.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gutendex_importer")


# for resuming import later
def save_import_state(state_data):
    """save import state to a JSON file,for resuming later"""
    with open('import_state.json', 'w') as f:
        json.dump(state_data, f)
    logger.info(f"Save import state：already treated {state_data['books_imported']} books")

def load_import_state():
    """load import state from a JSON file,for resuming import"""
    try:
        if os.path.exists('import_state.json'):
            with open('import_state.json', 'r') as f:
                state = json.load(f)
            logger.info(f"load import state：last time imported {state['books_imported']} books，final urls is {state['last_url']}")
            return state
        return None
    except Exception as e:
        logger.error(f"load import state failed：{str(e)}")
        return None


def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')



def preprocess_text(text):
    """preprocess text：lowercase,tokenize,remove stopwords and non-alphabetic characters"""
    download_nltk_resources()

    # lowercase and remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # tokenize
    tokens = word_tokenize(text)

    # remove stopwords and short words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]

    return filtered_tokens

def preprocess_gutenberg_text(text):
    """to process text from Project Gutenberg"""

    # try to extract the main content part
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*END*THE SMALL PRINT",
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
        "This eBook is for the use of anyone"
    ]

    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "End of the Project Gutenberg EBook",
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "End of Project Gutenberg"
    ]

    # we cut the text between the start and end markers
    main_content = text

    # find start marker
    start_pos = -1
    for marker in start_markers:
        pos = text.find(marker)
        if pos != -1:
            line_end = text.find('\n', pos)
            if line_end != -1:
                start_pos = line_end + 1
            break

    # find end marker
    end_pos = -1
    for marker in end_markers:
        pos = text.rfind(marker)
        if pos != -1:
            end_pos = pos
            break

    # cut the text
    if start_pos != -1 and end_pos != -1 and start_pos < end_pos:
        main_content = text[start_pos:end_pos]
    elif start_pos != -1:
        main_content = text[start_pos:]

    # clean
    main_content = re.sub(r'\r\n', '\n', main_content)
    main_content = re.sub(r'\n{3,}', '\n\n', main_content)

    return main_content

# we cut only the first 10k words for each book
def truncate_to_word_count(text, target_words=10000):
    words = text.split()
    if len(words) <= target_words:
        return text
    return ' '.join(words[:target_words])
