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
from .processing_text_utils import preprocess_text, preprocess_gutenberg_text, truncate_to_word_count,load_import_state, save_import_state
from ..models  import Book, Term, TermDocumentIndex, DocumentSimilarity, DocumentCentrality

logger = logging.getLogger("gutendex_importer")

def process_single_book(book_data, min_words=10000):
    """Parallel processing of a single book"""
    try:
        # get the content URL
        text_url = None
        for format_key in ["text/plain", "text/plain; charset=utf-8", "text/plain; charset=us-ascii"]:
            if format_key in book_data.get("formats", {}):
                text_url = book_data["formats"][format_key]
                break

        if not text_url:
            return None

        # try to get the text content
        text_response = requests.get(text_url, timeout=20)
        if text_response.status_code != 200:
            return None

        # get the text content
        try:
            book_text = text_response.text
        except UnicodeDecodeError:
            book_text = text_response.content.decode('utf-8', errors='ignore')

        # treat the text content with processing of Project Gutenberg
        book_text = preprocess_gutenberg_text(book_text)

        # clean
        book_text = re.sub(r'\s+', ' ', book_text).strip()

        # count the number of words
        words = re.findall(r'\b[a-zA-Z]+\b', book_text)
        word_count = len(words)

        # check if it has enough words(10k)
        if word_count >= min_words:
            # cut the first 10k words
            truncated_text = truncate_to_word_count(book_text, 10000)

            authors = ", ".join([a["name"] for a in book_data.get("authors", [])])
            languages = ", ".join(book_data.get("languages", []))
            download_count = book_data.get("download_count", 0)
            cover_url = book_data["formats"].get("image/jpeg", "")

            # return intermediate result of a book,we create the book record later
            return {
                "title": book_data["title"],
                "author": authors,
                "language": languages,
                "download_count": download_count,
                "cover_url": cover_url,
                "word_count": min(word_count, 10000),
                "content": truncated_text,
                "book_id": book_data.get("id", 0)
            }

        return None

    except Exception as e:
        logger.error(f"treat book {book_data.get('title', 'unknown')} failed: {str(e)}")
        return None


# process book terms with batch processing (parallel)
def process_book_terms(book, text, batch_size=1000):
    tokens = preprocess_text(text)

    if len(tokens) == 0:
        logger.warning(f"book '{book.title[:30]}...' is null after preprocessing")
        return False

    # TF-Calcul
    term_freq = Counter(tokens)

    # we divide the termes into 2 groups: existing and new
    unique_terms = list(term_freq.keys())
    existing_terms = {term.word: term for term in Term.objects.filter(word__in=unique_terms)}

    # for termes that do not exist, we create them
    new_terms = []
    for term in unique_terms:
        if term not in existing_terms:
            new_term = Term(word=term, document_count=1)
            new_terms.append(new_term)
            existing_terms[term] = new_term

    # Parallel creation of new terms
    if new_terms:
        Term.objects.bulk_create(new_terms)

    # update document_count for existing terms
    terms_to_update = []
    for term_word in existing_terms:
        term = existing_terms[term_word]
        if hasattr(term, 'id') and term.id is not None:  #existing term
            term.document_count += 1
            terms_to_update.append(term)

    # Parallel update Term's document_count
    if terms_to_update:
        Term.objects.bulk_update(terms_to_update, ['document_count'])

    # Parallel create TermDocumentIndex entries
    index_entries = []
    for term_word, count in term_freq.items():
        tf = count / len(tokens)
        index_entries.append(TermDocumentIndex(
            term=existing_terms[term_word],
            document=book,
            tf=tf,
            tfidf=0.0  # for tf-idf,we will calculate it later
        ))


    for i in range(0, len(index_entries), batch_size):
        batch = index_entries[i:i+batch_size]
        TermDocumentIndex.objects.bulk_create(batch)

    return True
