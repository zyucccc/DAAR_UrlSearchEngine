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

# calculate tf-idf for 1 batch
def calculate_term_batch_tfidf(term_batch, total_docs, output_queue, batch_size=1000):
    """
    Parallel function to calculate TF-IDF weights for a batch of terms
    """
    start_time = time.time()
    index_entries_to_update = []

    term_ids = [term.id for term in term_batch]

    # get term words in lowercase
    term_words = {term.id: term.word.lower() for term in term_batch}

    index_data = list(TermDocumentIndex.objects.filter(term_id__in=term_ids).values('id', 'term_id', 'tf', 'document_id'))

    document_ids = [index['document_id'] for index in index_data]

    # doc info
    doc_info = {}
    for doc in Book.objects.filter(id__in=document_ids).values('id', 'title', 'author'):
        doc_info[doc['id']] = {
            'title': doc['title'].lower(),
            'author': doc['author'].lower() if doc['author'] else ''
        }

    # map term_id to Term object
    term_map = {term.id: term for term in term_batch}

    # if terme is in title or author,we will boost the tf-idf score
    TITLE_BOOST = 1.5
    AUTHOR_BOOST = 0.8

    # treat for each index entry
    for index in index_data:
        term_id = index['term_id']
        doc_id = index['document_id']

        term = term_map.get(term_id)
        if term and term.document_count > 0:
            term_word = term_words.get(term_id, '')
            idf = math.log(total_docs / term.document_count)

            # calculate tf-idf de base
            tfidf = index['tf'] * idf

            # boost if the term is in the title
            if doc_id in doc_info:
                doc = doc_info[doc_id]

                if term_word and term_word in doc['title']:
                    tfidf += TITLE_BOOST
                if term_word and term_word in doc['author']:
                    tfidf += AUTHOR_BOOST

            index_entries_to_update.append({
                'id': index['id'],
                'tfidf': tfidf
            })

    if index_entries_to_update:
        output_queue.put(index_entries_to_update)

    duration = time.time() - start_time
    return len(index_entries_to_update), duration


def calculate_tfidf_weights_parallel(request=None, batch_size=2000, term_workers=8, update_workers=4,
                                     title_boost=1.5, author_boost=0.8):
    """
    Parallel function to calculate TF-IDF weights for all terms
    """
    start_time = time.time()

    # all documents
    total_docs = Book.objects.count()

    if total_docs == 0:
        logger.warning("No docs dispo for TF-IDF")
        if request:
            return JsonResponse({"error": "No docs dispo for TF-IDF"}, status=400)
        return

    logger.info("Optimize Term Table，Updating document_count...")
    try:
        with transaction.atomic():
            term_counts = TermDocumentIndex.objects.values('term').annotate(doc_count=Count('document', distinct=True))

            update_batches = []
            for item in term_counts:
                update_batches.append({
                    'id': item['term'],
                    'document_count': item['doc_count']
                })

            # update tf-idf by batch
            for i in range(0, len(update_batches), batch_size):
                batch = update_batches[i:i+batch_size]
                objs_to_update = []
                for item in batch:
                    term = Term(id=item['id'])
                    term.document_count = item['document_count']
                    objs_to_update.append(term)

                Term.objects.bulk_update(objs_to_update, ['document_count'])
    except Exception as e:
        logger.error(f"failed optimizing index table: {str(e)}")
        if request:
            return JsonResponse({"error": f"failed optimizing index table: {str(e)}"}, status=500)
        raise e

    logger.info(f"Boost factor for title: {title_boost}, Boost factor for author: {author_boost}")

    # terms in total
    total_terms = Term.objects.count()
    logger.info(f"Starting calculating {total_terms} tf-idf weights ，using {term_workers} workers")

    # queue
    result_queue = Queue()

    # treatment by batch
    all_term_batches = []
    for i in range(0, total_terms, batch_size):
        term_batch = list(Term.objects.all()[i:i+batch_size])
        all_term_batches.append(term_batch)

    # process parallel
    global TITLE_BOOST, AUTHOR_BOOST
    TITLE_BOOST = title_boost
    AUTHOR_BOOST = author_boost

    process_batch = partial(calculate_term_batch_tfidf, total_docs=total_docs, output_queue=result_queue, batch_size=batch_size)

    # calculate tf-idf for each batch of terms
    processed_terms = 0
    total_entries = 0

    with ThreadPoolExecutor(max_workers=term_workers) as executor:
        futures = [executor.submit(process_batch, term_batch) for term_batch in all_term_batches]
        for future in as_completed(futures):
            try:
                entries_count, duration = future.result()
                processed_terms += len(all_term_batches[futures.index(future)])
                total_entries += entries_count
                logger.info(f"Treated {processed_terms}/{total_terms} Terms ({processed_terms/total_terms*100:.1f}%)，generated {entries_count} index entrys，duration {duration:.2f}second")
            except Exception as e:
                logger.error(f"failed when treat terms: {str(e)}")

    # collect all entries to update
    all_entries_to_update = []
    while not result_queue.empty():
        entries = result_queue.get()
        all_entries_to_update.extend(entries)

    logger.info(f"Starting updating {len(all_entries_to_update)} index-entris TFIDF value")

    logger.info("reset all old tf-idf as 0...")
    with transaction.atomic():
        TermDocumentIndex.objects.update(tfidf=0.0)

    # update by batch
    update_batch_size = batch_size * 2
    update_start_time = time.time()

    for i in range(0, len(all_entries_to_update), update_batch_size):
        batch = all_entries_to_update[i:i+update_batch_size]

        objs_to_update = []
        for item in batch:
            idx = TermDocumentIndex(id=item['id'])
            idx.tfidf = item['tfidf']
            objs_to_update.append(idx)

        with transaction.atomic():
            TermDocumentIndex.objects.bulk_update(objs_to_update, ['tfidf'])

        progress = min((i + update_batch_size) / len(all_entries_to_update), 1.0) * 100
        logger.info(f"Updated {progress:.1f}% tf-idf")

    update_duration = time.time() - update_start_time
    total_duration = time.time() - start_time

    logger.info(f"TF-IDF computation finished！Treated {total_terms} Terms，updated {len(all_entries_to_update)} index entris")
    logger.info(f"computation duration: {update_start_time - start_time:.2f}second，update duration: {update_duration:.2f}second")
    logger.info(f"duration in total: {total_duration:.2f}秒")

    if request:
        return JsonResponse({
            "message": "tf-idf calculated finished",
            "terms_processed": total_terms,
            "entries_updated": len(all_entries_to_update),
            "title_boost": title_boost,
            "author_boost": author_boost,
            "total_duration_seconds": total_duration
        })

    return total_terms, len(all_entries_to_update), total_duration