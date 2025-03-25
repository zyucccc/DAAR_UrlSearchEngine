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


from .models import Book, Term, TermDocumentIndex

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


##################################################################
#####                                                        #####
#####          Calculate TF-IDF for base de donnee           #####
#####                                                        #####
##################################################################
def calculate_tfidf(request):
    batch_size = int(request.GET.get('batch_size', '2000'))
    term_workers = int(request.GET.get('term_workers', '8'))
    update_workers = int(request.GET.get('update_workers', '8'))

    # boost factor for title and author
    title_boost = float(request.GET.get('title_boost', '1.5'))
    author_boost = float(request.GET.get('author_boost', '0.8'))

    return calculate_tfidf_weights_parallel(
        request,
        batch_size,
        term_workers,
        update_workers,
        title_boost,
        author_boost
    )

##################################################################
#####                                                        #####
#####    fetch 1664 documents(>10k mots) from Gutendex API   #####
#####                                                        #####
##################################################################
def fetch_books_from_gutendex(request):
    # if this is a continuation of a previous import or new import
    continue_import = request.GET.get('continue', 'false').lower() == 'true'
    reset_data = request.GET.get('reset_data', 'false').lower() == 'true'
    max_workers = int(request.GET.get('workers', '8'))

    target_books = int(request.GET.get('target', '1664'))
    min_words = int(request.GET.get('min_words', '10000'))
    skip_tfidf = request.GET.get('skip_tfidf', 'false').lower() == 'true'

    books_imported = 0
    imported_book_ids = set()
    base_url = "https://gutendex.com/books/"
    pages_processed = 0
    max_pages = int(request.GET.get('max_pages', '1000'))

    # if this is a resume import:
    if continue_import:
        state = load_import_state()
        if state:
            books_imported = state.get('books_imported', 0)
            imported_book_ids = set(state.get('imported_book_ids', []))
            base_url = state.get('last_url', base_url)
            pages_processed = state.get('pages_processed', 0)
            logger.info(f"keep importing from resume point，imported already {books_imported} books，resume at {base_url} page")

    # clean old base de donnee
    try:
        if reset_data:
            with transaction.atomic():
                logger.info("Delete all old BD...")
                # TermDocumentIndex.objects.all().delete()
                # Term.objects.all().delete()
                # Book.objects.all().delete()
                logger.info("Deleting finished")

                books_imported = 0
                imported_book_ids = set()
                base_url = "https://gutendex.com/books/"
                pages_processed = 0

        # keep get books from API pages
        while base_url and books_imported < target_books and pages_processed < max_pages:
            try:
                logger.info(f"API pages: {base_url}")
                response = requests.get(base_url, timeout=30)
                if response.status_code != 200:
                    logger.error(f"failed to fetch book: {response.status_code}")
                    time.sleep(5)
                    continue

                data = response.json()
                books_data = data["results"]
                pages_processed += 1

                logger.info(f"fetched {len(books_data)} books")

                # filter to get just english books
                english_books = [book for book in books_data
                                 if "en" in book.get("languages", []) and
                                 book.get("id") not in imported_book_ids]

                logger.info(f"after filtering: {len(english_books)} books")

                if not english_books:
                    logger.info("no available books for this api page,continue...")
                    base_url = data.get("next")
                    continue

                # parallel treat
                processed_books = []
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(process_single_book, book, min_words)
                               for book in english_books]

                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result:
                                processed_books.append(result)
                        except Exception as e:
                            logger.error(f"failed when trating book: {str(e)}")

                logger.info(f"The treatment of this batch finished，added {len(processed_books)} books")

                with transaction.atomic():
                    for book_data in processed_books:
                        if books_imported >= target_books:
                            break

                        new_book = Book.objects.create(
                            title=book_data["title"],
                            author=book_data["author"],
                            language=book_data["language"],
                            download_count=book_data["download_count"],
                            cover_url=book_data["cover_url"],
                            word_count=book_data["word_count"],
                            content=book_data["content"],
                        )

                        if process_book_terms(new_book, book_data["content"]):
                            books_imported += 1
                            imported_book_ids.add(book_data["book_id"])
                            logger.info(f"imported {books_imported}/{target_books} books: '{book_data['title'][:30]}...'")
                        else:
                            new_book.delete()
                            logger.warning(f"Book '{book_data['title'][:30]}...' failed，deleted")

                if pages_processed % 5 == 0 or books_imported % 50 == 0:
                    save_import_state({
                        'books_imported': books_imported,
                        'imported_book_ids': list(imported_book_ids),
                        'last_url': base_url,
                        'pages_processed': pages_processed
                    })

                # get next api page
                base_url = data.get("next")

                # if 1664 books imported,break
                if books_imported >= target_books:
                    logger.info(f"imported  {target_books} books，import finished")
                    break

                time.sleep(0.5)

            except requests.exceptions.RequestException as e:
                logger.error(f"request failed: {str(e)}")
                time.sleep(10)
            except Exception as e:
                logger.error(f"import failed: {str(e)}")
                save_import_state({
                    'books_imported': books_imported,
                    'imported_book_ids': list(imported_book_ids),
                    'last_url': base_url,
                    'pages_processed': pages_processed
                })
                time.sleep(5)

        # after importing,we can calculate tf-idf weight
        if books_imported > 0 and not skip_tfidf:
            logger.info("all book imported,start calculating tf-idf")
            calculate_tfidf_weights()
            logger.info("TF-IDF computation finished")
        elif skip_tfidf:
            logger.info("skip TF-IDF")

        save_import_state({
            'books_imported': books_imported,
            'imported_book_ids': list(imported_book_ids),
            'last_url': base_url,
            'pages_processed': pages_processed,
            'completed': books_imported >= target_books
        })

        return JsonResponse({
            "message": f"importer {books_imported} books，and calculated tf-idf",
            "books_count": books_imported,
            "terms_count": Term.objects.count(),
            "index_entries": TermDocumentIndex.objects.count(),
            "pages_processed": pages_processed,
            "completed": books_imported >= target_books
        }, safe=False)

    except Exception as e:
        logger.error(f"import failed: {str(e)}")
        save_import_state({
            'books_imported': books_imported,
            'imported_book_ids': list(imported_book_ids),
            'last_url': base_url,
            'pages_processed': pages_processed,
            'error': str(e)
        })
        return JsonResponse({"error": f"import failed: {str(e)}"}, status=500)

##################################################################
#####                                                        #####
#####            Recherche simple par mot-clé                #####
#####                                                        #####
##################################################################
def search_books(request):
    """
    search based on TF-IDF score,y a un threshold pour decider si on prends en compte le book
    """
    query = request.GET.get("q", "")

    # check if this is a regex search
    if is_regex_query(query):
        logger.info(f"this search is a regex search: {query}")
        return regex_search_books(request)


    min_score = float(request.GET.get("min_score", "0.0001"))

    if not query:
        return JsonResponse({"error": "pls fournir le mot cle"}, status=400)

    search_terms = preprocess_text(query)

    if not search_terms:
        return JsonResponse({"error": "non valide mot cle"}, status=400)

    # we filter valide mot-cle existant dans index table depuis le searching query
    matching_terms = Term.objects.filter(word__in=search_terms)

    if not matching_terms.exists():
        logger.info(f"didnt find '{query}'，back to search base")
        books = Book.objects.filter(Q(title__icontains=query) | Q(author__icontains=query))
        data = list(books.values('id', 'title', 'author', 'language', 'download_count', 'cover_url'))
        return JsonResponse(data, safe=False)

    # search in index_table,return all books which have higher score tf-idf than threshold
    matching_indices = TermDocumentIndex.objects.filter(
        term__in=matching_terms,
        tfidf__gt=min_score
    ).values('document', 'document__title', 'document__author', 'document__language',
             'document__download_count', 'document__cover_url', 'document__word_count',
             'term__word', 'tfidf')

    if not matching_indices:
        return JsonResponse({"error": f"didnt find result which have higher score than {min_score}"}, status=404)

    document_scores = {}
    document_details = {}

    for idx in matching_indices:
        doc_id = idx['document']
        tfidf_score = idx['tfidf']

        if doc_id not in document_scores:
            document_scores[doc_id] = 0
            document_details[doc_id] = {
                'id': doc_id,
                'title': idx['document__title'],
                'author': idx['document__author'],
                'language': idx['document__language'],
                'download_count': idx['document__download_count'],
                'cover_url': idx['document__cover_url'],
                'word_count': idx['document__word_count'],
                'matched_terms': [],
                'total_score': 0
            }

        document_scores[doc_id] += tfidf_score
        document_details[doc_id]['matched_terms'].append({
            'term': idx['term__word'],
            'score': tfidf_score
        })

    for doc_id in document_scores:
        document_details[doc_id]['total_score'] = document_scores[doc_id]

    sorted_results = sorted(
        document_details.values(),
        key=lambda x: x['total_score'],
        reverse=True
    )

    logger.info(f"Search '{query}' find {len(sorted_results)} results，TF-IDF threshold: {min_score}")

    return JsonResponse(sorted_results, safe=False)

##################################################################
#####                                                        #####
#####                     regex_search                       #####
#####                                                        #####
##################################################################
def is_regex_query(query):
    """check if search query contains special regex character"""
    regex_special_chars = r'[*+?^$\\.[\]{}()|/]'
    return bool(re.search(regex_special_chars, query))


def regex_search_books(request):

    regex_query = request.GET.get("q", "")
    min_score = float(request.GET.get("min_score", "0.0001"))

    if not regex_query:
        return JsonResponse({"error": "pls provide regex search query"}, status=400)

    try:
        # try to compile regex query
        regex_pattern = re.compile(regex_query, re.IGNORECASE)
    except re.error as e:
        return JsonResponse({"error": f"invalid regex query: {str(e)}"}, status=400)

    all_terms = Term.objects.values_list('id', 'word')

    matching_term_ids = []
    matched_words = []

    for term_id, word in all_terms:
        if regex_pattern.search(word):
            matching_term_ids.append(term_id)
            matched_words.append(word)

    if not matching_term_ids:
        return JsonResponse({"error": f"Regex query doesnt fit any '{regex_query}' terme"}, status=404)

    matching_indices = TermDocumentIndex.objects.filter(
        term_id__in=matching_term_ids,
        tfidf__gt=min_score
    ).values(
        'document', 'document__title', 'document__author', 'document__language',
        'document__download_count', 'document__cover_url', 'document__word_count',
        'term__word', 'tfidf'
    )

    if not matching_indices:
        return JsonResponse({"error": f"didnt find any docment that has higher score than {min_score}"}, status=404)

    document_scores = {}
    document_details = {}

    for idx in matching_indices:
        doc_id = idx['document']
        tfidf_score = idx['tfidf']
        term_word = idx['term__word']

        if doc_id not in document_scores:
            document_scores[doc_id] = 0
            document_details[doc_id] = {
                'id': doc_id,
                'title': idx['document__title'],
                'author': idx['document__author'],
                'language': idx['document__language'],
                'download_count': idx['document__download_count'],
                'cover_url': idx['document__cover_url'],
                'word_count': idx['document__word_count'],
                'matched_terms': {},
                'total_score': 0
            }

        document_scores[doc_id] += tfidf_score
        document_details[doc_id]['matched_terms'][term_word] = tfidf_score

    for doc_id in document_scores:
        document_details[doc_id]['total_score'] = round(document_scores[doc_id], 4)

    sorted_results = sorted(
        document_details.values(),
        key=lambda x: x['total_score'],
        reverse=True
    )

    stats = {
        'regex_query': regex_query,
        'matched_terms': matched_words,
        'result_count': len(sorted_results),
        'min_score': min_score
    }

    # return JsonResponse({
    #     'stats': stats,
    #     'results': sorted_results
    # }, safe=False)
    return JsonResponse(sorted_results,safe=False)

""" ---------------- Recherche avancée par expression régulière ----------------- """

# def regex_search_books(request):
#     regex = request.GET.get("regex", "")
#
#     if regex:
#         books = Book.objects.all()
#         matched_books = [book for book in books if re.search(regex, book.title, re.IGNORECASE)]
#         data = [{"title": b.title, "author": b.author} for b in matched_books]
#         return JsonResponse(data, safe=False)
#
#     return JsonResponse({"error": "No books found"}, status=404)

""" ----------------- Classement des livres avec PageRank -----------------"""

def rank_books(request):
    books = Book.objects.all()
    G = nx.Graph()

    for book in books:
        G.add_node(book.title, download_count=book.download_count)

    for book1 in books:
        for book2 in books:
            if book1.author == book2.author and book1.title != book2.title:
                G.add_edge(book1.title, book2.title)

    pagerank = nx.pagerank(G)
    ranked_books = sorted(books, key=lambda b: pagerank.get(b.title, 0), reverse=True)

    data = [{"title": b.title, "author": b.author, "score": pagerank.get(b.title, 0)} for b in ranked_books]
    return JsonResponse(data, safe=False)

""" ------------------ Suggestion basée sur la similarité de Jaccard -------------------- """

def suggest_books(request):
    books = Book.objects.all()
    G = nx.Graph()

    for book in books:
        G.add_node(book.title)

    for book1 in books:
        for book2 in books:
            if book1.author == book2.author and book1.title != book2.title:
                G.add_edge(book1.title, book2.title)

    suggestions = {}
    for book in books:
        neighbors = list(G.neighbors(book.title))
        suggestions[book.title] = neighbors

    return JsonResponse(suggestions, safe=False)
