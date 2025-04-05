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

from .utils.processing_text_utils import preprocess_text, preprocess_gutenberg_text, truncate_to_word_count,load_import_state, save_import_state
from .utils.fetch_books_utils import process_single_book, process_book_terms
from .utils.computation_tfidf_utils import calculate_tfidf_weights_parallel
from .models import Book, Term, TermDocumentIndex, DocumentSimilarity, DocumentCentrality

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
#####    fetch 1664 documents(~10k mots) from Gutendex API   #####
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
            # calculate_tfidf()
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


    min_score = float(request.GET.get("min_score", "0.001"))

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
             'term__word','document__content', 'tfidf')

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
                'content': idx['document__content'],
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

    # sorted_results = sorted(
    #     document_details.values(),
    #     key=lambda x: x['total_score'],
    #     reverse=True
    # )
    # Trier les résultats par classement
    logger.info(f"Start sorting results by TF-IDF score + centrality,defaut setting: Avg score for centrality + 0.3 weight")
    document_details_list = list(document_details.values())
    sorted_results = classement(document_details_list)

    logger.info(f"Search '{query}' find {len(sorted_results)} results，TF-IDF threshold: {min_score}")
    response_data = {
        "results": sorted_results,
    }

    if sorted_results:
        logger.info(f"generating '{query}' for recommendations")
        recommendations = get_document_suggestions(
            sorted_results
        )

        if recommendations:
            response_data["suggestions"] = recommendations
            logger.info(f"for '{query}' generated  {len(recommendations)} recommendations")

    return JsonResponse(response_data, safe=False)

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
        'term__word', 'document__content','tfidf'
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
                'content': idx['document__content'],
                'matched_terms': {},
                'total_score': 0
            }

        document_scores[doc_id] += tfidf_score
        document_details[doc_id]['matched_terms'][term_word] = tfidf_score

    for doc_id in document_scores:
        document_details[doc_id]['total_score'] = round(document_scores[doc_id], 4)

    # sorted_results = sorted(
    #     document_details.values(),
    #     key=lambda x: x['total_score'],
    #     reverse=True
    # )
    # Trier les résultats par classement
    logger.info(f"Start sorting results by TF-IDF score + centrality,defaut setting: Avg score for centrality + 0.3 weight")
    document_details_list = list(document_details.values())
    sorted_results = classement(document_details_list)

    response_data = {
        "results": sorted_results,
    }

    if sorted_results:
        logger.info(f"generating '{regex_query}' for recommendations")
        recommendations = get_document_suggestions(
            sorted_results
        )

        if recommendations:
            response_data["suggestions"] = recommendations
            logger.info(f"for '{regex_query}' generated {len(recommendations)} recommendations")

    return JsonResponse(response_data,safe=False)


##################################################################
#####                                                        #####
#####             calculate Jaccard distance                 #####
#####                                                        #####
##################################################################

def calculate_document_similarities(request=None, batch_size=100, similarity_threshold=0.1, max_workers=8):
    """
    calculate Jaccard similarity between all documents and store the results in the database.
    we set a min threshold for similarity to avoid storing too many results.

    - batch_size
    - similarity_threshold
    - max_workers: le max number of workers to use for parallel processing
    """
    start_time = time.time()

    # get all docs
    documents = list(Book.objects.all())
    total_docs = len(documents)

    if total_docs == 0:
        logger.warning("no docs dispos for Jaccard similarity")
        if request:
            return JsonResponse({"error": "no docs dispos for Jaccard similarity"}, status=400)
        return

    # get ensemble de termes pour chaque document
    doc_term_sets = {}

    logger.info(f"start get ensemble de termes pour {total_docs} docs")

    # get all term-doc indices
    term_doc_indices = list(TermDocumentIndex.objects.all().values('document_id', 'term_id'))

    # trie les termes par docs
    for idx in term_doc_indices:
        doc_id = idx['document_id']
        term_id = idx['term_id']

        if doc_id not in doc_term_sets:
            doc_term_sets[doc_id] = set()

        doc_term_sets[doc_id].add(term_id)

    # pre-load all TF-IDF values to memory
    logger.info("starting to pre-load all TF-IDF values to memory")
    term_doc_weights = {}

    # get all term-document pairs and their TF-IDF values
    all_term_document_pairs = TermDocumentIndex.objects.all().values('term_id', 'document_id', 'tfidf')

    logger.info(f"founded{all_term_document_pairs.count()}documents and {len(doc_term_sets)} terms")

    batch_size_load = 10000
    total_loaded = 0

    for i in range(0, all_term_document_pairs.count(), batch_size_load):
        batch = all_term_document_pairs[i:i+batch_size_load]

        for record in batch:
            term_id = record['term_id']
            doc_id = record['document_id']
            tfidf = record['tfidf']

            if term_id not in term_doc_weights:
                term_doc_weights[term_id] = {}

            term_doc_weights[term_id][doc_id] = tfidf

        total_loaded += len(batch)
        logger.info(f"loaded {total_loaded}/{all_term_document_pairs.count()} TF-IDF values ({(total_loaded/all_term_document_pairs.count()*100):.1f}%)")

    logger.info(f"succes loaded {len(term_doc_weights)} TF-IDF")

    # delete acutuel similarity
    DocumentSimilarity.objects.all().delete()

    # calculer similarity parallelement
    total_pairs = (total_docs * (total_docs - 1)) // 2  # on garde des entrie unique,on compte que 1 fois pour 2 doc

    logger.info(f"start calculating {total_docs} docs's jaccard similarity（{total_pairs} pairs）using {max_workers} processus")

    # task queue
    from queue import Queue
    result_queue = Queue()

    # calculate jaccard similarity
    def process_document_batch(batch_docs, doc_term_sets, term_doc_weights, similarity_threshold, result_queue):
        batch_similarities = []
        batch_processed = 0

        for doc1_idx, doc2_idx in batch_docs:
            doc1_id = documents[doc1_idx].id
            doc2_id = documents[doc2_idx].id

            if doc1_id not in doc_term_sets or doc2_id not in doc_term_sets:
                continue

            # get terms ensemble for doc
            terms1 = doc_term_sets[doc1_id]
            terms2 = doc_term_sets[doc2_id]

            # find common terms
            common_terms = terms1.intersection(terms2)

            if not common_terms:
                batch_processed += 1
                continue

            # get tf-idf for common terms
            numerator = 0.0
            denominator = 0.0

            # indices = TermDocumentIndex.objects.filter(
            #     term_id__in=common_terms,
            #     document_id__in=[doc1_id, doc2_id]
            # ).values('term_id', 'document_id', 'tfidf')

            # dictionary to store term weights
            # term_weights = {}
            # for idx in indices:
            #     term_id = idx['term_id']
            #     doc_id = idx['document_id']
            #     tfidf = idx['tfidf']
            #
            #     if term_id not in term_weights:
            #         term_weights[term_id] = {}
            #
            #     term_weights[term_id][doc_id] = tfidf

            # calculer le numérateur et le dénominateur
            for term_id in common_terms:
                if term_id in term_doc_weights and doc1_id in term_doc_weights[term_id] and doc2_id in term_doc_weights[term_id]:
                    k1 = term_doc_weights[term_id][doc1_id]
                    k2 = term_doc_weights[term_id][doc2_id]

                    max_k = max(k1, k2)
                    min_k = min(k1, k2)

                    numerator += (max_k - min_k)
                    denominator += max_k

            # jaccard distance
            if denominator > 0:
                jaccard_distance = numerator / denominator
                jaccard_similarity = 1.0 - jaccard_distance  # similarity = 1 - distance

                if jaccard_similarity >= similarity_threshold:
                    batch_similarities.append(
                        DocumentSimilarity(
                            document1_id=doc1_id,
                            document2_id=doc2_id,
                            jaccard_similarity=jaccard_similarity
                        )
                    )

            # calculate jaccard: intersection / union
            # intersection = len(terms1.intersection(terms2))
            # union = len(terms1.union(terms2))

            # /0
            # if union == 0:
            #     continue
            #
            # similarity = intersection / union

            # on garde le score quand le score est plus grand que threshold
            # if similarity >= similarity_threshold:
            #     batch_similarities.append(
            #         DocumentSimilarity(
            #             document1_id=doc1_id,
            #             document2_id=doc2_id,
            #             jaccard_similarity=similarity
            #         )
            #     )

            batch_processed += 1

        result_queue.put(batch_similarities)
        return batch_processed

    # traiter parallelement pour chaque pair de document
    doc_pairs = []
    for i in range(total_docs):
        for j in range(i + 1, total_docs):
            doc_pairs.append((i, j))

    pairs_per_worker = len(doc_pairs) // max_workers + 1

    # distribuer les taches
    batches = []
    for i in range(0, len(doc_pairs), pairs_per_worker):
        batches.append(doc_pairs[i:i+pairs_per_worker])

    processed_pairs = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for batch in batches:
            future = executor.submit(
                process_document_batch,
                batch,
                doc_term_sets,
                term_doc_weights,
                similarity_threshold,
                result_queue
            )
            futures.append(future)

        # wait tasks
        for future in as_completed(futures):
            try:
                batch_processed = future.result()
                processed_pairs += batch_processed
                # log: avancement
                progress = (processed_pairs / total_pairs) * 100
                logger.info(f"avancement: {progress:.1f}%, treated {processed_pairs}/{total_pairs} pairs")
            except Exception as e:
                logger.error(f"failed when calculate the jaccard similarity: {str(e)}")

    all_similarities = []
    while not result_queue.empty():
        batch_similarities = result_queue.get()
        all_similarities.extend(batch_similarities)

    logger.info(f"start saving {len(all_similarities)} similarity donnees")

    for i in range(0, len(all_similarities), batch_size):
        batch = all_similarities[i:i+batch_size]
        with transaction.atomic():
            DocumentSimilarity.objects.bulk_create(batch)

        # log avancement
        progress = min((i + batch_size) / len(all_similarities), 1.0) * 100
        logger.info(f"saving avancement: {progress:.1f}%")

    total_similarities = DocumentSimilarity.objects.count()
    duration = time.time() - start_time

    logger.info(f"Jaccard similarity computation finished。created {total_similarities} entries。")
    logger.info(f"computation time: {duration:.2f} seconds")

    if request:
        return JsonResponse({
            "message": "Jaccard similarity computation succes",
            "documents_processed": total_docs,
            "total_similarities": total_similarities,
            "similarity_threshold": similarity_threshold,
            "max_workers": max_workers,
            "total_duration_seconds": duration
        })

    return total_docs, total_similarities, duration


##################################################################
#####                                                        #####
#####   calculate centralite(closness,betweeness,pagerank)   #####
#####                                                        #####
##################################################################

def calculate_centrality_scores(request=None, centrality_type='closeness', min_similarity=0.5, max_workers=8):
    """
    Based on graph Jaccard (sommet = doc, arete = distance jaccard)

    - centrality_type: 'closeness', 'betweenness', 或or'pagerank'
    - min_similarity: threshold of distance
    - max_workers: parallele processus
    """
    start_time = time.time()

    # creer centralite non-oriente graphe
    G = nx.Graph()

    # add sommets
    for book in Book.objects.all():
        G.add_node(book.id, title=book.title, author=book.author)

    # add aretes
    similarities = DocumentSimilarity.objects.filter(jaccard_similarity__gte=min_similarity)
    edge_count = 0

    for sim in similarities:
        # le poids ici est l'inverse du Jaccard distance,car le plus similaire,la distance plus courte
        # poids = 1/jaccard_similarity
        # weight = 1.0 / sim.jaccard_similarity if sim.jaccard_similarity > 0 else float('inf')
        # poids = 1 - jaccard_similarity
        weight = 1.0 - sim.jaccard_similarity if sim.jaccard_similarity < 1 else 0.0
        G.add_edge(sim.document1_id, sim.document2_id, weight=weight, similarity=sim.jaccard_similarity)
        edge_count += 1

    logger.info(f"created {G.number_of_nodes()} sommets et {edge_count} aretes dans le graphe")

    # calculate centralite based on le type (closeness,pagerank,betweeness)
    centrality_scores = {}

    try:
        if centrality_type == 'closeness':
            # closeness
            logger.info(f"start calculating closeness，using {max_workers} processus...")

            # parallele computation
            def calculate_node_closeness(node, graph):
                try:
                    closeness = nx.closeness_centrality(graph, u=node, distance='weight')
                    return node, closeness
                except Exception as e:
                    logger.error(f"failed {node} computation of closeness: {str(e)}")
                    return node, 0.0

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for node in G.nodes():
                    futures.append(executor.submit(calculate_node_closeness, node, G))

                for future in as_completed(futures):
                    try:
                        node, score = future.result()
                        centrality_scores[node] = score
                    except Exception as e:
                        logger.error(f"failed computation of closeness: {str(e)}")

        elif centrality_type == 'betweenness':
            # betweenness
            logger.info(f"computation betweenness，using {max_workers} processus...")
            try:
                centrality_scores = nx.betweenness_centrality(G, weight='weight', k=None, normalized=True,
                                                              endpoints=False, seed=None, parallel=True,
                                                              n_jobs=max_workers)
            except TypeError:
                logger.warning("computation parallel not supported，back to sequential")
                centrality_scores = nx.betweenness_centrality(G, weight='weight')

        elif centrality_type == 'pagerank':
            # PageRank
            logger.info("computation PageRank...")
            # pour pagerank,on utilise similarity mais pas distance
            centrality_scores = nx.pagerank(G)
        else:
            logger.error(f"type de centralite unknown: {centrality_type}")
            if request:
                return JsonResponse({"error": f"type de centralite unknown: {centrality_type}"}, status=400)
            return
    except Exception as e:
        logger.error(f"failed computation of centralite: {str(e)}")
        if request:
            return JsonResponse({"error": f"failed computation of centralite: {str(e)}"}, status=500)
        return

    # delete actual document centralite donnees
    DocumentCentrality.objects.filter(centrality_type=centrality_type).delete()

    # save donnee
    centrality_records = []
    for book_id, score in centrality_scores.items():
        centrality_records.append(
            DocumentCentrality(
                document_id=book_id,
                centrality_type=centrality_type,
                score=score
            )
        )

    with transaction.atomic():
        DocumentCentrality.objects.bulk_create(centrality_records)

    duration = time.time() - start_time

    logger.info(f"{centrality_type.capitalize()} computation finished in {duration:.2f} second")

    if request:
        return JsonResponse({
            "message": f"{centrality_type.capitalize()} computation finished",
            "documents_processed": len(centrality_scores),
            "centrality_type": centrality_type,
            "min_similarity": min_similarity,
            "max_workers": max_workers,
            "total_duration_seconds": duration
        })

    return len(centrality_scores), duration

##################################################################
#####                                                        #####
#####                     Classement                         #####
#####                                                        #####
##################################################################
def classement(search_results, centrality_type='all', centrality_weight=0.3):
    """
    classement des livres en fonction de la centralité et du score TF-IDF
    Normalement, on normalise le score TF-IDF et le score de centralité entre 0 et 1
    et apres on combine les deux scores en utilisant un poids(par defaut, 0,7pour TF-IDF et 0,3 pour centralité)
    - search_results
    - centrality_type:
       - 'best': by priority, closeness > betweenness > pagerank
       - 'closeness', 'betweenness', 'pagerank': we use the specified centrality type
       - 'all': AVG of all centrality types
    - centrality_weight: weight for centrality score, default 0.3
    """
    if not search_results:
        return []

    doc_ids = [doc['id'] for doc in search_results]
    # get total td-idf score
    document_scores = {doc['id']: doc.get('total_score', 0) for doc in search_results}

    # get all available centrality types
    available_centrality_types = list(DocumentCentrality.objects.filter(
        document_id__in=doc_ids
    ).values_list('centrality_type', flat=True).distinct())

    if not available_centrality_types:
        logger.warning("not found any centrality type")
        return search_results

    # moyen1: by priority, closeness > betweenness > pagerank
    if centrality_type == 'best':
        if 'closeness' in available_centrality_types:
            centrality_type = 'closeness'
        elif 'betweenness' in available_centrality_types:
            centrality_type = 'betweenness'
        elif 'pagerank' in available_centrality_types:
            centrality_type = 'pagerank'
        else:
            centrality_type = available_centrality_types[0]

        logger.info(f"we pick the centrality by priority,closeness > betweenness > pagerank: {centrality_type}")

    # moyen2: on calculate the avg of all centrality types
    if centrality_type == 'all':
        # get all centrality scores
        centrality_records = DocumentCentrality.objects.filter(
            document_id__in=doc_ids
        ).values('document_id', 'centrality_type', 'score')

        centrality_scores = {}
        for record in centrality_records:
            doc_id = record['document_id']
            c_type = record['centrality_type']
            score = record['score']

            if doc_id not in centrality_scores:
                centrality_scores[doc_id] = {}

            centrality_scores[doc_id][c_type] = score
    else:
        centrality_records = DocumentCentrality.objects.filter(
            document_id__in=doc_ids,
            centrality_type=centrality_type
        ).values('document_id', 'score')

        centrality_scores = {record['document_id']: record['score'] for record in centrality_records}

    # Normalise TF-IDF A [0,1]
    max_tfidf = max(document_scores.values()) if document_scores else 1.0
    normalized_tfidf = {
        doc_id: score / max_tfidf if max_tfidf > 0 else 0.0
        for doc_id, score in document_scores.items()
    }

    combined_scores = {}

    if centrality_type == 'all':
        # for each type of centrality, we calculate the average score
        for doc_id in doc_ids:
            if doc_id not in centrality_scores:
                combined_scores[doc_id] = normalized_tfidf.get(doc_id, 0)
                continue

            # Normalise centrality scores
            normalized_centralities = {}
            for c_type in available_centrality_types:
                if c_type not in centrality_scores[doc_id]:
                    continue

                c_scores = [centrality_scores[d_id].get(c_type, 0) for d_id in centrality_scores if c_type in centrality_scores[d_id]]
                if not c_scores:
                    continue

                max_c = max(c_scores)
                min_c = min(c_scores)
                range_c = max_c - min_c

                if range_c > 0:
                    normalized_centralities[c_type] = (centrality_scores[doc_id][c_type] - min_c) / range_c
                else:
                    normalized_centralities[c_type] = 0.5

            # AVG of all centrality types
            if normalized_centralities:
                avg_centrality = sum(normalized_centralities.values()) / len(normalized_centralities)

                tfidf_norm = normalized_tfidf.get(doc_id, 0.0)
                weight = _calculate_adaptive_weight(tfidf_norm, centrality_weight)
                logger.info(f"Classment: for document: {Book.objects.get(id=doc_id).title}")
                logger.info("Classment: After normalising,tfidf_norm: {},AVG_centrality_norm: {},weight: {}".format(tfidf_norm, avg_centrality, weight))
                combined_scores[doc_id] = (1 - weight) * tfidf_norm + weight * avg_centrality
            else:
                combined_scores[doc_id] = normalized_tfidf.get(doc_id, 0)
    else:
        # for one type of centrality, we use the score directly(Normalise)
        max_centrality = max(centrality_scores.values()) if centrality_scores else 1.0
        min_centrality = min(centrality_scores.values()) if centrality_scores else 0.0
        range_centrality = max_centrality - min_centrality

        normalized_centrality = {}
        if range_centrality > 0:
            normalized_centrality = {
                doc_id: (score - min_centrality) / range_centrality
                for doc_id, score in centrality_scores.items()
            }
        else:
            normalized_centrality = {doc_id: 0.5 for doc_id in centrality_scores}

        # calculate the combined score by multiplying the normalized scores and the adaptive weight
        for doc_id in doc_ids:
            tfidf_norm = normalized_tfidf.get(doc_id, 0.0)
            centrality_norm = normalized_centrality.get(doc_id, 0.0)

            weight = _calculate_adaptive_weight(tfidf_norm, centrality_weight)

            combined_scores[doc_id] = (1 - weight) * tfidf_norm + weight * centrality_norm

    # update search results with combined scores
    for doc in search_results:
        doc_id = doc['id']
        doc['tfidf_score'] = document_scores.get(doc_id, 0)

        if centrality_type == 'all':
            # "All" centrality types
            if doc_id in centrality_scores:
                doc['centrality_scores'] = centrality_scores[doc_id]
            else:
                doc['centrality_scores'] = {}
        else:
            doc['centrality_type'] = centrality_type
            doc['centrality_score'] = centrality_scores.get(doc_id, 0)

        doc['combined_score'] = combined_scores.get(doc_id, 0)

    # Trier par combined_score
    sorted_results = sorted(search_results, key=lambda x: x['combined_score'], reverse=True)

    return sorted_results


def _calculate_adaptive_weight(tfidf_norm, base_weight):
    """
    we calculate the adaptive weight based on the TF-IDF score
    if we have a high TF-IDF score, we reduce the centrality weight (because in this case, the terme is in Title or Author,tf-idf is more important)

    if we have a low TF-IDF score, we increase the centrality weight (because in this case, the terme is not in Title or Author,
    tf-idf is less important,but centrality is more important)

    - tfidf_norm: tf-idf normalised (0-1)
    - base_weight

    """
    # for high TF-IDF score documents, reduce the centrality weight
    if tfidf_norm > 0.7:
        return base_weight * 0.5
    elif tfidf_norm > 0.4:
        return base_weight
    # for low TF-IDF score documents, increase the centrality weight
    else:
        return min(base_weight * 1.2, 0.5)

##################################################################
#####                                                        #####
##### Recommendation basée sur la similarité de Jaccard      #####
#####                                                        #####
##################################################################
def get_document_suggestions(search_results, top_k=3, max_per_doc=2, max_suggestions=5, min_similarity=0.15):
    """
    pour les k meilleurs résultats de recherche, nous recommandons les documents les plus similaires
    - search_results
    - top_k: les k meilleurs résultats de recherche
    - max_per_doc: nb de voisins les plus proches à recommander pour chaque document
    - max_suggestions:
    - min_similarity:
    """
    if not search_results or len(search_results) == 0:
        return []

    # get les ID des k le plus meilleurs documents de recherche
    top_doc_ids = [doc['id'] for doc in search_results[:min(top_k, len(search_results))]]

    if not top_doc_ids:
        return []

    logger.info(f"start generating {len(top_doc_ids)}recommendations for: {top_doc_ids}")

    # les voisins de Jaccard
    similar_docs_query = DocumentSimilarity.objects.filter(
        (Q(document1_id__in=top_doc_ids) | Q(document2_id__in=top_doc_ids)),
        jaccard_similarity__gte=min_similarity
    ).values('document1_id', 'document2_id', 'jaccard_similarity')

    # Structure: {source_id {recommendation_doc_id: {similarity, source_doc_id}}}
    recommendations_by_source = {doc_id: {} for doc_id in top_doc_ids}

    for relation in similar_docs_query:
        doc1_id = relation['document1_id']
        doc2_id = relation['document2_id']
        similarity = relation['jaccard_similarity']

        # check which one is source and which one is recommendation
        if doc1_id in top_doc_ids and doc2_id not in top_doc_ids:
            # doc1 source
            if doc2_id not in recommendations_by_source[doc1_id] or similarity > recommendations_by_source[doc1_id][doc2_id]['similarity']:
                recommendations_by_source[doc1_id][doc2_id] = {
                    'similarity': similarity,
                    'source_doc_id': doc1_id
                }
        elif doc2_id in top_doc_ids and doc1_id not in top_doc_ids:
            # doc2 source
            if doc1_id not in recommendations_by_source[doc2_id] or similarity > recommendations_by_source[doc2_id][doc1_id]['similarity']:
                recommendations_by_source[doc2_id][doc1_id] = {
                    'similarity': similarity,
                    'source_doc_id': doc2_id
                }

    # check if we have at least 1 recommendations
    has_recommendations = any(len(recs) > 0 for recs in recommendations_by_source.values())
    if not has_recommendations:
        logger.info(f"not found higher than {min_similarity} recommendations")
        return []

    # delete duplicates in search results
    result_doc_ids = {doc['id'] for doc in search_results}

    all_recommendations = {}

    # pick the top max_per_doc recommendations for each source
    for source_id, recommendations in recommendations_by_source.items():
        filtered_recommendations = {rec_id: data for rec_id, data in recommendations.items()
                                    if rec_id not in result_doc_ids}

        # trier par similarity
        sorted_recommendations = sorted(
            filtered_recommendations.items(),
            key=lambda x: x[1]['similarity'],
            reverse=True
        )[:max_per_doc]

        for rec_id, data in sorted_recommendations:
            if rec_id not in all_recommendations or data['similarity'] > all_recommendations[rec_id]['similarity']:
                all_recommendations[rec_id] = data

    # si le nombre total de recommandations dépasse max_suggestions, trier par similarité
    if len(all_recommendations) > max_suggestions:
        sorted_recommendations = sorted(
            all_recommendations.items(),
            key=lambda x: x[1]['similarity'],
            reverse=True
        )[:max_suggestions]
        all_recommendations = {rec_id: data for rec_id, data in sorted_recommendations}

    if not all_recommendations:
        return []

    # details des recommandations
    recommendation_ids = list(all_recommendations.keys())
    recommendations = list(Book.objects.filter(id__in=recommendation_ids).values(
        'id', 'title', 'author', 'language', 'download_count', 'cover_url', 'word_count','content'
    ))

    for recommendation in recommendations:
        doc_id = recommendation['id']
        similarity_info = all_recommendations[doc_id]
        recommendation['similarity_score'] = similarity_info['similarity']

        # pick source document info
        source_doc_id = similarity_info['source_doc_id']
        source_doc = next((doc for doc in search_results if doc['id'] == source_doc_id), None)
        if source_doc:
            recommendation['recommended_based_on'] = {
                'id': source_doc_id,
                'title': source_doc['title']
            }

    # trier par similarite
    recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)

    logger.info(f"generated {len(recommendations)} recommendations")
    return recommendations

##################################################################
#####                                                        #####
#####          API for single book detail dans frontend      #####
#####                                                        #####
##################################################################
def get_book_detail(request, book_id):
    try:
        book = Book.objects.get(id=book_id)

        # Prepare the book data
        book_data = {
            'id': book.id,
            'title': book.title,
            'author': book.author,
            'language': book.language,
            'download_count': book.download_count,
            'cover_url': book.cover_url,
            'word_count': book.word_count,
            'content': book.content,
        }

        related_books = []
        try:
            similar_docs = DocumentSimilarity.objects.filter(
                Q(document1_id=book_id) | Q(document2_id=book_id)
            ).order_by('-jaccard_similarity')[:5]

            for sim in similar_docs:
                related_id = sim.document2_id if sim.document1_id == book_id else sim.document1_id
                related_book = Book.objects.get(id=related_id)

                related_books.append({
                    'id': related_book.id,
                    'title': related_book.title,
                    'author': related_book.author,
                    'similarity': sim.jaccard_similarity
                })

            book_data['related_books'] = related_books
        except Exception as e:
            logger.error(f"Error fetching related books: {str(e)}")

        return JsonResponse(book_data)

    except Book.DoesNotExist:
        return JsonResponse({"error": "Livre non trouvé"}, status=404)

    except Exception as e:
        logger.error(f"Error fetching book detail: {str(e)}")
        return JsonResponse({"error": f"Erreur serveur: {str(e)}"}, status=500)


##################################################################
#####                                                        #####
#####               reduce taille of indextable              #####
#####                                                        #####
##################################################################
def prune_term_document_index(request=None, terms_to_keep=1000, batch_size=100, max_workers=8):
    """
    Prune the term-document index to keep only the top N terms with highest TF-IDF scores
    for each document. Uses parallel processing to improve performance.
    """
    start_time = time.time()

    logger.info(f"Starting parallel index pruning with {max_workers} workers, keeping top {terms_to_keep} terms per document...")

    # Get all document IDs
    document_ids = list(Book.objects.values_list('id', flat=True))
    total_docs = len(document_ids)

    # Statistics before pruning
    total_indices_before = TermDocumentIndex.objects.count()

    # Define worker function to process a batch of documents
    def process_document_batch(doc_ids_batch):
        batch_indices_to_keep = []
        processed_count = 0

        for doc_id in doc_ids_batch:
            # Get all index records for this document, ordered by TF-IDF score descending
            doc_indices = list(TermDocumentIndex.objects.filter(
                document_id=doc_id
            ).order_by('-tfidf').values_list('id', flat=True))

            # If there are more records than we want to keep
            if len(doc_indices) > terms_to_keep:
                # Keep only the highest scoring terms
                batch_indices_to_keep.extend(doc_indices[:terms_to_keep])
            else:
                # If there are fewer records than the limit, keep all of them
                batch_indices_to_keep.extend(doc_indices)

            processed_count += 1

        return batch_indices_to_keep, processed_count

    # Divide documents into batches for parallel processing
    doc_batches = []
    docs_per_worker = (total_docs + max_workers - 1) // max_workers  # Ceiling division

    for i in range(0, total_docs, docs_per_worker):
        doc_batches.append(document_ids[i:i+docs_per_worker])

    logger.info(f"Split {total_docs} documents into {len(doc_batches)} batches for processing")

    # Process batches in parallel
    indices_to_keep = []
    total_processed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_batch = {executor.submit(process_document_batch, batch): i
                           for i, batch in enumerate(doc_batches)}

        # Process results as they complete
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_indices, batch_processed = future.result()
                indices_to_keep.extend(batch_indices)
                total_processed += batch_processed

                # Log progress
                progress = (total_processed / total_docs) * 100
                logger.info(f"Batch {batch_idx+1}/{len(doc_batches)} completed. Overall progress: {progress:.1f}% ({total_processed}/{total_docs} documents)")

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")

    logger.info(f"Determined {len(indices_to_keep)} index entries to keep. Starting deletion process...")

    all_indices = set(TermDocumentIndex.objects.values_list('id', flat=True))
    indices_to_keep_set = set(indices_to_keep)
    indices_to_delete = list(all_indices - indices_to_keep_set)

    logger.info(f"Identified {len(indices_to_delete)} index entries to delete. Starting batch deletion...")

    delete_batch_size = 500  # limit of sqlite: 1000
    deleted_count = 0

    for i in range(0, len(indices_to_delete), delete_batch_size):
        batch = indices_to_delete[i:i+delete_batch_size]
        with transaction.atomic():
            batch_deleted = TermDocumentIndex.objects.filter(id__in=batch).delete()[0]
            deleted_count += batch_deleted

        progress = (i + len(batch)) / len(indices_to_delete) * 100
        logger.info(f"Deleted batch {i//delete_batch_size + 1}/{(len(indices_to_delete) + delete_batch_size - 1)//delete_batch_size}. "
                    f"Progress: {progress:.1f}% ({i + len(batch)}/{len(indices_to_delete)})")

    # Statistics after pruning
    total_indices_after = TermDocumentIndex.objects.count()
    duration = time.time() - start_time
    percentage_reduced = ((total_indices_before - total_indices_after) / total_indices_before) * 100 if total_indices_before > 0 else 0

    logger.info(f"Parallel index pruning completed in {duration:.2f} seconds")
    logger.info(f"Index entries: {total_indices_before} before, {total_indices_after} after")
    logger.info(f"Deleted {deleted_count} index records, reduced by {percentage_reduced:.2f}%")

    result = {
        "message": "Parallel index pruning completed",
        "documents_processed": total_processed,
        "workers_used": max_workers,
        "indices_before": total_indices_before,
        "indices_after": total_indices_after,
        "indices_deleted": deleted_count,
        "percentage_reduced": percentage_reduced,
        "duration_seconds": duration
    }

    if request:
        return JsonResponse(result)

    return result

def prune_index(request):
    """
    API endpoint to prune the term-document index, keeping only the top N terms
    with highest TF-IDF scores for each document.
    """
    terms_to_keep = int(request.GET.get('terms_to_keep', '1000'))
    batch_size = int(request.GET.get('batch_size', '100'))
    max_workers = int(request.GET.get('max_workers', '8'))

    return prune_term_document_index(request, terms_to_keep, batch_size, max_workers)