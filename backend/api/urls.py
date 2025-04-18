from django.urls import path
from .views import fetch_books_from_gutendex, regex_search_books, search_books, calculate_tfidf, \
    calculate_document_similarities, calculate_centrality_scores, get_book_detail, prune_index

urlpatterns = [
    # ----------------- Init BD avec gutendex ----------------- #
    # exemples:
    # http://127.0.0.1:8000/api/gutendex/?reset_data=true&workers=10&skip_tfidf=true
    # http://127.0.0.1:8000/api/calculate-tfidf/?term_workers=8&batch_size=3000&update_workers=8&title_boost=1.5&author_boost=0.8
    path('gutendex/', fetch_books_from_gutendex, name="fetch-books"),
    # ----------------- calculate TD-IDF dans indextable ----------------- #
    path('calculate-tfidf/', calculate_tfidf, name='calculate-tfidf'),
    # ----------------- Prune index table ----------------- #
    # ex: http://127.0.0.1:8000/api/prune-index/?terms_to_keep=1000&batch_size=100&max_workers=8
    path('prune-index/', prune_index, name='prune-index'),
    # ----------------- calculate Jaccard distance ----------------- #
    # ex: http://127.0.0.1:8000/api/calculate-similarities/?similarity_threshold=0.1&max_workers=8
    path('calculate-similarities/', calculate_document_similarities, name="calculate-similarities"),
    # ----------------- calculate centrality (closeness,bewteeness,page rank) ----------------- #
    # ex: http://127.0.0.1:8000/api/calculate-centrality/?min_similarity=0.5&max_workers=8
    path('calculate-centrality/', calculate_centrality_scores, name="calculate-centrality"),

    # ---------------------------- Search ------------------------------- #
    path("search-books/", search_books, name="search-books"),
    path("regex-search-books/", regex_search_books, name="regex-search-books"),
    # ---------------------------- single book detail ------------------------------- #
    path('book/<int:book_id>/', get_book_detail, name='book_detail'),
]
