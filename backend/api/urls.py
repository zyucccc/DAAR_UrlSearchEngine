from django.urls import path
from .views import fetch_books_from_gutendex, rank_books, regex_search_books, search_books, suggest_books, calculate_tfidf, calculate_document_similarities, calculate_centrality_scores

urlpatterns = [
    # ----------------- Init BD avec gutendex ----------------- #
    # exemples:
    # http://127.0.0.1:8000/api/gutendex/?reset_data=true&workers=10&skip_tfidf=true
    # # http://127.0.0.1:8000/api/calculate-tfidf/?term_workers=8&batch_size=3000&update_workers=8&title_boost=1.5&author_boost=0.8
    path('gutendex/', fetch_books_from_gutendex, name="fetch-books"),
    # ----------------- calculate TD-IDF dans indextable ----------------- #
    path('calculate-tfidf/', calculate_tfidf, name='calculate-tfidf'),
    # ----------------- calculate Jaccard distance ----------------- #
    path('calculate-similarities/', calculate_document_similarities, name="calculate-similarities"),
    # ----------------- calculate centrality (closeness,bewteeness,page rank) ----------------- #
    path('calculate-centrality/', calculate_centrality_scores, name="calculate-centrality"),
    # ----------------- Search ----------------- #
    path("search-books/", search_books, name="search-books"),
    path("regex-search-books/", regex_search_books, name="regex-search-books"),
    path('rank/', rank_books, name="rank-books"),
    path('suggest/', suggest_books, name="suggest-books"),
]
