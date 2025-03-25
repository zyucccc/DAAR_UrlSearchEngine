from django.urls import path
from .views import fetch_books_from_gutendex, rank_books, regex_search_books, search_books, suggest_books

urlpatterns = [
    path('gutendex/', fetch_books_from_gutendex, name="fetch-books"),
    #path('search-books/', search_books, name='search-books'),  
    path("search-books/", search_books, name="search-books"),
    path("regex-search-books/", regex_search_books, name="regex-search-books"),
    path('rank/', rank_books, name="rank-books"),
    path('suggest/', suggest_books, name="suggest-books"),
]
