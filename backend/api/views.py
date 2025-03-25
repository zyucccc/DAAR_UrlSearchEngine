import requests
from django.http import JsonResponse
from .models import Book

from django.shortcuts import get_list_or_404
from django.db.models import Q

import re

import networkx as nx
import numpy as np


""" ----------------- Récupérer des livres depuis Gutendex API ----------------- """

"""def fetch_books_from_gutendex(request):
    url = "https://gutendex.com/books/"
    response = requests.get(url)
    
    if response.status_code == 200:
        books_data = response.json()["results"]
        for book in books_data:
            Book.objects.get_or_create(
                title=book["title"],
                author=", ".join([a["name"] for a in book.get("authors", [])]),  # ← Fix ici
                language=", ".join(book["languages"]),
                download_count=book["download_count"],
                cover_url=book["formats"].get("image/jpeg", "")
            )

        return JsonResponse(books_data, safe=False)

    return JsonResponse({"error": "Failed to fetch books"}, status=500)"""

def fetch_books_from_gutendex(request):
    base_url = "https://gutendex.com/books/"
    count = 0

    while base_url:
        response = requests.get(base_url)
        if response.status_code != 200:
            return JsonResponse({"error": "Erreur lors de la récupération des livres"}, status=500)

        data = response.json()
        books_data = data["results"]

        for book in books_data:
            authors = ", ".join([a["name"] for a in book.get("authors", [])])
            languages = ", ".join(book.get("languages", []))
            download_count = book.get("download_count", 0)
            cover_url = book["formats"].get("image/jpeg", "")
            
            Book.objects.get_or_create(
                title=book["title"],
                author=authors,
                language=languages,
                download_count=download_count,
                cover_url=cover_url
            )
            count += 1

        base_url = data.get("next")

    return JsonResponse({"message": f"{count} livres importés"}, safe=False)


""" ----------------- Recherche simple par mot-clé -----------------"""

def search_books(request):
    query = request.GET.get("q", "")
    
    if query:
        books = Book.objects.filter(Q(title__icontains=query) | Q(author__icontains=query))
        data = list(books.values())
        return JsonResponse(data, safe=False)
    
    return JsonResponse({"error": "No books found"}, status=404)

""" ---------------- Recherche avancée par expression régulière ----------------- """

def regex_search_books(request):
    regex = request.GET.get("regex", "")
    
    if regex:
        books = Book.objects.all()
        matched_books = [book for book in books if re.search(regex, book.title, re.IGNORECASE)]
        data = [{"title": b.title, "author": b.author} for b in matched_books]
        return JsonResponse(data, safe=False)
    
    return JsonResponse({"error": "No books found"}, status=404)

"""def regex_search_books(request):
    regex = request.GET.get("regex", "")
    
    if not regex:
        return JsonResponse({"error": "Expression vide"}, status=400)

    try:
        pattern = re.compile(regex, re.IGNORECASE)
        books = Book.objects.all()
        matched_books = [book for book in books if pattern.search(book.title)]
    except re.error:
        return JsonResponse({"error": "Expression régulière invalide"}, status=400)

    if matched_books:
        data = [{"title": b.title, "author": b.author} for b in matched_books]
        return JsonResponse(data, safe=False)

    return JsonResponse({"error": "Aucun livre trouvé"}, status=404)"""

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
