from django.contrib import admin
from .models import Book, Term, TermDocumentIndex

# Register your models here.
# admin.site.register(Book)
@admin.register(Book)
class BookAdmin(admin.ModelAdmin):
    list_display = ('title', 'author', 'language', 'download_count', 'word_count')
    list_filter = ('language',)
    search_fields = ('title', 'author')
    list_per_page = 20

@admin.register(Term)
class TermAdmin(admin.ModelAdmin):
    list_display = ('word', 'document_count')
    search_fields = ('word',)
    list_per_page = 20

@admin.register(TermDocumentIndex)
class TermDocumentIndexAdmin(admin.ModelAdmin):
    list_display = ('term', 'document', 'tf', 'tfidf')
    list_filter = ('term', 'document')
    search_fields = ('term__word', 'document__title')
    list_per_page = 20