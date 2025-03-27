from django.db import models

class Book(models.Model):
    """Document - 1664 docs in total"""
    title = models.CharField(max_length=255)
    author = models.CharField(max_length=255, blank=True, null=True)
    language = models.CharField(max_length=10)
    download_count = models.IntegerField(default=0)
    cover_url = models.URLField(blank=True, null=True)
    word_count = models.IntegerField(default=0) #termes count
    content = models.TextField(blank=True) #contenu du livre

    def __str__(self):
        return self.title

class Term(models.Model):
    """Terme - pour chaque mot unique dans les documents"""
    word = models.CharField(max_length=100, unique=True)
    document_count = models.IntegerField(default=0)  # the number of documents containing this term,for idf

    def __str__(self):
        return self.word

class TermDocumentIndex(models.Model):
    """Index Table - Terme ->DoC->Score TF_IDF"""
    term = models.ForeignKey(Term, on_delete=models.CASCADE)
    document = models.ForeignKey(Book, on_delete=models.CASCADE)
    tf = models.FloatField(default=0.0)  # 词频
    tfidf = models.FloatField(default=0.0)  # TF-IDF权重

    class Meta:
        # make sure that in the database, the combination of 1 term and 1 document is unique
        unique_together = ('term', 'document')
        indexes = [
            models.Index(fields=['term']),
            models.Index(fields=['document']),
        ]


class DocumentSimilarity(models.Model):
    """Jaccard - Similarity entre 2 doc"""
    document1 = models.ForeignKey(Book, related_name='similarities_as_doc1', on_delete=models.CASCADE)
    document2 = models.ForeignKey(Book, related_name='similarities_as_doc2', on_delete=models.CASCADE)
    jaccard_similarity = models.FloatField(default=0.0) # Jaccard distance

    class Meta:
        unique_together = ('document1', 'document2')
        indexes = [
            models.Index(fields=['document1']),
            models.Index(fields=['document2']),
        ]

    def __str__(self):
        return f"{self.document1.title} - {self.document2.title}: {self.jaccard_similarity}"


class DocumentCentrality(models.Model):
    """Centrality entre des docs - closeness'"""
    document = models.ForeignKey(Book, on_delete=models.CASCADE)
    centrality_type = models.CharField(max_length=20)  # 'closeness', 'betweenness', 'pagerank'
    score = models.FloatField(default=0.0)

    class Meta:
        unique_together = ('document', 'centrality_type')
        indexes = [
            models.Index(fields=['document']),
            models.Index(fields=['centrality_type']),
        ]

    def __str__(self):
        return f"{self.document.title} - {self.centrality_type}: {self.score}"
