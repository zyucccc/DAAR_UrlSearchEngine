from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=255)
    author = models.CharField(max_length=255, blank=True, null=True)
    language = models.CharField(max_length=10)
    download_count = models.IntegerField(default=0)
    cover_url = models.URLField(blank=True, null=True)

    def __str__(self):
        return self.title
