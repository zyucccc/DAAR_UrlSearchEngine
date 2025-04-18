# Generated by Django 4.2.20 on 2025-03-26 23:43

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0002_term_book_content_book_word_count_termdocumentindex"),
    ]

    operations = [
        migrations.CreateModel(
            name="DocumentSimilarity",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("jaccard_similarity", models.FloatField(default=0.0)),
                (
                    "document1",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="similarities_as_doc1",
                        to="api.book",
                    ),
                ),
                (
                    "document2",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="similarities_as_doc2",
                        to="api.book",
                    ),
                ),
            ],
            options={
                "indexes": [
                    models.Index(
                        fields=["document1"], name="api_documen_documen_f24968_idx"
                    ),
                    models.Index(
                        fields=["document2"], name="api_documen_documen_d9de36_idx"
                    ),
                ],
                "unique_together": {("document1", "document2")},
            },
        ),
        migrations.CreateModel(
            name="DocumentCentrality",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("centrality_type", models.CharField(max_length=20)),
                ("score", models.FloatField(default=0.0)),
                (
                    "document",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="api.book"
                    ),
                ),
            ],
            options={
                "indexes": [
                    models.Index(
                        fields=["document"], name="api_documen_documen_f9bb24_idx"
                    ),
                    models.Index(
                        fields=["centrality_type"],
                        name="api_documen_central_501933_idx",
                    ),
                ],
                "unique_together": {("document", "centrality_type")},
            },
        ),
    ]
