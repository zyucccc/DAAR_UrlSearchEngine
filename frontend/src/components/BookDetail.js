import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";

function BookDetail() {
    const { id } = useParams();
    const [book, setBook] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");

    useEffect(() => {
        const fetchBookDetail = async () => {
            setLoading(true);
            setError("");

            try {
                const response = await fetch(`http://127.0.0.1:8000/api/book/${id}/`);

                if (!response.ok) {
                    throw new Error("Livre non trouvé");
                }

                const data = await response.json();
                setBook(data);
            } catch (err) {
                console.error("Erreur lors du chargement du livre:", err);
                setError("Impossible de charger les détails du livre. " + err.message);
            } finally {
                setLoading(false);
            }
        };

        if (id) {
            fetchBookDetail();
        }
    }, [id]);

    // Format the content with paragraphs
    const formatContent = (content) => {
        if (!content) return [];
        return content.split('\n\n').filter(p => p.trim().length > 0);
    };

    if (loading) {
        return (
            <div className="container mx-auto p-6 flex justify-center items-center h-screen">
                <div className="text-center">
                    <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
                    <p className="mt-4 text-gray-600">Chargement du livre...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="container mx-auto p-6">
                <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-lg">
                    <h2 className="font-bold text-lg mb-2">Erreur</h2>
                    <p>{error}</p>
                    <Link to="/" className="mt-4 inline-block text-blue-600 hover:underline">
                        Retour à l'accueil
                    </Link>
                </div>
            </div>
        );
    }

    if (!book) {
        return (
            <div className="container mx-auto p-6">
                <div className="bg-yellow-50 border border-yellow-200 text-yellow-700 p-4 rounded-lg">
                    <h2 className="font-bold text-lg mb-2">Livre non trouvé</h2>
                    <p>Nous n'avons pas pu trouver le livre que vous cherchez.</p>
                    <Link to="/" className="mt-4 inline-block text-blue-600 hover:underline">
                        Retour à l'accueil
                    </Link>
                </div>
            </div>
        );
    }

    const paragraphs = formatContent(book.content);

    return (
        <div className="container mx-auto p-6">
            <div className="mb-6">
                <Link to="/" className="text-blue-600 hover:underline flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z" clipRule="evenodd" />
                    </svg>
                    Retour à la recherche
                </Link>
            </div>

            <div className="bg-white shadow-lg rounded-lg overflow-hidden">
                <div className="p-6">
                    <h1 className="text-3xl font-bold mb-4">{book.title}</h1>

                    <div className="mb-6 flex flex-wrap gap-4">
                        <div className="bg-blue-50 rounded-lg p-3 flex-grow">
                            <h2 className="text-lg font-semibold mb-1">Auteur</h2>
                            <p>{book.author || "Inconnu"}</p>
                        </div>

                        <div className="bg-blue-50 rounded-lg p-3 flex-grow">
                            <h2 className="text-lg font-semibold mb-1">Langue</h2>
                            <p>{book.language}</p>
                        </div>

                        <div className="bg-blue-50 rounded-lg p-3 flex-grow">
                            <h2 className="text-lg font-semibold mb-1">Téléchargements</h2>
                            <p>{book.download_count.toLocaleString()}</p>
                        </div>

                        <div className="bg-blue-50 rounded-lg p-3 flex-grow">
                            <h2 className="text-lg font-semibold mb-1">Nombre de mots</h2>
                            <p>{book.word_count.toLocaleString()}</p>
                        </div>
                    </div>

                    {book.cover_url && (
                        <div className="mb-6">
                            <img
                                src={book.cover_url}
                                alt={`Couverture de ${book.title}`}
                                className="max-w-xs mx-auto shadow-md rounded"
                            />
                        </div>
                    )}

                    <div className="border-t border-gray-200 my-6 pt-6">
                        <h2 className="text-2xl font-bold mb-4">Contenu</h2>
                        <div className="prose max-w-none">
                            {paragraphs.length > 0 ? (
                                paragraphs.map((paragraph, index) => (
                                    <p key={index} className="mb-4">{paragraph}</p>
                                ))
                            ) : (
                                <p className="text-gray-500 italic">Aucun contenu disponible pour ce livre.</p>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default BookDetail;