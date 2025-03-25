import React, { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import BookCard from "../components/BookCard";

function useQuery() {
  return new URLSearchParams(useLocation().search);
}

function SearchResults() {
  const query = useQuery().get("q");
  const [books, setBooks] = useState([]);
  const [error, setError] = useState("");
  const [isLoading,setIsLoading] = useState(false);

  useEffect(() => {
    if (query) {
      const fetchBooks = async () => {
        setError("");  
        setBooks([]);
        setIsLoading(true);
        
        const encodedQuery = encodeURIComponent(query);
        const url = `http://127.0.0.1:8000/api/search-books/?q=${encodedQuery}`;

        try {
          const response = await fetch(url);
          const data = await response.json();

          if (response.ok) {
            setBooks(data);  // Met à jour la liste des livres
          } else {
            setBooks([]);
            setError(data.error || "Aucun livre trouvé.");
          }
        } catch (err) {
          console.error("Erreur :", err);
          setError("Problème de connexion avec le serveur.");
        } finally {
          setIsLoading(false);
        }
      };

      fetchBooks();
    }
  }, [query]);

  return (
    <div className="container py-10">
      <h1 className="text-2xl font-bold">Résultats pour "{query}"</h1>

      {error && (
        <p className="text-center text-red-500 mt-6">
          {error}
        </p>
      )}

      {isLoading ? (
          <div className="text-center mt-6">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
            <p className="text-center text-gray-600 mt-2">
              Recherche en cours...
            </p>
          </div>
      ) : (
          //result == vide
          books.length === 0 && !error && (
              <p className="text-center text-gray-500 mt-6">
                Aucun livre trouvé pour "<span className="font-bold">{query}</span>". <br />
                Essayez un autre mot-clé ou une variante !
              </p>
          )
      )}

      <div className="grid grid-cols-3 gap-6 mt-6">
        {books.map((book, index) => (
            <BookCard key={index} book={book} />
        ))}
      </div>
    </div>
  );
}
export default SearchResults;
