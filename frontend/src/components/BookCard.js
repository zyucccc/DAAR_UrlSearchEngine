import React from "react";
import { Link } from "react-router-dom";

function BookCard({ book }) {
    return (
        <div className="border p-4 rounded shadow-md bg-white hover:shadow-lg transition-shadow flex flex-col h-full">
            {/* cover image */}
            {book.cover_url ? (
                <div className="mb-4 flex justify-center">
                    <img
                        src={book.cover_url}
                        alt={`Couverture de ${book.title}`}
                        className="object-cover h-48 w-auto rounded shadow"
                    />
                </div>
            ) : (
                <div className="mb-4 flex justify-center items-center bg-gray-100 h-48 w-full rounded shadow">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                    </svg>
                </div>
            )}

            <h2 className="text-lg font-bold mb-2 line-clamp-2">{book.title}</h2>
            <p className="mb-3 text-sm text-gray-600">
                Auteur(s) :{" "}
                {book.author
                    ? book.author.split(",").map((author, index) => (
                        <span key={index}>{author.trim()}{index < book.author.split(",").length - 1 ? ", " : ""}</span>
                    ))
                    : "Inconnu"}
            </p>

            {book.combined_score && (
                <div className="flex items-center mb-3">
                    <div className="text-yellow-500 mr-1">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                        </svg>
                    </div>
                    <div className="text-gray-700">
                        Pertinence Score: <span className="font-medium">{(book.combined_score * 100).toFixed(2)}</span>
                    </div>
                </div>
            )}

            <div className="mt-auto pt-3 border-t border-gray-200">
                <Link
                    to={`/book/${book.id}`}
                    className="block w-full text-center py-2 px-4 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-md transition-colors"
                >
                    Voir ce livre
                </Link>
            </div>
        </div>
    );
}

export default BookCard;