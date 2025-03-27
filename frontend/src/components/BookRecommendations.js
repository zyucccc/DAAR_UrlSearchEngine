import React, { useState } from 'react';
import { Link } from 'react-router-dom';

const BookRecommendations = ({ suggestions }) => {
    const [isOpen, setIsOpen] = useState(true);

    if (!suggestions || suggestions.length === 0 || !isOpen) {
        return null;
    }

    return (
        <div className="fixed bottom-0 left-0 right-0 bg-white shadow-lg rounded-t-lg p-4 border-t border-gray-200 z-50 transform transition-transform duration-300">
            <div className="flex justify-between items-center mb-4">
                <h3 className="text-xl font-bold text-gray-800">Recommandations pour vous</h3>
                <div>
                    <button
                        onClick={() => setIsOpen(false)}
                        className="p-2 rounded-full hover:bg-gray-100 transition-colors"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>
            </div>
            <div className="flex overflow-x-auto space-x-4 pb-4">
                {suggestions.map(book => (
                    <div key={book.id} className="flex-none w-64">
                        <div className="border rounded-lg overflow-hidden shadow-md bg-white h-full flex flex-col">
                            <div className="p-4 flex-grow">
                                <h4 className="font-bold text-lg mb-2 line-clamp-2">{book.title}</h4>
                                <p className="text-gray-600 mb-2">{book.author}</p>

                                {book.recommended_based_on && (
                                    <div className="mb-2 text-sm">
                                        <span className="text-gray-500">Basé sur: </span>
                                        <span className="font-medium">{book.recommended_based_on.title}</span>
                                    </div>
                                )}

                                <div className="flex items-center mt-2">
                                    <div className="text-yellow-500 mr-1">
                                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                                        </svg>
                                    </div>
                                    <div className="text-gray-700">
                                        Similarité: <span className="font-medium">{Math.round(book.similarity_score * 100)}%</span>
                                    </div>
                                </div>
                            </div>

                            <div className="bg-indigo-50 p-3 border-t">
                                <Link
                                    to={`/book/${book.id}`}
                                    className="block w-full text-center py-2 px-4 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-md transition-colors"
                                >
                                    Voir ce livre
                                </Link>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default BookRecommendations;