import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { FaSearch } from "react-icons/fa";


function Navbar() {
  const navigate = useNavigate();
  const handleSearch = (event) => {
    if (event.key === "Enter") {
      navigate(`/search?q=${event.target.value}`);
    }
  };

  return (
    <nav className="bg-red-800 text-white py-4 shadow-md">
      <div className="container mx-auto flex justify-between items-center px-6">
        {/* Logo */}
        <Link to="/">
          <img src="/photodaar2.png" alt="Logo" className="h-16" /> 
        </Link>
        <div className="bg-white flex items-center px-4 py-2 rounded-full shadow-md w-1/3">
          <FaSearch className="text-gray-500" />
          <input
            type="text"
            placeholder="Rechercher un livre..."
            className="w-full pl-2 focus:outline-none bg-transparent text-black"
            onKeyPress={handleSearch}
          />
        </div>
        <div className="space-x-6">
          <Link to="/" className="text-lg font-semibold hover:underline">Accueil</Link>
          <Link to="/about" className="text-lg font-semibold hover:underline">À propos</Link>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;

/*const handleSearch = (event) => {
    if (event.key === "Enter") {
      const query = encodeURIComponent(event.target.value.trim());
      navigate(`/search_regex?regex=${query}`);
    }
  };  */