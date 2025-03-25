import React from "react";
import { FaFacebook, FaInstagram, FaPinterest, FaTiktok } from "react-icons/fa";

function Footer() {
  return (
    <footer className="bg-red-800 text-white py-6 mt-10">
      <div className="container mx-auto text-center space-y-4">
        {/* Texte principal */}
        <p className="text-lg font-semibold">© 2025 Bibliothèque</p>
        <p className="text-sm">Mentions légales | Conditions d'utilisation</p>

        {/* Icônes sociales bien espacées */}
        <div className="flex justify-center space-x-6 mt-4">
          <FaFacebook className="text-2xl cursor-pointer hover:text-gray-300" />
          <FaInstagram className="text-2xl cursor-pointer hover:text-gray-300" />
          <FaPinterest className="text-2xl cursor-pointer hover:text-gray-300" />
          <FaTiktok className="text-2xl cursor-pointer hover:text-gray-300" />
        </div>
      </div>
    </footer>
  );
}

export default Footer;
