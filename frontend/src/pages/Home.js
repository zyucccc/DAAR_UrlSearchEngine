import React from "react";

function Home() {
  return (
    <div className="relative w-full h-[80.5vh] flex items-center justify-center bg-black">
      {/* Image de fond */}
      <img 
        src="/photodaar1.png" 
        alt="Livres en vedette" 
        className="absolute w-full h-full object-cover opacity-50"
      />

      {/* Contenu au-dessus de l’image */}
      <div className="absolute text-white text-left max-w-2xl px-10">
        <h1 className="text-5xl font-bold leading-tight">
          Aslak Nore : le retour du maître du polar scandinave
        </h1>
        <button className="mt-6 px-6 py-3 bg-white text-red-800 font-semibold rounded-lg hover:bg-gray-300 transition">
          Découvrir
        </button>
      </div>
    </div>
  );
}

export default Home;
