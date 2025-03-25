import React from "react";

function BookCard({ book }) {
  return (
    <div className="border p-4 rounded shadow-md bg-white">
      <h2 className="text-lg font-bold">{book.title}</h2>
      <p>
        Auteur(s) :{" "}
        {book.author
          ? book.author.split(",").map((author, index) => (
              <span key={index}>{author.trim()}{index < book.author.split(",").length - 1 ? ", " : ""}</span>
            ))
          : "Inconnu"}
      </p>
    </div>
  );
}

export default BookCard;
