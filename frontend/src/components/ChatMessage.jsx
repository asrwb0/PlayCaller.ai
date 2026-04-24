import React from "react";

function ChatMessage({ message, sender }) {
  return (
    <div style={{
      display: "flex",
      justifyContent: sender === "user" ? "flex-end" : "flex-start",
      marginBottom: "8px"
    }}>
      <div style={{
        backgroundColor: sender === "user" ? "#4CAF50" : "#333",
        color: "white",
        padding: "10px 15px",
        borderRadius: "20px",
        maxWidth: "70%",
        wordBreak: "break-word"
      }}>
        {message}
      </div>
    </div>
  );
}

export default ChatMessage;