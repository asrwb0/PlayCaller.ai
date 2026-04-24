import React, { useState, useRef, useEffect } from "react";

function Chatbot() {
  const [messages, setMessages] = useState([
    { sender: "bot", message: "Hello! How can I help you today?" }
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  // Scroll to bottom whenever messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const getBotResponse = (userMessage) => {
    if (!userMessage) return "Please type something!";
    // Simulate a delay for typing
    setIsTyping(true);
    return new Promise((resolve) => {
      setTimeout(() => {
        setIsTyping(false);
        resolve(`You said: "${userMessage}"`);
      }, 1000);
    });
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { sender: "user", message: input };
    setMessages(prev => [...prev, userMessage]);
    setInput("");

    const botReply = await getBotResponse(input);
    const botMessage = { sender: "bot", message: botReply };
    setMessages(prev => [...prev, botMessage]);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") handleSend();
  };

  return (
    <div style={styles.chatContainer}>
      <div style={styles.messagesContainer}>
        {messages.map((msg, idx) => (
          <div
            key={idx}
            style={{
              display: "flex",
              justifyContent: msg.sender === "user" ? "flex-end" : "flex-start",
              marginBottom: "10px"
            }}
          >
            <div style={{
              backgroundColor: msg.sender === "user" ? "#4caf50" : "#2c2c2c",
              color: "white",
              padding: "10px 15px",
              borderRadius: "20px",
              maxWidth: "70%",
              wordBreak: "break-word",
              boxShadow: "0 1px 3px rgba(0,0,0,0.3)"
            }}>
              {msg.message}
            </div>
          </div>
        ))}

        {isTyping && (
          <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: "10px" }}>
            <div style={{ ...styles.typingBubble }}>Bot is typing...</div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div style={styles.inputContainer}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type a message..."
          style={styles.input}
        />
        <button onClick={handleSend} style={styles.sendButton}>Send</button>
      </div>
    </div>
  );
}

const styles = {
  chatContainer: {
    width: "400px",
    maxWidth: "90%",
    height: "500px",
    display: "flex",
    flexDirection: "column",
    backgroundColor: "#1e1e1e",
    borderRadius: "15px",
    boxShadow: "0 4px 15px rgba(0,0,0,0.3)",
    overflow: "hidden",
    fontFamily: "Arial, sans-serif"
  },
  messagesContainer: {
    flex: 1,
    padding: "15px",
    overflowY: "auto",
  },
  inputContainer: {
    display: "flex",
    padding: "10px",
    borderTop: "1px solid #333",
    backgroundColor: "#2a2a2a"
  },
  input: {
    flex: 1,
    padding: "10px 15px",
    borderRadius: "20px",
    border: "none",
    outline: "none",
    marginRight: "10px",
    backgroundColor: "#3a3a3a",
    color: "white",
  },
  sendButton: {
    padding: "10px 20px",
    borderRadius: "20px",
    border: "none",
    backgroundColor: "#4caf50",
    color: "white",
    cursor: "pointer"
  },
  typingBubble: {
    backgroundColor: "#2c2c2c",
    color: "white",
    padding: "10px 15px",
    borderRadius: "20px",
    fontStyle: "italic",
    boxShadow: "0 1px 3px rgba(0,0,0,0.3)"
  }
};

export default Chatbot;