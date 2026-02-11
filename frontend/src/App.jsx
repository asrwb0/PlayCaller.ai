import './App.css'
import Navbar from "./components/Navbar";
import { BrowserRouter, Routes, Route } from "react-router-dom";

function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={
          <div style={{ padding: "2rem", color: "#ffffff" }}>
            <h1>Welcome to PlayCaller</h1>
            <p>Enter your data below.</p>
          </div>
        } />
        <Route path="/about" element={
          <div style={{ padding: "2rem", color: "#ffffff" }}>
            <h1>About</h1>
            <p>This is an ML prediction application.</p>
          </div>
        } />
        <Route path="/dashboard" element={
          <div style={{ padding: "2rem", color: "#ffffff" }}>
            <h1>Dashboard</h1>
            <p>Dashboard content coming soon.</p>
          </div>
        } />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
