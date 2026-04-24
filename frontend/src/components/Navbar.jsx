import React from "react";
import { Link } from "react-router-dom";

function Navbar() {
  return (
    <nav style={styles.nav}>
      <Link to="/" style={{ ...styles.logo, textDecoration: "none", color: "white" }}>
        PlayCaller
      </Link>

      <div style={styles.links}>
        <Link to="/" style={styles.link}>Home</Link>
        <Link to="/about" style={styles.link}>About</Link>
        <Link to="/dashboard" style={styles.link}>Dashboard</Link>
      </div>
    </nav>
  );
}

const styles = {
  nav: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "1rem 2rem",
    backgroundColor: "#1e1e1e",
    color: "#ffffff",
  },
  logo: {
    margin: 0,
    fontSize: "1.5rem",
    fontWeight: "bold",
    cursor: "pointer", // makes it feel clickable
  },
  links: {
    display: "flex",
    gap: "1rem",
  },
  link: {
    color: "white",
    textDecoration: "none",
  },
};

export default Navbar;
