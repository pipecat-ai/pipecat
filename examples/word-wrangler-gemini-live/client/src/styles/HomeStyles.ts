export const styles = {
  main: {
    display: "flex",
    flexDirection: "column" as const,
    justifyContent: "flex-start",
    alignItems: "center",
    minHeight: "100vh",
    padding: "2rem 0",
  },
  container: {
    width: "100%",
    maxWidth: "800px",
    padding: "0 1rem",
  },
  title: {
    fontSize: "2rem",
    textAlign: "center" as const,
    marginBottom: "2rem",
    color: "#333",
  },
  gameContainer: {
    marginBottom: "2rem",
  },
  controlsContainer: {
    display: "flex",
    flexDirection: "column" as const,
    gap: "1rem",
    marginBottom: "2rem",
  },
  settings: {
    backgroundColor: "white",
    padding: "1rem",
    borderRadius: "8px",
    boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
  },
  label: {
    display: "flex",
    flexDirection: "column" as const,
    gap: "0.5rem",
    fontSize: "0.9rem",
    color: "#555",
  },
  select: {
    padding: "0.5rem",
    border: "1px solid #ddd",
    borderRadius: "4px",
    fontSize: "1rem",
  },
  instructions: {
    backgroundColor: "white",
    padding: "1.5rem",
    borderRadius: "8px",
    boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
  },
  instructionsTitle: {
    fontSize: "1.4rem",
    marginBottom: "1rem",
    color: "#333",
  },
  instructionsList: {
    paddingLeft: "1.5rem",
    lineHeight: 1.6,
  },
};
