require("dotenv").config();
const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const faceRoutes = require("./src/routes/faceRoutes");
const { initializeDatabase } = require("./src/config/database");
const { initializeModel } = require("./src/services/modelService");

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true, limit: "10mb" }));

// Routes
app.use("/api", faceRoutes);

// Health check endpoint
app.get("/health", (req, res) => {
  res.json({
    status: "OK",
    message: "Face Verification Microservice is running",
    timestamp: new Date().toISOString(),
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error("Error:", err.message);
  res.status(500).json({
    error: "Internal server error",
    message:
      process.env.NODE_ENV === "development"
        ? err.message
        : "Something went wrong",
  });
});

// 404 handler
app.use("*", (req, res) => {
  res.status(404).json({ error: "Endpoint not found" });
});

// Initialize and start server
async function startServer() {
  try {
    console.log("🚀 Starting Face Verification Microservice...");

    // Initialize database (optional)
    console.log("📊 Initializing database...");
    try {
      await initializeDatabase();
      console.log("✅ Database initialized successfully");
    } catch (error) {
      console.warn(
        "⚠️ Database initialization failed, continuing without database:",
        error.message
      );
      console.warn(
        "⚠️ Face registration and verification will not work until database is connected"
      );
    }

    // Initialize AI model
    console.log("🤖 Loading ArcFace model...");
    await initializeModel();
    console.log("✅ ArcFace model loaded successfully");

    // Start server
    app.listen(PORT, () => {
      console.log(`\n🎉 Server running on http://localhost:${PORT}`);
      console.log(`📋 Health check: http://localhost:${PORT}/health`);
      console.log(`🔍 API endpoints:`);
      console.log(
        `   POST http://localhost:${PORT}/api/encode - Register face`
      );
      console.log(`   POST http://localhost:${PORT}/api/compare - Verify face`);
      console.log(
        `\n🛡️  Environment: ${process.env.NODE_ENV || "development"}`
      );
    });
  } catch (error) {
    console.error("❌ Failed to start server:", error.message);
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on("SIGINT", () => {
  console.log("\n🛑 Shutting down server gracefully...");
  process.exit(0);
});

process.on("SIGTERM", () => {
  console.log("\n🛑 Shutting down server gracefully...");
  process.exit(0);
});

startServer();
