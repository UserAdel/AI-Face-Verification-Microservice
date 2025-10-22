const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const embeddingService = require("../services/embeddingService");
const { isModelLoaded } = require("../services/modelService");

const router = express.Router();

// Configure multer for file uploads
const storage = multer.memoryStorage(); // Store files in memory
const upload = multer({
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
    files: 1, // Only allow 1 file
  },
  fileFilter: (req, file, cb) => {
    // Check file type
    const allowedMimes = ["image/jpeg", "image/jpg", "image/png", "image/webp"];

    if (allowedMimes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(
        new Error(
          `Invalid file type. Allowed types: ${allowedMimes.join(", ")}`
        ),
        false
      );
    }
  },
});

// Middleware to check if model is loaded
function checkModelLoaded(req, res, next) {
  if (!isModelLoaded()) {
    return res.status(503).json({
      success: false,
      error:
        "AI model not loaded yet. Please wait for initialization to complete.",
    });
  }
  next();
}

// POST /encode - Register user and generate embedding
router.post(
  "/encode",
  checkModelLoaded,
  upload.single("image"),
  async (req, res) => {
    try {
      console.log("📝 POST /encode - Starting user registration...");

      // Validate image upload
      if (!req.file) {
        return res.status(400).json({
          success: false,
          error: "No image file provided. Please upload an image.",
        });
      }

      // Optional: Get userId from request body or generate one
      const userId =
        req.body.userId ||
        `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      console.log(`📝 Processing registration for user: ${userId}`);
      console.log(
        `🖼️ Image details: ${req.file.originalname}, ${req.file.size} bytes, ${req.file.mimetype}`
      );

      // Generate embedding
      const embedding = await embeddingService.createEmbedding(req.file.buffer);

      // Store embedding in database (as required by specifications)
      const { storeUserEmbedding } = require("../config/database");
      const dbResult = await storeUserEmbedding(userId, embedding);
      console.log(
        `💾 Embedding successfully saved to database for user: ${userId} (DB ID: ${dbResult.id})`
      );

      // Return success response with embedding
      const response = {
        success: true,
        embedding: embedding,
        userId: userId,
        timestamp: new Date().toISOString(),
        stored: true,
        dbInfo: {
          id: dbResult.id,
          createdAt: dbResult.created_at,
          updatedAt: dbResult.updated_at,
        },
        imageInfo: {
          originalName: req.file.originalname,
          size: req.file.size,
          mimeType: req.file.mimetype,
        },
      };

      console.log(`✅ Registration completed for user: ${userId}`);
      res.status(200).json(response);
    } catch (error) {
      console.error("❌ /encode endpoint error:", error);

      // Return appropriate error response
      const statusCode =
        error.message.includes("No face detected") ||
        error.message.includes("Image too") ||
        error.message.includes("Invalid")
          ? 400
          : 500;

      res.status(statusCode).json({
        success: false,
        error:
          error.message || "Failed to process image and generate embedding",
      });
    }
  }
);

// POST /compare - Compare image against stored embedding
router.post(
  "/compare",
  checkModelLoaded,
  upload.single("image"),
  async (req, res) => {
    try {
      console.log("🔍 POST /compare - Starting face verification...");

      // Validate image upload
      if (!req.file) {
        return res.status(400).json({
          success: false,
          error: "No image file provided. Please upload an image.",
        });
      }

      // Validate stored embedding
      if (!req.body.storedEmbedding) {
        return res.status(400).json({
          success: false,
          error:
            "No stored embedding provided. Please provide storedEmbedding in the request body.",
        });
      }

      console.log(
        `🖼️ Verification image: ${req.file.originalname}, ${req.file.size} bytes, ${req.file.mimetype}`
      );

      // Parse stored embedding
      let storedEmbeddingArray;
      try {
        storedEmbeddingArray = embeddingService.parseStoredEmbedding(
          req.body.storedEmbedding
        );
      } catch (parseError) {
        return res.status(400).json({
          success: false,
          error: `Invalid stored embedding format: ${parseError.message}`,
        });
      }

      // Compare embeddings
      const comparisonResult = await embeddingService.compareEmbeddings(
        req.file.buffer,
        storedEmbeddingArray
      );

      // Add additional metadata to response
      const response = {
        ...comparisonResult,
        timestamp: new Date().toISOString(),
        imageInfo: {
          originalName: req.file.originalname,
          size: req.file.size,
          mimeType: req.file.mimetype,
        },
        embeddingInfo: {
          storedDimensions: storedEmbeddingArray.length,
          newDimensions: comparisonResult.newEmbeddingLength || 512,
        },
      };

      console.log(
        `✅ Verification completed - Match: ${comparisonResult.isMatch}, Similarity: ${comparisonResult.similarity}`
      );
      res.status(200).json(response);
    } catch (error) {
      console.error("❌ /compare endpoint error:", error);

      // Return appropriate error response
      const statusCode =
        error.message.includes("No face detected") ||
        error.message.includes("Image too") ||
        error.message.includes("Invalid") ||
        error.message.includes("dimension mismatch")
          ? 400
          : 500;

      res.status(statusCode).json({
        success: false,
        error: error.message || "Failed to compare face embeddings",
      });
    }
  }
);

// GET /info - Get service information
router.get("/info", (req, res) => {
  try {
    const serviceInfo = embeddingService.getServiceInfo();

    res.status(200).json({
      success: true,
      service: "Face Verification Microservice",
      version: "1.0.0",
      ...serviceInfo,
      endpoints: {
        encode: {
          method: "POST",
          path: "/api/encode",
          description: "Generate face embedding from image",
          parameters: {
            image: "multipart/form-data file (JPEG, PNG, WebP)",
            userId: "optional string for user identification",
          },
        },
        compare: {
          method: "POST",
          path: "/api/compare",
          description: "Compare face image against stored embedding",
          parameters: {
            image: "multipart/form-data file (JPEG, PNG, WebP)",
            storedEmbedding: "JSON string array of stored face embedding",
          },
        },
        info: {
          method: "GET",
          path: "/api/info",
          description: "Get service information and status",
        },
      },
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("❌ /info endpoint error:", error);
    res.status(500).json({
      success: false,
      error: "Failed to retrieve service information",
    });
  }
});

// GET /health - Health check endpoint
router.get("/health", (req, res) => {
  const health = {
    status: "OK",
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    modelLoaded: isModelLoaded(),
    version: "1.0.0",
  };

  res.status(200).json(health);
});

// Error handling middleware for multer
router.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    let message = "File upload error";

    switch (error.code) {
      case "LIMIT_FILE_SIZE":
        message = "File too large. Maximum size is 10MB.";
        break;
      case "LIMIT_FILE_COUNT":
        message = "Too many files. Only 1 file allowed.";
        break;
      case "LIMIT_UNEXPECTED_FILE":
        message = 'Unexpected file field. Use "image" field name.';
        break;
      default:
        message = error.message;
    }

    return res.status(400).json({
      success: false,
      error: message,
    });
  }

  // Handle other errors
  console.error("Route error:", error);
  res.status(500).json({
    success: false,
    error: error.message || "Internal server error",
  });
});

module.exports = router;
