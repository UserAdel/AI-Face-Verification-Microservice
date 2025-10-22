const faceDetection = require("./faceDetection");
const { generateEmbedding, isModelLoaded } = require("./modelService");
const { storeUserEmbedding, getUserEmbedding } = require("../config/database");
const { calculateCosineSimilarity } = require("../utils/similarity");

class EmbeddingService {
  constructor() {
    this.similarityThreshold =
      parseFloat(process.env.SIMILARITY_THRESHOLD) || 0.6;
  }

  // Generate embedding from image buffer
  async createEmbedding(imageBuffer) {
    try {
      // Check if model is loaded
      if (!isModelLoaded()) {
        throw new Error("AI model not loaded. Please wait for initialization.");
      }

      console.log("🧪 Starting embedding generation...");

      // Process face image (detection + preprocessing)
      const processedImageData = await faceDetection.processFaceImage(
        imageBuffer
      );

      // Generate embedding using AI model
      const embedding = await generateEmbedding(processedImageData);

      // Validate embedding
      if (!embedding || embedding.length === 0) {
        throw new Error("Failed to generate valid embedding");
      }

      console.log(`✅ Successfully generated ${embedding.length}D embedding`);
      return embedding;
    } catch (error) {
      console.error("❌ Embedding generation failed:", error);
      throw error;
    }
  }

  // Register user with embedding
  async registerUser(userId, imageBuffer) {
    try {
      console.log(`📝 Registering user: ${userId}`);

      // Generate embedding
      const embedding = await this.createEmbedding(imageBuffer);

      // Store in database
      const result = await storeUserEmbedding(userId, embedding);

      console.log(`✅ User ${userId} registered successfully`);
      return {
        success: true,
        userId: result.user_id,
        embedding: embedding,
        timestamp: result.updated_at,
      };
    } catch (error) {
      console.error(`❌ User registration failed for ${userId}:`, error);
      throw error;
    }
  }

  // Verify user against stored embedding
  async verifyUser(userId, imageBuffer) {
    try {
      console.log(`🔍 Verifying user: ${userId}`);

      // Get stored embedding
      const storedEmbedding = await getUserEmbedding(userId);
      if (!storedEmbedding) {
        throw new Error(`User ${userId} not found in database`);
      }

      // Generate new embedding from verification image
      const newEmbedding = await this.createEmbedding(imageBuffer);

      // Calculate similarity
      const similarity = calculateCosineSimilarity(
        storedEmbedding,
        newEmbedding
      );
      const isMatch = similarity >= this.similarityThreshold;

      console.log(
        `📊 Similarity: ${similarity.toFixed(4)}, Threshold: ${
          this.similarityThreshold
        }, Match: ${isMatch}`
      );

      return {
        success: true,
        isMatch: isMatch,
        similarity: parseFloat(similarity.toFixed(4)),
        threshold: this.similarityThreshold,
        userId: userId,
      };
    } catch (error) {
      console.error(`❌ User verification failed for ${userId}:`, error);
      throw error;
    }
  }

  // Compare two embeddings directly
  async compareEmbeddings(imageBuffer, storedEmbeddingArray) {
    try {
      console.log("🔍 Comparing embeddings directly...");

      // Validate stored embedding
      if (
        !Array.isArray(storedEmbeddingArray) ||
        storedEmbeddingArray.length === 0
      ) {
        throw new Error(
          "Invalid stored embedding format. Expected non-empty array."
        );
      }

      // Generate new embedding from image
      const newEmbedding = await this.createEmbedding(imageBuffer);

      // Validate embedding dimensions match
      if (newEmbedding.length !== storedEmbeddingArray.length) {
        throw new Error(
          `Embedding dimension mismatch. New: ${newEmbedding.length}, Stored: ${storedEmbeddingArray.length}`
        );
      }

      // Calculate similarity
      const similarity = calculateCosineSimilarity(
        storedEmbeddingArray,
        newEmbedding
      );
      const isMatch = similarity >= this.similarityThreshold;

      console.log(
        `📊 Similarity: ${similarity.toFixed(4)}, Threshold: ${
          this.similarityThreshold
        }, Match: ${isMatch}`
      );

      return {
        success: true,
        isMatch: isMatch,
        similarity: parseFloat(similarity.toFixed(4)),
        threshold: this.similarityThreshold,
      };
    } catch (error) {
      console.error("❌ Embedding comparison failed:", error);
      throw error;
    }
  }

  // Parse stored embedding from string format
  parseStoredEmbedding(embeddingString) {
    try {
      // Handle different input formats
      let embedding;

      if (typeof embeddingString === "string") {
        // Try to parse as JSON
        embedding = JSON.parse(embeddingString);
      } else if (Array.isArray(embeddingString)) {
        // Already an array
        embedding = embeddingString;
      } else {
        throw new Error("Invalid embedding format");
      }

      // Validate array
      if (!Array.isArray(embedding)) {
        throw new Error("Parsed embedding is not an array");
      }

      // Validate all elements are numbers
      const invalidElements = embedding.filter(
        (val) => typeof val !== "number" || isNaN(val)
      );
      if (invalidElements.length > 0) {
        throw new Error("Embedding contains non-numeric values");
      }

      // Validate reasonable embedding size (typically 128, 256, 512, etc.)
      if (embedding.length < 64 || embedding.length > 2048) {
        throw new Error(
          `Unusual embedding size: ${embedding.length}. Expected between 64-2048 dimensions.`
        );
      }

      console.log(`✅ Successfully parsed ${embedding.length}D embedding`);
      return embedding;
    } catch (error) {
      console.error("❌ Failed to parse stored embedding:", error);
      throw new Error(`Invalid embedding format: ${error.message}`);
    }
  }

  // Get comprehensive service statistics
  getServiceInfo() {
    const detectionInfo = faceDetection.getDetectionInfo();

    return {
      service: "Face Verification Microservice",
      version: "1.0.0",
      model: {
        name: "ArcFace ONNX",
        embeddingDimensions: "512D",
        targetImageSize: detectionInfo.targetSize,
        modelLoaded: isModelLoaded(),
      },
      validation: {
        similarityThreshold: this.similarityThreshold,
        supportedFormats: detectionInfo.supportedFormats,
        imageRequirements: {
          minResolution: detectionInfo.minResolution,
          maxResolution: detectionInfo.maxResolution,
          maxFileSize: detectionInfo.maxFileSize,
          minFaceSize: detectionInfo.minFaceSize,
          maxFaces: detectionInfo.maxFaces,
        },
        qualityChecks: detectionInfo.validations,
      },
      features: [
        "Face detection and validation",
        "Blur/sharpness detection",
        "Lighting condition analysis",
        "Multiple face detection",
        "Image quality validation",
        "Face size and position validation",
      ],
    };
  }

  // Update similarity threshold
  setSimilarityThreshold(threshold) {
    if (typeof threshold !== "number" || threshold < 0 || threshold > 1) {
      throw new Error("Similarity threshold must be a number between 0 and 1");
    }

    this.similarityThreshold = threshold;
    console.log(`⚙️ Similarity threshold updated to: ${threshold}`);
  }
}

module.exports = new EmbeddingService();
