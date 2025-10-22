/**
 * Utility functions for calculating similarity between face embeddings
 */

// Calculate cosine similarity between two vectors
function calculateCosineSimilarity(vectorA, vectorB) {
  try {
    // Validate inputs
    if (!Array.isArray(vectorA) || !Array.isArray(vectorB)) {
      throw new Error("Both vectors must be arrays");
    }

    if (vectorA.length !== vectorB.length) {
      throw new Error(
        `Vector dimensions must match. A: ${vectorA.length}, B: ${vectorB.length}`
      );
    }

    if (vectorA.length === 0) {
      throw new Error("Vectors cannot be empty");
    }

    // Calculate dot product
    let dotProduct = 0;
    for (let i = 0; i < vectorA.length; i++) {
      if (typeof vectorA[i] !== "number" || typeof vectorB[i] !== "number") {
        throw new Error(
          `Invalid vector element at index ${i}. Expected numbers.`
        );
      }
      dotProduct += vectorA[i] * vectorB[i];
    }

    // Calculate magnitudes
    let magnitudeA = 0;
    let magnitudeB = 0;

    for (let i = 0; i < vectorA.length; i++) {
      magnitudeA += vectorA[i] * vectorA[i];
      magnitudeB += vectorB[i] * vectorB[i];
    }

    magnitudeA = Math.sqrt(magnitudeA);
    magnitudeB = Math.sqrt(magnitudeB);

    // Avoid division by zero
    if (magnitudeA === 0 || magnitudeB === 0) {
      throw new Error("Cannot calculate similarity for zero vectors");
    }

    // Calculate cosine similarity
    const similarity = dotProduct / (magnitudeA * magnitudeB);

    // Clamp to [-1, 1] range due to floating point precision
    return Math.max(-1, Math.min(1, similarity));
  } catch (error) {
    console.error("❌ Cosine similarity calculation failed:", error);
    throw error;
  }
}

// Calculate Euclidean distance between two vectors
function calculateEuclideanDistance(vectorA, vectorB) {
  try {
    // Validate inputs
    if (!Array.isArray(vectorA) || !Array.isArray(vectorB)) {
      throw new Error("Both vectors must be arrays");
    }

    if (vectorA.length !== vectorB.length) {
      throw new Error(
        `Vector dimensions must match. A: ${vectorA.length}, B: ${vectorB.length}`
      );
    }

    // Calculate squared differences
    let sumSquaredDifferences = 0;
    for (let i = 0; i < vectorA.length; i++) {
      if (typeof vectorA[i] !== "number" || typeof vectorB[i] !== "number") {
        throw new Error(
          `Invalid vector element at index ${i}. Expected numbers.`
        );
      }
      const diff = vectorA[i] - vectorB[i];
      sumSquaredDifferences += diff * diff;
    }

    // Return Euclidean distance
    return Math.sqrt(sumSquaredDifferences);
  } catch (error) {
    console.error("❌ Euclidean distance calculation failed:", error);
    throw error;
  }
}

// Calculate Manhattan distance between two vectors
function calculateManhattanDistance(vectorA, vectorB) {
  try {
    // Validate inputs
    if (!Array.isArray(vectorA) || !Array.isArray(vectorB)) {
      throw new Error("Both vectors must be arrays");
    }

    if (vectorA.length !== vectorB.length) {
      throw new Error(
        `Vector dimensions must match. A: ${vectorA.length}, B: ${vectorB.length}`
      );
    }

    // Calculate sum of absolute differences
    let sumAbsoluteDifferences = 0;
    for (let i = 0; i < vectorA.length; i++) {
      if (typeof vectorA[i] !== "number" || typeof vectorB[i] !== "number") {
        throw new Error(
          `Invalid vector element at index ${i}. Expected numbers.`
        );
      }
      sumAbsoluteDifferences += Math.abs(vectorA[i] - vectorB[i]);
    }

    return sumAbsoluteDifferences;
  } catch (error) {
    console.error("❌ Manhattan distance calculation failed:", error);
    throw error;
  }
}

// Normalize vector to unit length (L2 normalization)
function normalizeVector(vector) {
  try {
    if (!Array.isArray(vector)) {
      throw new Error("Input must be an array");
    }

    if (vector.length === 0) {
      throw new Error("Vector cannot be empty");
    }

    // Calculate magnitude
    let magnitude = 0;
    for (let i = 0; i < vector.length; i++) {
      if (typeof vector[i] !== "number") {
        throw new Error(
          `Invalid vector element at index ${i}. Expected number.`
        );
      }
      magnitude += vector[i] * vector[i];
    }

    magnitude = Math.sqrt(magnitude);

    // Avoid division by zero
    if (magnitude === 0) {
      throw new Error("Cannot normalize zero vector");
    }

    // Normalize
    return vector.map((val) => val / magnitude);
  } catch (error) {
    console.error("❌ Vector normalization failed:", error);
    throw error;
  }
}

// Calculate multiple similarity metrics
function calculateAllSimilarities(vectorA, vectorB) {
  try {
    const cosineSimilarity = calculateCosineSimilarity(vectorA, vectorB);
    const euclideanDistance = calculateEuclideanDistance(vectorA, vectorB);
    const manhattanDistance = calculateManhattanDistance(vectorA, vectorB);

    // Convert distances to similarities (0-1 range)
    // For distances, smaller values indicate higher similarity
    const maxPossibleEuclidean = Math.sqrt(vectorA.length * 2); // Assuming normalized vectors
    const maxPossibleManhattan = vectorA.length * 2;

    const euclideanSimilarity = 1 - euclideanDistance / maxPossibleEuclidean;
    const manhattanSimilarity = 1 - manhattanDistance / maxPossibleManhattan;

    return {
      cosine: cosineSimilarity,
      euclidean: {
        distance: euclideanDistance,
        similarity: Math.max(0, euclideanSimilarity),
      },
      manhattan: {
        distance: manhattanDistance,
        similarity: Math.max(0, manhattanSimilarity),
      },
    };
  } catch (error) {
    console.error("❌ Multi-similarity calculation failed:", error);
    throw error;
  }
}

// Determine if two embeddings represent the same person
function isMatch(vectorA, vectorB, threshold = 0.6, metric = "cosine") {
  try {
    let similarity;

    switch (metric.toLowerCase()) {
      case "cosine":
        similarity = calculateCosineSimilarity(vectorA, vectorB);
        break;
      case "euclidean":
        const euclideanDistance = calculateEuclideanDistance(vectorA, vectorB);
        const maxDistance = Math.sqrt(vectorA.length * 2);
        similarity = 1 - euclideanDistance / maxDistance;
        break;
      case "manhattan":
        const manhattanDistance = calculateManhattanDistance(vectorA, vectorB);
        const maxManhattan = vectorA.length * 2;
        similarity = 1 - manhattanDistance / maxManhattan;
        break;
      default:
        throw new Error(`Unsupported similarity metric: ${metric}`);
    }

    return {
      isMatch: similarity >= threshold,
      similarity: similarity,
      threshold: threshold,
      metric: metric,
    };
  } catch (error) {
    console.error("❌ Match determination failed:", error);
    throw error;
  }
}

module.exports = {
  calculateCosineSimilarity,
  calculateEuclideanDistance,
  calculateManhattanDistance,
  normalizeVector,
  calculateAllSimilarities,
  isMatch,
};
