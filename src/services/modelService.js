const ort = require("onnxruntime-node");
const path = require("path");
const fs = require("fs");

let session = null;

// Initialize the ONNX model
async function initializeModel() {
  try {
    const modelPath =
      process.env.MODEL_PATH ||
      path.join(__dirname, "../../models/arcface.onnx");

    // Check if model file exists
    if (!fs.existsSync(modelPath)) {
      throw new Error(`Model file not found at: ${modelPath}`);
    }

    console.log(`Loading model from: ${modelPath}`);

    // Create inference session
    session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ["cpu"], // Use CPU provider for compatibility
      graphOptimizationLevel: "all",
    });

    console.log("✅ ArcFace ONNX model loaded successfully");
    console.log("📊 Model input shape:", session.inputNames);
    console.log("📊 Model output shape:", session.outputNames);

    return session;
  } catch (error) {
    console.error("❌ Failed to load ONNX model:", error);
    throw error;
  }
}

// Generate embedding from preprocessed face image
async function generateEmbedding(imageData) {
  try {
    if (!session) {
      throw new Error("Model not initialized. Call initializeModel() first.");
    }

    // Validate input dimensions
    if (!imageData || imageData.length !== 112 * 112 * 3) {
      throw new Error("Invalid input data. Expected 112x112x3 RGB image data.");
    }

    // Reshape data to [1, 112, 112, 3] format (NHWC)
    const batchSize = 1;
    const height = 112;
    const width = 112;
    const channels = 3;

    const inputTensor = new Float32Array(batchSize * height * width * channels);

    // Keep HWC format and normalize to [-1, 1]
    for (let i = 0; i < imageData.length; i++) {
      // Normalize pixel values from [0, 255] to [-1, 1]
      inputTensor[i] = imageData[i] / 127.5 - 1.0;
    }

    // Create tensor with NHWC shape
    const tensor = new ort.Tensor("float32", inputTensor, [
      batchSize,
      height,
      width,
      channels,
    ]);

    // Run inference
    const feeds = {};
    feeds[session.inputNames[0]] = tensor;

    const results = await session.run(feeds);
    const outputTensor = results[session.outputNames[0]];

    // Extract embedding (should be 512-dimensional)
    const embedding = Array.from(outputTensor.data);

    // Normalize embedding (L2 normalization)
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    const normalizedEmbedding = embedding.map((val) => val / norm);

    console.log(`✅ Generated ${normalizedEmbedding.length}D embedding`);
    return normalizedEmbedding;
  } catch (error) {
    console.error("❌ Error generating embedding:", error);
    throw error;
  }
}

// Get model info
function getModelInfo() {
  if (!session) {
    return null;
  }

  return {
    inputNames: session.inputNames,
    outputNames: session.outputNames,
    inputShape: session.inputNames.map((name) => {
      const input = session.inputMetadata[name];
      return {
        name,
        type: input.type,
        dims: input.dims,
      };
    }),
    outputShape: session.outputNames.map((name) => {
      const output = session.outputMetadata[name];
      return {
        name,
        type: output.type,
        dims: output.dims,
      };
    }),
  };
}

// Check if model is loaded
function isModelLoaded() {
  return session !== null;
}

// Dispose model (cleanup)
async function disposeModel() {
  if (session) {
    await session.release();
    session = null;
    console.log("✅ Model disposed successfully");
  }
}

module.exports = {
  initializeModel,
  generateEmbedding,
  getModelInfo,
  isModelLoaded,
  disposeModel,
};
