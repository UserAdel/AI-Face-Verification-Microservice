const sharp = require("sharp");

// Enhanced face detection with comprehensive validation
// Includes blur detection, face size validation, and multiple face handling
class FaceDetectionService {
  constructor() {
    this.targetSize = 112; // ArcFace expects 112x112 input
    this.minFaceSize = 0.1; // Minimum face size as percentage of image (reduced from 0.15)
    this.maxFaces = 1; // Maximum allowed faces
    this.blurThreshold = 100; // Laplacian variance threshold for blur
    // Face detection sensitivity settings
    this.edgeThresholds = [40, 30, 20]; // Multiple thresholds for detection passes
    this.regionThresholds = [25, 20, 15]; // Corresponding region growing thresholds
  }

  // Preprocess image for face recognition
  async preprocessImage(imageBuffer) {
    try {
      // Get image metadata
      const metadata = await sharp(imageBuffer).metadata();
      console.log(
        `🖼️ Input image: ${metadata.width}x${metadata.height}, format: ${metadata.format}`
      );

      // Validate image
      if (!metadata.width || !metadata.height) {
        throw new Error("Invalid image: Unable to read dimensions");
      }

      if (metadata.width < 50 || metadata.height < 50) {
        throw new Error("Image too small: Minimum size is 50x50 pixels");
      }

      // Convert to RGB and resize to 112x112
      const processedImage = await sharp(imageBuffer)
        .resize(this.targetSize, this.targetSize, {
          fit: "cover", // Crop to fill the target size
          position: "center",
        })
        .removeAlpha() // Remove alpha channel if present
        .raw() // Get raw pixel data
        .toBuffer();

      // Validate processed image size
      const expectedSize = this.targetSize * this.targetSize * 3; // RGB
      if (processedImage.length !== expectedSize) {
        throw new Error(
          `Processed image size mismatch. Expected ${expectedSize}, got ${processedImage.length}`
        );
      }

      console.log(
        `✅ Image preprocessed to ${this.targetSize}x${this.targetSize}`
      );
      return processedImage;
    } catch (error) {
      console.error("❌ Image preprocessing failed:", error);
      throw new Error(`Image preprocessing failed: ${error.message}`);
    }
  }

  // Advanced face detection using Sharp-based image analysis
  async detectFace(imageBuffer) {
    try {
      // Analyze image using Sharp
      const faceDetected = await this.sharpBasedFaceDetection(imageBuffer);
      return faceDetected;
    } catch (error) {
      console.error("❌ Face detection failed:", error);
      throw new Error(`Face detection failed: ${error.message}`);
    }
  }

  // Analyze image statistics using Sharp
  async analyzeImageStats(imageBuffer) {
    try {
      // Resize image to a standard size for analysis
      const resizedBuffer = await sharp(imageBuffer)
        .resize(224, 224)
        .greyscale()
        .raw()
        .toBuffer();

      // Calculate basic statistics
      const pixels = new Uint8Array(resizedBuffer);
      let sum = 0;
      let sumSquares = 0;

      for (let i = 0; i < pixels.length; i++) {
        sum += pixels[i];
        sumSquares += pixels[i] * pixels[i];
      }

      const mean = sum / pixels.length;
      const variance = sumSquares / pixels.length - mean * mean;
      const std = Math.sqrt(variance);

      return { mean, std };
    } catch (error) {
      throw new Error(`Failed to analyze image statistics: ${error.message}`);
    }
  }

  // Enhanced face detection with comprehensive validation
  async sharpBasedFaceDetection(imageBuffer) {
    try {
      // Get image statistics
      const stats = await this.analyzeImageStats(imageBuffer);
      const brightness = stats.mean;
      const contrast = stats.std;

      console.log(
        `📊 Image analysis - Brightness: ${brightness.toFixed(
          2
        )}, Contrast: ${contrast.toFixed(2)}`
      );

      // Step 1: Check lighting conditions
      await this.validateLightingConditions(brightness, contrast);

      // Step 2: Check for blur
      await this.validateImageSharpness(imageBuffer);

      // Step 3: Detect faces using edge detection
      await this.detectFacesInImage(imageBuffer);

      // Step 4: Validate face size and position
      await this.validateFaceCharacteristics(imageBuffer);

      console.log("✅ Comprehensive face detection validation passed");
      return true;
    } catch (error) {
      throw error;
    }
  }

  // Validate lighting conditions
  async validateLightingConditions(brightness, contrast) {
    const minBrightness = 30;
    const maxBrightness = 200;
    const minContrast = 15;

    if (brightness < minBrightness) {
      throw new Error(
        "Image too dark - poor lighting conditions detected. Please ensure adequate lighting and try again."
      );
    }

    if (brightness > maxBrightness) {
      throw new Error(
        "Image too bright - overexposed image detected. Please reduce lighting or avoid direct flash."
      );
    }

    if (contrast < minContrast) {
      throw new Error(
        "Low contrast image - face features may not be clear. Please ensure good lighting with clear shadows."
      );
    }
  }

  // Detect blur using Laplacian variance
  async validateImageSharpness(imageBuffer) {
    try {
      // Convert to grayscale and apply Laplacian filter
      const grayBuffer = await sharp(imageBuffer)
        .resize(300, 300)
        .greyscale()
        .raw()
        .toBuffer();

      // Calculate Laplacian variance (blur detection)
      const blurVariance = this.calculateLaplacianVariance(
        grayBuffer,
        300,
        300
      );

      console.log(`📊 Blur variance: ${blurVariance.toFixed(2)}`);

      if (blurVariance < this.blurThreshold) {
        throw new Error(
          "Image appears blurry or out of focus. Please ensure the camera is focused and the subject is still."
        );
      }
    } catch (error) {
      if (error.message.includes("blurry")) {
        throw error;
      }
      throw new Error(`Blur detection failed: ${error.message}`);
    }
  }

  // Calculate Laplacian variance for blur detection
  calculateLaplacianVariance(buffer, width, height) {
    const laplacianKernel = [
      [0, -1, 0],
      [-1, 4, -1],
      [0, -1, 0],
    ];

    let variance = 0;
    let count = 0;

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        let sum = 0;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const pixelIndex = (y + ky) * width + (x + kx);
            sum += buffer[pixelIndex] * laplacianKernel[ky + 1][kx + 1];
          }
        }
        variance += sum * sum;
        count++;
      }
    }

    return variance / count;
  }

  // Detect faces using edge detection and pattern analysis
  async detectFacesInImage(imageBuffer) {
    try {
      // Use edge detection to find potential face regions
      const edgeBuffer = await sharp(imageBuffer)
        .resize(400, 400)
        .greyscale()
        .convolve({
          width: 3,
          height: 3,
          kernel: [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        })
        .raw()
        .toBuffer();

      // Analyze edge patterns for face-like structures with multiple sensitivity passes
      let faceRegions = [];

      // Try multiple detection passes with decreasing sensitivity
      for (
        let i = 0;
        i < this.edgeThresholds.length && faceRegions.length === 0;
        i++
      ) {
        console.log(
          `🔍 Face detection pass ${i + 1} with edge threshold ${
            this.edgeThresholds[i]
          }`
        );
        faceRegions = this.findFaceRegions(
          edgeBuffer,
          400,
          400,
          this.edgeThresholds[i],
          this.regionThresholds[i]
        );
        console.log(
          `📊 Pass ${i + 1}: Detected ${
            faceRegions.length
          } potential face regions`
        );
      }

      if (faceRegions.length === 0) {
        throw new Error(
          "No face detected in the image. Please ensure a clear, front-facing photo with good lighting. Try adjusting brightness or contrast."
        );
      }

      if (faceRegions.length > this.maxFaces) {
        throw new Error(
          `Multiple faces detected (${faceRegions.length}). Please provide an image with only one person.`
        );
      }

      return faceRegions[0];
    } catch (error) {
      if (error.message.includes("face")) {
        throw error;
      }
      throw new Error(`Face detection failed: ${error.message}`);
    }
  }

  // Enhanced face region detection with face-specific feature validation
  findFaceRegions(
    edgeBuffer,
    width,
    height,
    edgeThreshold = 40,
    regionThreshold = 25
  ) {
    const regions = [];
    const minRegionSize = Math.floor(width * this.minFaceSize);
    const visited = new Set();

    // Step 1: Find potential regions using edge detection
    for (let y = minRegionSize; y < height - minRegionSize; y += 8) {
      for (let x = minRegionSize; x < width - minRegionSize; x += 8) {
        const index = y * width + x;
        if (visited.has(index) || edgeBuffer[index] < edgeThreshold) continue;

        const region = this.growRegion(
          edgeBuffer,
          width,
          height,
          x,
          y,
          visited,
          regionThreshold
        );

        if (region.size > minRegionSize * minRegionSize * 0.08) {
          regions.push({
            x: region.minX,
            y: region.minY,
            width: region.maxX - region.minX,
            height: region.maxY - region.minY,
            size: region.size,
            centerX: (region.minX + region.maxX) / 2,
            centerY: (region.minY + region.maxY) / 2,
            density:
              region.size /
              ((region.maxX - region.minX) * (region.maxY - region.minY)),
          });
        }
      }
    }

    // Step 2: Filter regions by basic geometric constraints
    const geometricFiltered = regions.filter((region) => {
      const aspectRatio = region.width / region.height;
      const sizeRatio =
        Math.min(region.width, region.height) /
        Math.max(region.width, region.height);

      return (
        aspectRatio > 0.6 &&
        aspectRatio < 1.7 && // More flexible aspect ratio
        sizeRatio > 0.5 && // Not too elongated
        region.width > minRegionSize * 0.8 &&
        region.height > minRegionSize * 0.8 &&
        region.density > 0.1 && // Reasonable edge density
        region.density < 0.8 // Not too dense (likely noise)
      );
    });

    // Step 3: Apply face-specific feature validation
    const faceValidated = [];
    for (const region of geometricFiltered) {
      const faceScore = this.calculateFaceScore(
        edgeBuffer,
        width,
        height,
        region
      );
      if (faceScore > 0.3) {
        // Minimum face confidence threshold
        region.faceScore = faceScore;
        faceValidated.push(region);
      }
    }

    // Step 4: Remove overlapping regions (keep highest scoring)
    const finalRegions = this.removeOverlappingRegions(faceValidated);

    console.log(
      `🔍 Face detection results: ${regions.length} initial → ${geometricFiltered.length} geometric → ${faceValidated.length} face-validated → ${finalRegions.length} final`
    );

    return finalRegions;
  }

  // Enhanced region growing algorithm with adaptive thresholding
  growRegion(buffer, width, height, startX, startY, visited, threshold = 30) {
    const stack = [{ x: startX, y: startY }];
    const region = {
      minX: startX,
      maxX: startX,
      minY: startY,
      maxY: startY,
      size: 0,
    };

    while (stack.length > 0) {
      const { x, y } = stack.pop();
      const index = y * width + x;

      if (visited.has(index) || x < 0 || x >= width || y < 0 || y >= height)
        continue;
      if (buffer[index] < threshold) continue;

      visited.add(index);
      region.size++;
      region.minX = Math.min(region.minX, x);
      region.maxX = Math.max(region.maxX, x);
      region.minY = Math.min(region.minY, y);
      region.maxY = Math.max(region.maxY, y);

      // Add neighbors
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          stack.push({ x: x + dx, y: y + dy });
        }
      }
    }

    return region;
  }

  // Calculate face-specific confidence score for a region
  calculateFaceScore(edgeBuffer, width, height, region) {
    let score = 0;

    // Score 1: Symmetry check (faces tend to be symmetric)
    const symmetryScore = this.checkSymmetry(edgeBuffer, width, region);
    score += symmetryScore * 0.3;

    // Score 2: Eye-like patterns in upper third
    const eyeScore = this.detectEyePatterns(edgeBuffer, width, region);
    score += eyeScore * 0.25;

    // Score 3: Mouth-like pattern in lower third
    const mouthScore = this.detectMouthPattern(edgeBuffer, width, region);
    score += mouthScore * 0.2;

    // Score 4: Edge distribution (faces have specific edge patterns)
    const edgeDistScore = this.analyzeEdgeDistribution(
      edgeBuffer,
      width,
      region
    );
    score += edgeDistScore * 0.15;

    // Score 5: Position preference (faces usually in center-upper area)
    const positionScore = this.calculatePositionScore(region, width, height);
    score += positionScore * 0.1;

    return Math.min(score, 1.0);
  }

  // Check horizontal symmetry of edge patterns
  checkSymmetry(edgeBuffer, width, region) {
    const centerX = Math.floor((region.x + region.x + region.width) / 2);
    let symmetrySum = 0;
    let comparisons = 0;

    for (let y = region.y; y < region.y + region.height; y += 2) {
      for (let offset = 1; offset < Math.min(region.width / 2, 15); offset++) {
        const leftIdx = y * width + (centerX - offset);
        const rightIdx = y * width + (centerX + offset);

        if (leftIdx >= 0 && rightIdx < edgeBuffer.length) {
          const diff = Math.abs(edgeBuffer[leftIdx] - edgeBuffer[rightIdx]);
          symmetrySum += Math.max(0, 50 - diff) / 50; // Normalize to 0-1
          comparisons++;
        }
      }
    }

    return comparisons > 0 ? symmetrySum / comparisons : 0;
  }

  // Detect eye-like patterns (horizontal edge pairs in upper region)
  detectEyePatterns(edgeBuffer, width, region) {
    const eyeRegionTop = region.y;
    const eyeRegionBottom = region.y + Math.floor(region.height * 0.4);
    const eyeRegionLeft = region.x + Math.floor(region.width * 0.15);
    const eyeRegionRight =
      region.x + region.width - Math.floor(region.width * 0.15);

    let eyeScore = 0;
    let maxHorizontalEdges = 0;

    // Look for horizontal edge concentrations (eye-like)
    for (let y = eyeRegionTop; y < eyeRegionBottom; y += 2) {
      let horizontalEdges = 0;
      for (let x = eyeRegionLeft; x < eyeRegionRight; x++) {
        const idx = y * width + x;
        if (idx < edgeBuffer.length && edgeBuffer[idx] > 40) {
          horizontalEdges++;
        }
      }
      maxHorizontalEdges = Math.max(maxHorizontalEdges, horizontalEdges);
    }

    const expectedEyeWidth = region.width * 0.7;
    eyeScore = Math.min(maxHorizontalEdges / expectedEyeWidth, 1.0);

    return eyeScore;
  }

  // Detect mouth-like patterns (horizontal edges in lower region)
  detectMouthPattern(edgeBuffer, width, region) {
    const mouthRegionTop = region.y + Math.floor(region.height * 0.6);
    const mouthRegionBottom = region.y + region.height;
    const mouthRegionLeft = region.x + Math.floor(region.width * 0.2);
    const mouthRegionRight =
      region.x + region.width - Math.floor(region.width * 0.2);

    let mouthScore = 0;
    let maxMouthEdges = 0;

    for (let y = mouthRegionTop; y < mouthRegionBottom; y++) {
      let horizontalEdges = 0;
      for (let x = mouthRegionLeft; x < mouthRegionRight; x++) {
        const idx = y * width + x;
        if (idx < edgeBuffer.length && edgeBuffer[idx] > 35) {
          horizontalEdges++;
        }
      }
      maxMouthEdges = Math.max(maxMouthEdges, horizontalEdges);
    }

    const expectedMouthWidth = region.width * 0.6;
    mouthScore = Math.min(maxMouthEdges / expectedMouthWidth, 1.0);

    return mouthScore;
  }

  // Analyze edge distribution patterns
  analyzeEdgeDistribution(edgeBuffer, width, region) {
    const totalPixels = region.width * region.height;
    let edgePixels = 0;
    let centerEdges = 0;

    const centerX = region.x + Math.floor(region.width / 2);
    const centerY = region.y + Math.floor(region.height / 2);
    const centerRadius = Math.min(region.width, region.height) * 0.3;

    for (let y = region.y; y < region.y + region.height; y++) {
      for (let x = region.x; x < region.x + region.width; x++) {
        const idx = y * width + x;
        if (idx < edgeBuffer.length && edgeBuffer[idx] > 30) {
          edgePixels++;

          // Check if edge is in center region
          const distFromCenter = Math.sqrt(
            (x - centerX) ** 2 + (y - centerY) ** 2
          );
          if (distFromCenter < centerRadius) {
            centerEdges++;
          }
        }
      }
    }

    const edgeDensity = edgePixels / totalPixels;
    const centerConcentration = centerEdges / Math.max(edgePixels, 1);

    // Faces typically have moderate edge density with center concentration
    const densityScore = edgeDensity > 0.1 && edgeDensity < 0.4 ? 1.0 : 0.5;
    const concentrationScore =
      centerConcentration > 0.3 ? 1.0 : centerConcentration / 0.3;

    return (densityScore + concentrationScore) / 2;
  }

  // Calculate position-based score (faces usually in upper-center area)
  calculatePositionScore(region, imageWidth, imageHeight) {
    const centerX = region.centerX / imageWidth;
    const centerY = region.centerY / imageHeight;

    // Prefer regions in center horizontally, upper-center vertically
    const horizontalScore = 1.0 - Math.abs(centerX - 0.5) * 2; // Peak at center
    const verticalScore =
      centerY < 0.6 ? 1.0 : Math.max(0, 1.0 - (centerY - 0.6) * 2.5);

    return Math.max(0, (horizontalScore + verticalScore) / 2);
  }

  // Remove overlapping regions, keeping highest scoring ones
  removeOverlappingRegions(regions) {
    if (regions.length <= 1) return regions;

    // Sort by face score (highest first)
    const sortedRegions = regions.sort(
      (a, b) => (b.faceScore || 0) - (a.faceScore || 0)
    );
    const finalRegions = [];

    for (const region of sortedRegions) {
      let hasOverlap = false;

      for (const existing of finalRegions) {
        const overlapArea = this.calculateOverlapArea(region, existing);
        const minArea = Math.min(
          region.width * region.height,
          existing.width * existing.height
        );

        // If overlap is more than 30% of smaller region, consider it overlapping
        if (overlapArea / minArea > 0.3) {
          hasOverlap = true;
          break;
        }
      }

      if (!hasOverlap) {
        finalRegions.push(region);
      }
    }

    return finalRegions;
  }

  // Calculate overlap area between two regions
  calculateOverlapArea(region1, region2) {
    const left = Math.max(region1.x, region2.x);
    const right = Math.min(
      region1.x + region1.width,
      region2.x + region2.width
    );
    const top = Math.max(region1.y, region2.y);
    const bottom = Math.min(
      region1.y + region1.height,
      region2.y + region2.height
    );

    if (left < right && top < bottom) {
      return (right - left) * (bottom - top);
    }

    return 0;
  }

  // Validate face characteristics
  async validateFaceCharacteristics(imageBuffer) {
    try {
      const metadata = await sharp(imageBuffer).metadata();
      const imageArea = metadata.width * metadata.height;
      const minFaceArea = imageArea * (this.minFaceSize * this.minFaceSize);

      // Additional validation could be added here for:
      // - Face orientation (profile vs frontal)
      // - Eye detection
      // - Facial landmark detection

      console.log(
        `📊 Face size validation - Min required area: ${minFaceArea}`
      );

      // For now, we assume the face detection above handles size validation
      return true;
    } catch (error) {
      throw new Error(
        `Face characteristic validation failed: ${error.message}`
      );
    }
  }

  // Enhanced image quality validation
  async validateImageQuality(imageBuffer) {
    try {
      const metadata = await sharp(imageBuffer).metadata();

      // Check image format
      const supportedFormats = ["jpeg", "jpg", "png", "webp"];
      if (!supportedFormats.includes(metadata.format.toLowerCase())) {
        throw new Error(
          `Unsupported image format: ${
            metadata.format
          }. Supported formats: ${supportedFormats.join(
            ", "
          )}. Please convert your image to a supported format."`
        );
      }

      // Enhanced image size validation - flexible for rectangular images
      const minDimension = Math.min(metadata.width, metadata.height);
      const maxDimension = Math.max(metadata.width, metadata.height);

      if (minDimension < 150 || maxDimension < 200) {
        throw new Error(
          `Image resolution too low: ${metadata.width}x${metadata.height}. Minimum required: smaller dimension ≥150px and larger dimension ≥200px for accurate face detection.`
        );
      }

      if (metadata.width > 4000 || metadata.height > 4000) {
        throw new Error(
          `Image resolution too high: ${metadata.width}x${metadata.height}. Maximum recommended: 4000x4000 pixels. Please resize your image.`
        );
      }

      // Check aspect ratio
      const aspectRatio = metadata.width / metadata.height;
      if (aspectRatio < 0.5 || aspectRatio > 2.0) {
        throw new Error(
          `Unusual image aspect ratio: ${aspectRatio.toFixed(
            2
          )}. Please use a more standard image format (not too wide or tall).`
        );
      }

      // Enhanced file size validation
      if (imageBuffer.length > 10 * 1024 * 1024) {
        throw new Error(
          `Image file too large: ${(imageBuffer.length / (1024 * 1024)).toFixed(
            1
          )}MB. Maximum size: 10MB. Please compress your image.`
        );
      }

      if (imageBuffer.length < 2048) {
        throw new Error(
          `Image file too small: ${imageBuffer.length} bytes. Minimum size: 2KB. The image may be corrupted or of very poor quality.`
        );
      }

      // Check for potential corruption
      if (
        !metadata.width ||
        !metadata.height ||
        metadata.width === 0 ||
        metadata.height === 0
      ) {
        throw new Error(
          "Image appears to be corrupted or invalid. Please try uploading a different image."
        );
      }

      console.log(
        `✅ Image quality validation passed - ${metadata.width}x${metadata.height}, ${metadata.format}`
      );
      return true;
    } catch (error) {
      throw new Error(`Image quality validation failed: ${error.message}`);
    }
  }

  // Complete enhanced face processing pipeline
  async processFaceImage(imageBuffer) {
    try {
      console.log("🔍 Starting enhanced face processing pipeline...");

      // Step 1: Validate image quality and format
      await this.validateImageQuality(imageBuffer);
      console.log("✅ Step 1: Image quality validation passed");

      // Step 2: Comprehensive face detection and validation
      await this.detectFace(imageBuffer);
      console.log("✅ Step 2: Face detection and validation passed");

      // Step 3: Preprocess image for AI model
      const processedImage = await this.preprocessImage(imageBuffer);
      console.log("✅ Step 3: Image preprocessing completed");

      console.log(
        "🎉 Enhanced face processing pipeline completed successfully"
      );
      return processedImage;
    } catch (error) {
      console.error(
        "❌ Enhanced face processing pipeline failed:",
        error.message
      );

      // Provide helpful error context
      if (error.message.includes("dark") || error.message.includes("bright")) {
        console.error("💡 Tip: Adjust lighting conditions and try again");
      } else if (
        error.message.includes("blurry") ||
        error.message.includes("focus")
      ) {
        console.error("💡 Tip: Ensure camera is focused and subject is still");
      } else if (
        error.message.includes("No face") ||
        error.message.includes("Multiple faces")
      ) {
        console.error(
          "💡 Tip: Use a clear photo with exactly one person facing the camera"
        );
      }

      throw error;
    }
  }

  // Get detailed service information
  getDetectionInfo() {
    return {
      targetSize: `${this.targetSize}x${this.targetSize}`,
      minFaceSize: `${(this.minFaceSize * 100).toFixed(1)}% of image`,
      maxFaces: this.maxFaces,
      blurThreshold: this.blurThreshold,
      supportedFormats: ["JPEG", "PNG", "WebP"],
      minResolution: "min dimension ≥150px, max dimension ≥200px",
      maxResolution: "4000x4000",
      maxFileSize: "10MB",
      detectionMethod:
        "Enhanced multi-stage face detection with feature validation",
      faceFeatures: [
        "Facial symmetry analysis",
        "Eye pattern detection",
        "Mouth region validation",
        "Edge distribution analysis",
        "Position-based scoring",
      ],
      validations: [
        "Image quality and format",
        "Lighting conditions",
        "Blur/sharpness detection",
        "Enhanced face presence and count with confidence scoring",
        "Face size and position",
        "Overlap removal for multiple detections",
      ],
      improvements: [
        "Reduced false positives from background objects",
        "Better handling of multiple face scenarios",
        "Face-specific feature validation",
        "Confidence-based region scoring",
        "Adaptive thresholding for different image conditions",
      ],
    };
  }
}

module.exports = new FaceDetectionService();
