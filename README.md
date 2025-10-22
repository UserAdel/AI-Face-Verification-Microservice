# AI Face-Verification Microservice

🚀 A complete, standalone Node.js microservice for face verification using ArcFace ONNX model and PostgreSQL.

## 🎯 Overview

This microservice provides real-world face verification capabilities through two main endpoints:

- **`/encode`** - Register users by generating 512-dimensional face embeddings
- **`/compare`** - Verify faces against stored embeddings using cosine similarity

## 🔧 Tech Stack

- **Runtime**: Node.js (>=16.0.0)
- **Framework**: Express.js
- **Database**: PostgreSQL
- **AI Model**: ArcFace ONNX (from Hugging Face)
- **Image Processing**: Sharp, TensorFlow.js
- **Model Inference**: ONNX Runtime

## 📋 Prerequisites

1. **Node.js** (version 16 or higher)
2. **PostgreSQL** (version 12 or higher)
3. **Git** (for cloning the repository)
4. **Git LFS** (for handling the large ONNX model file)

## 🚀 Installation & Setup

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd face-verification-microservice
```

### Step 2: Install Git LFS (Required for Model File)

```bash
# Install Git LFS if not already installed
git lfs install

# Pull the large model file
git lfs pull
```

### Step 3: Install Dependencies

```bash
# Install all required Node.js packages
npm install

# Verify installation
npm list --depth=0
```

### Step 4: Setup PostgreSQL Database

```sql
-- Create database
CREATE DATABASE face_verification;

-- Create user (optional)
CREATE USER face_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE face_verification TO face_user;
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=face_verification
DB_USER=postgres
DB_PASSWORD=your_password

# Server Configuration
PORT=3000
NODE_ENV=development

# Face Recognition Configuration
SIMILARITY_THRESHOLD=0.6
MODEL_PATH=./arcface.onnx
```

### 5. Verify Model File

Ensure the ArcFace ONNX model is in the root directory:

```bash
ls -la arcface.onnx
# Should show: arcface.onnx (approximately 130MB)
```

### 6. Start the Service

```bash
# Development mode with auto-reload
npm run dev

# Production mode
npm start
```

## 🔗 Model Information

**ArcFace ONNX Model**: [garavv/arcface-onnx](https://huggingface.co/garavv/arcface-onnx)

- **Input**: 112×112×3 RGB images
- **Output**: 512-dimensional face embeddings
- **Architecture**: ResNet-based with ArcFace loss
- **Accuracy**: State-of-the-art face recognition performance

## 📡 API Endpoints

### 🔐 POST `/api/encode`

**Description**: Generate face embedding for user registration

**Request**:

```bash
curl -X POST http://localhost:3000/api/encode \
  -F "image=@path/to/face_image.jpg" \
  -F "userId=john_doe_123"
```

**Response**:

```json
{
  "success": true,
  "embedding": [0.0123, -0.0456, ..., 0.0789],
  "userId": "john_doe_123",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "imageInfo": {
    "originalName": "face_image.jpg",
    "size": 245760,
    "mimeType": "image/jpeg"
  }
}
```

### 🔍 POST `/api/compare`

**Description**: Verify face against stored embedding

**Request**:

```bash
curl -X POST http://localhost:3000/api/compare \
  -F "image=@path/to/verification_image.jpg" \
  -F 'storedEmbedding=[0.0123,-0.0456,...,0.0789]'
```

**Response**:

```json
{
  "success": true,
  "isMatch": true,
  "similarity": 0.9218,
  "threshold": 0.6,
  "timestamp": "2024-01-15T10:35:00.000Z",
  "imageInfo": {
    "originalName": "verification_image.jpg",
    "size": 198432,
    "mimeType": "image/jpeg"
  },
  "embeddingInfo": {
    "storedDimensions": 512,
    "newDimensions": "N/A"
  }
}
```

### 📊 GET `/api/info`

**Description**: Get service information and configuration

**Response**:

```json
{
  "success": true,
  "service": "Face Verification Microservice",
  "version": "1.0.0",
  "similarityThreshold": 0.6,
  "modelLoaded": true,
  "supportedFormats": ["jpeg", "jpg", "png", "webp"],
  "targetImageSize": "112x112",
  "embeddingDimensions": "512D (ArcFace)"
}
```

### ❤️ GET `/api/health`

**Description**: Health check endpoint

## 🧪 Complete Workflow Testing Guide

### 🚀 Quick Start Testing

1. **Start the service**: `npm start`
2. **Verify health**: `curl http://localhost:3000/health`
3. **Check service info**: `curl http://localhost:3000/api/info`

### 📋 Full Workflow Test Cases

#### **Test Case 1: User Registration (Encode)**

**Purpose**: Register a new user and generate face embedding

```bash
# Test user registration
curl -X POST http://localhost:3000/api/encode \
  -F "image=@test_images/person1_clear.jpg" \
  -F "userId=john_doe_001"
```

**Expected Response**:

```json
{
  "success": true,
  "embedding": [0.0123, -0.0456, ...],
  "userId": "john_doe_001",
  "stored": true,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

**Save the embedding array** for the next test!

#### **Test Case 2: Successful Face Verification**

**Purpose**: Verify the same person with a different photo

```bash
# Use embedding from Test Case 1
curl -X POST http://localhost:3000/api/compare \
  -F "image=@test_images/person1_different_angle.jpg" \
  -F 'storedEmbedding=[0.0123,-0.0456,0.0789,...]'
```

**Expected Response**:

```json
{
  "success": true,
  "isMatch": true,
  "similarity": 0.8542,
  "threshold": 0.6
}
```

#### **Test Case 3: Failed Verification (Different Person)**

**Purpose**: Verify that different people are correctly rejected

```bash
curl -X POST http://localhost:3000/api/compare \
  -F "image=@test_images/person2_clear.jpg" \
  -F 'storedEmbedding=[0.0123,-0.0456,0.0789,...]'
```

**Expected Response**:

```json
{
  "success": true,
  "isMatch": false,
  "similarity": 0.3421,
  "threshold": 0.6
}
```

#### **Test Case 4: Poor Image Quality Rejection**

**Purpose**: Test image quality validation

```bash
# Test with low resolution image
curl -X POST http://localhost:3000/api/encode \
  -F "image=@test_images/low_resolution.jpg"

# Test with dark/poor lighting
curl -X POST http://localhost:3000/api/encode \
  -F "image=@test_images/dark_image.jpg"

# Test with blurry image
curl -X POST http://localhost:3000/api/encode \
  -F "image=@test_images/blurry_image.jpg"
```

**Expected Response**:

```json
{
  "success": false,
  "error": "Image resolution too low: 150x100. Minimum required: smaller dimension ≥150px and larger dimension ≥200px"
}
```

#### **Test Case 5: Multiple Faces Rejection**

**Purpose**: Ensure only single-person images are accepted

```bash
curl -X POST http://localhost:3000/api/encode \
  -F "image=@test_images/group_photo.jpg"
```

**Expected Response**:

```json
{
  "success": false,
  "error": "Multiple faces detected (3). Please provide an image with only one person."
}
```

#### **Test Case 6: Invalid File Format**

**Purpose**: Test file format validation

```bash
# Test with unsupported format
curl -X POST http://localhost:3000/api/encode \
  -F "image=@test_images/document.pdf"

# Test with text file
curl -X POST http://localhost:3000/api/encode \
  -F "image=@test_images/readme.txt"
```

**Expected Response**:

```json
{
  "success": false,
  "error": "Invalid file type. Allowed types: image/jpeg, image/jpg, image/png, image/webp"
}
```

#### **Test Case 7: No Face Detection**

**Purpose**: Test behavior when no face is found

```bash
curl -X POST http://localhost:3000/api/encode \
  -F "image=@test_images/landscape.jpg"
```

**Expected Response**:

```json
{
  "success": false,
  "error": "No face detected in the image. Please ensure a clear, front-facing photo with good lighting."
}
```

### 📋 Postman Collection Testing

**Import the provided Postman collection** for GUI-based testing:

1. Open Postman
2. Import `postman_Collection.json`
3. Set environment variable: `baseUrl = http://localhost:3000`
4. Run the collection tests in order

### 📈 Performance Testing

```bash
# Test concurrent requests
for i in {1..10}; do
  curl -X POST http://localhost:3000/api/encode \
    -F "image=@test_images/person1_clear.jpg" \
    -F "userId=load_test_$i" &
done
wait
```

### ✅ Success Criteria

- ✅ **Health Check**: Service responds with status "OK"
- ✅ **Registration**: Successfully generates 512D embeddings
- ✅ **Database Storage**: Embeddings saved to PostgreSQL
- ✅ **Verification**: Correct similarity calculations
- ✅ **Security**: Rejects multiple faces and poor quality images
- ✅ **Error Handling**: Proper error messages for all failure cases
- ✅ **Performance**: <500ms response time for face processing

## 🛠️ Advanced Configuration

### Similarity Threshold Tuning

- **0.4-0.5**: More permissive (higher false positives)
- **0.6-0.7**: Balanced (recommended)
- **0.8-0.9**: Strict (higher false negatives)

## 📊 Error Handling

### Common Error Responses

```json
{
  "success": false,
  "error": "No face detected in the image"
}
```

```json
{
  "success": false,
  "error": "Image too dark - poor lighting conditions detected"
}
```

```json
{
  "success": false,
  "error": "Invalid stored embedding format"
}
```

## 🐛 Troubleshooting

### Model Loading Issues

```bash
# Check model file
ls -la arcface.onnx

# Verify file integrity
file arcface.onnx
```

### Database Connection Issues

```bash
# Test PostgreSQL connection
psql -h localhost -U postgres -d face_verification
```
