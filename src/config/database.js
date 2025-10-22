const { Pool } = require("pg");

// Database connection pool
const pool = new Pool({
  host: process.env.DB_HOST || "localhost",
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || "face_verification",
  user: process.env.DB_USER || "postgres",
  password: process.env.DB_PASSWORD,
  ssl: process.env.DB_SSL === "true" ? { rejectUnauthorized: false } : false,
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 10000, // Increased from 2s to 10s
  acquireTimeoutMillis: 10000,
});

// Initialize database tables
async function initializeDatabase() {
  try {
    // Create users table for storing embeddings
    await pool.query(`
      CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(255) UNIQUE NOT NULL,
        embedding FLOAT8[] NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);

    // Create index on user_id for faster lookups
    await pool.query(`
      CREATE INDEX IF NOT EXISTS idx_users_user_id ON users(user_id)
    `);

    console.log("✅ Database initialized successfully");
  } catch (error) {
    console.error("❌ Database initialization failed:", error);
    throw error;
  }
}

// Store user embedding
async function storeUserEmbedding(userId, embedding) {
  try {
    const query = `
      INSERT INTO users (user_id, embedding, updated_at) 
      VALUES ($1, $2, CURRENT_TIMESTAMP)
      ON CONFLICT (user_id) 
      DO UPDATE SET 
        embedding = EXCLUDED.embedding,
        updated_at = CURRENT_TIMESTAMP
      RETURNING id, user_id, created_at, updated_at
    `;

    const result = await pool.query(query, [userId, embedding]);
    return result.rows[0];
  } catch (error) {
    console.error("Error storing user embedding:", error);
    throw error;
  }
}

// Get user embedding
async function getUserEmbedding(userId) {
  try {
    const query = "SELECT embedding FROM users WHERE user_id = $1";
    const result = await pool.query(query, [userId]);
    return result.rows[0]?.embedding || null;
  } catch (error) {
    console.error("Error getting user embedding:", error);
    throw error;
  }
}

// Get all users
async function getAllUsers() {
  try {
    const query =
      "SELECT user_id, created_at, updated_at FROM users ORDER BY created_at DESC";
    const result = await pool.query(query);
    return result.rows;
  } catch (error) {
    console.error("Error getting all users:", error);
    throw error;
  }
}

// Delete user
async function deleteUser(userId) {
  try {
    const query = "DELETE FROM users WHERE user_id = $1 RETURNING *";
    const result = await pool.query(query, [userId]);
    return result.rows[0] || null;
  } catch (error) {
    console.error("Error deleting user:", error);
    throw error;
  }
}

// Test database connection
async function testConnection() {
  try {
    const client = await pool.connect();
    await client.query("SELECT NOW()");
    client.release();
    console.log("✅ Database connection successful");
    return true;
  } catch (error) {
    console.error("❌ Database connection failed:", error);
    return false;
  }
}

module.exports = {
  pool,
  initializeDatabase,
  storeUserEmbedding,
  getUserEmbedding,
  getAllUsers,
  deleteUser,
  testConnection,
};
