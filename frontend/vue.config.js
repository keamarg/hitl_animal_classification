const { defineConfig } = require("@vue/cli-service");
const dotenv = require("dotenv");
const path = require("path");

// Load the environment variables from the shared .env file
dotenv.config({ path: path.resolve(__dirname, "../.env") });

module.exports = defineConfig({
  transpileDependencies: true,
  publicPath: process.env.NODE_ENV === "production" ? "/hitl" : "/",
  // Your existing configurations...
});
