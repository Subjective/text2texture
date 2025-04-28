# Stage 1: Build the React application
# Use a specific LTS version for better reproducibility than 'latest'
FROM node:20-alpine AS builder
LABEL maintainer="Joshua Yin"
LABEL description="Build stage for Text2Texture Demo Frontend"

WORKDIR /app

# Copy package files and install dependencies
# Use npm ci for faster, more reliable builds in CI/CD environments
COPY package.json package-lock.json* ./
RUN npm ci

# Copy the rest of the application source code
COPY . .

# Build the application
# The build command comes from package.json -> "build": "tsc -b && vite build"
RUN npm run build

# Stage 2: Serve the application with Nginx
FROM nginx:stable-alpine
LABEL maintainer="Joshua Yin"
LABEL description="Runtime stage for Text2Texture Demo Frontend"

# Copy the built static assets from the builder stage
# Vite typically builds to a 'dist' directory
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy the custom Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80 for Nginx
EXPOSE 80

# Default command to start Nginx
CMD ["nginx", "-g", "daemon off;"]