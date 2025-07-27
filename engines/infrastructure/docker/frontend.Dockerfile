# Frontend Dockerfile for Arabic Morphophonology System
FROM node:18-alpine as build

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy all frontend files
COPY . .

# Build argument for API URL
ARG API_URL=http://localhost:8000/api
ENV REACT_APP_API_URL=$API_URL

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy build files from the build stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY nginx/default.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Start Nginx server
CMD ["nginx", "-g", "daemon off;"]
