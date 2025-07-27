# Docker Integration with npm

This document outlines how to use the Docker npm scripts for development.

## Prerequisites

- Docker and Docker Desktop installed and running
- Node.js and npm installed
- WSL2 properly configured (for Windows users)

## Available npm Scripts

### Root Project Level

```bash
# Start local development environment
npm run docker:dev              # Start containers in interactive mode
npm run docker:dev:detached     # Start containers in detached mode

# Stop local development environment
npm run docker:stop

# Building images
npm run docker:build            # Build all images
npm run docker:rebuild          # Rebuild all images without cache

# Logs
npm run docker:logs             # Show logs from all containers
npm run docker:logs:backend     # Show logs from backend only
npm run docker:logs:frontend    # Show logs from frontend only
npm run docker:logs:db          # Show logs from database only

# Container shell access
npm run docker:shell:backend    # Get a bash shell in backend container
npm run docker:shell:frontend   # Get a shell in frontend container
npm run docker:shell:db         # Connect to PostgreSQL shell

# Utility commands
npm run docker:clean            # Clean unused Docker resources
npm run docker:restart          # Restart all containers
npm run docker:status           # Show status of all containers

# Convenience aliases
npm start                       # Same as docker:dev
npm run start:detached          # Same as docker:dev:detached
npm run stop                    # Same as docker:stop
```

### Frontend Level

```bash
# Docker image commands
npm run docker:build            # Build production frontend image
npm run docker:build:dev        # Build development frontend image
npm run docker:run              # Run production frontend container
npm run docker:run:dev          # Run development frontend container with volume mounts

# Docker Compose commands
npm run docker:dev              # Start only frontend container
npm run docker:compose:up       # Start all services in detached mode
npm run docker:compose:down     # Stop all services
npm run docker:compose:logs     # Show logs for all services
npm run docker:compose:build    # Build all services
npm run docker:compose:rebuild  # Rebuild all services without cache

# Utility commands
npm run docker:clean            # Clean unused Docker resources

# Development workflow
npm run start:dev               # Start Docker services, wait for API, then start dev server
npm run stop:dev                # Stop all Docker services
```

## Development Workflow

1. Start the local development environment:

```bash
npm run docker:dev:detached
```

2. View logs from containers:

```bash
npm run docker:logs
```

3. Access the frontend at http://localhost:3000

4. Access the API at http://localhost:8000

5. When finished, stop all containers:

```bash
npm run docker:stop
```

## Troubleshooting

If you encounter issues:

1. Check Docker status:

```bash
npm run docker:status
```

2. Clean unused Docker resources:

```bash
npm run docker:clean
```

3. Rebuild containers:

```bash
npm run docker:rebuild
```

4. Check logs for specific services:

```bash
npm run docker:logs:backend
# or
npm run docker:logs:frontend
```
