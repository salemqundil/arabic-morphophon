# Arabic NLP Engine - Development Environment

## Overview

This project provides both production and local development environments for the Arabic NLP Engine. The local development setup is designed to be a simplified version of the production environment with additional developer tools and conveniences.

## Project Structure

```
├── local/                       # Local development configuration
│   ├── backend/                 # Backend development setup
│   │   └── Dockerfile.dev       # Development Dockerfile for backend
│   ├── database/                # Database setup for development
│   │   ├── init.sql             # Database initialization script
│   │   └── init-db.sh           # Database initialization shell script
│   ├── frontend/                # Frontend development setup
│   │   └── Dockerfile.dev       # Development Dockerfile for frontend
│   ├── .env.example             # Example environment variables
│   └── README.md                # Local development documentation
├── docker-compose.local.yml     # Docker Compose for local development
├── docker-compose.yml           # Docker Compose for production
├── local-dev.ps1                # PowerShell script for managing local development
├── deploy.py                    # Deployment script for production
└── build_production.py          # Production build script
```

## Local Development Environment

The local development environment includes:

1. **FastAPI Backend**
   - Hot-reload enabled for rapid development
   - Code mounted directly from host for real-time changes
   - API documentation accessible at http://localhost:8000/docs

2. **React Frontend**
   - Development server with hot module replacement
   - Code mounted directly from host for real-time changes
   - Running at http://localhost:3000

3. **Database**
   - PostgreSQL 15
   - Persistent data with Docker volumes
   - Automatic initialization with sample data
   - Accessible at localhost:5432

4. **Caching**
   - Redis 7
   - Persistent data with Docker volumes
   - Accessible at localhost:6379

5. **Developer Tools**
   - PgAdmin for database management (http://localhost:5050)
   - MailHog for email testing (http://localhost:8025)

## Getting Started

To start the local development environment:

```bash
./local-dev.ps1 start
```

For complete documentation, see the README.md file in the `local/` directory.

## Key Features of Local Development Setup

1. **Developer Experience**
   - Hot-reload for both backend and frontend
   - Persistent database and cache data
   - Direct code editing without rebuilds

2. **Debugging**
   - Debug endpoints with detailed logs
   - Interactive database access with pgAdmin
   - Email testing with MailHog

3. **Convenience**
   - PowerShell management script for common operations
   - Environment variable management
   - Database initialization with sample data

4. **Similarity to Production**
   - Same service architecture as production
   - Similar environment variables
   - Consistent database schema

## Managing the Environment

Use the `local-dev.ps1` script for managing your local environment:

```
Usage: ./local-dev.ps1 [command] [arguments]

Available commands:
  start         Start the local development environment
  stop          Stop the local development environment
  restart       Restart the local development environment
  remove        Remove the local environment and volumes
  logs [service] Show logs for all services or a specific service
  backend-shell Access a shell in the backend container
  frontend-shell Access a shell in the frontend container
  db-shell      Access the PostgreSQL shell
  status        Show the status of all services
  help          Show this help message
```
