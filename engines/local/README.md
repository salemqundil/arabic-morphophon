# Arabic NLP Engine - Local Development Environment

This directory contains the configuration files and scripts needed to set up a local development environment for the Arabic NLP Engine.

## Prerequisites

- Docker and Docker Compose installed on your machine
- Git (for version control)

## Getting Started

### 1. Clone the repository (if you haven't already)

```bash
git clone <repository-url>
cd engines
```

### 2. Set up environment variables

Copy the example environment file and modify if needed:

```bash
cp local/.env.example local/.env.local
```

Edit `local/.env.local` to set any custom configuration variables.

### 3. Start the development environment

```bash
docker compose -f docker-compose.local.yml up -d
```

This will start:
- Backend API server with hot-reload at http://localhost:8000
- Frontend development server with hot-reload at http://localhost:3000
- PostgreSQL database at localhost:5432
- Redis cache at localhost:6379
- PgAdmin web interface at http://localhost:5050
- MailHog (for email testing) at http://localhost:8025

### 4. Access the services

- **API Documentation**: http://localhost:8000/docs
- **Frontend**: http://localhost:3000
- **PgAdmin**: http://localhost:5050
  - Email: admin@example.com
  - Password: admin
- **MailHog**: http://localhost:8025

### 5. Database Access

Default database credentials:
- Host: localhost
- Port: 5432
- Database: arabic_morphophon
- Username: arabic_morpho
- Password: dev_password

To connect using PgAdmin:
1. Open http://localhost:5050
2. Login with admin credentials
3. Add a new server connection:
   - Name: Local Development
   - Host: db
   - Port: 5432
   - Database: arabic_morphophon
   - Username: arabic_morpho
   - Password: dev_password

### 6. Development Workflow

#### Backend Development
- The backend code is mounted into the container, so changes will be automatically detected and the server will reload
- API endpoints are accessible at http://localhost:8000/api/
- Swagger documentation is available at http://localhost:8000/docs

#### Frontend Development
- The frontend code is mounted into the container with hot-reload enabled
- Changes to React components will be automatically reflected in the browser

### 7. Running Tests

To run backend tests:

```bash
docker compose -f docker-compose.local.yml exec backend pytest
```

To run frontend tests:

```bash
docker compose -f docker-compose.local.yml exec frontend npm test
```

### 8. Stopping the Development Environment

```bash
docker compose -f docker-compose.local.yml down
```

To completely remove all data (database, volumes):

```bash
docker compose -f docker-compose.local.yml down -v
```

## Troubleshooting

### Database Issues

If you encounter database initialization issues:

```bash
docker compose -f docker-compose.local.yml exec db bash
psql -U arabic_morpho -d arabic_morphophon
```

This will give you a PostgreSQL shell to run queries and check the database state.

### Backend Container Issues

To inspect logs:

```bash
docker compose -f docker-compose.local.yml logs backend
```

To get a shell in the backend container:

```bash
docker compose -f docker-compose.local.yml exec backend bash
```

### Frontend Container Issues

To inspect logs:

```bash
docker compose -f docker-compose.local.yml logs frontend
```

To get a shell in the frontend container:

```bash
docker compose -f docker-compose.local.yml exec frontend sh
```

## Additional Development Tools

### Database Migrations

If you need to run database migrations manually:

```bash
docker compose -f docker-compose.local.yml exec backend alembic upgrade head
```

### Adding Dependencies

For backend Python dependencies:

```bash
docker compose -f docker-compose.local.yml exec backend poetry add <package-name>
```

For frontend dependencies:

```bash
docker compose -f docker-compose.local.yml exec frontend npm install <package-name>
```
