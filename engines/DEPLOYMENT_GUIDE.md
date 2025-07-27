# Arabic Morphophonology System - Deployment Guide

This guide explains how to deploy the Arabic Morphophonology System using different deployment methods.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Backup and Restore](#backup-and-restore)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

Before deploying the system, ensure you have the following prerequisites:

- Python 3.8 or higher
- Docker and Docker Compose (for Docker deployment)
- Kubernetes cluster and kubectl (for Kubernetes deployment)
- Git

## Local Development

To set up the system for local development:

1. Clone the repository:

   ```bash
   git clone [repository-url]
   cd arabic-morphophonology
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -e .
   pip install -r backend/requirements.txt
   ```

4. Set up the backend structure:

   ```bash
   python infrastructure/scripts/setup_backend.py
   ```

5. Set up environment variables:

   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

6. Start the development server:

   ```bash
   cd backend
   uvicorn main:app --reload
   ```

7. In a separate terminal, start the frontend development server:

   ```bash
   cd frontend
   npm install
   npm start
   ```

## Docker Deployment

To deploy the system using Docker:

1. Make sure Docker and Docker Compose are installed.

2. Set up environment variables:

   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

3. Build and run the containers:

   ```bash
   python deploy.py docker --build
   ```

   Or manually:

   ```bash
   docker compose build
   docker compose up -d
   ```

4. Access the application at `http://localhost` (or the configured port).

### Configuration Options

You can configure the deployment by editing the `.env` file. Important settings include:

- `ENVIRONMENT`: Set to `development`, `staging`, or `production`
- `DB_USER`, `DB_PASSWORD`, `DB_NAME`: Database credentials
- `JWT_SECRET`: Secret key for JWT authentication
- `HTTP_PORT`, `HTTPS_PORT`: Ports for HTTP and HTTPS

## Kubernetes Deployment

To deploy the system to a Kubernetes cluster:

1. Make sure you have kubectl installed and configured to access your cluster.

2. Build and push the Docker images to a registry:

   ```bash
   docker build -t your-registry/arabic-morphophon-backend:latest -f infrastructure/docker/backend.Dockerfile .
   docker build -t your-registry/arabic-morphophon-frontend:latest -f infrastructure/docker/frontend.Dockerfile ./frontend

   docker push your-registry/arabic-morphophon-backend:latest
   docker push your-registry/arabic-morphophon-frontend:latest
   ```

3. Update the Kubernetes manifests with your registry information:

   ```bash
   sed -i 's|${DOCKER_REGISTRY}|your-registry|g' infrastructure/kubernetes/*.yaml
   ```

4. Deploy to Kubernetes:

   ```bash
   python deploy.py kubernetes
   ```

   Or manually:

   ```bash
   kubectl apply -f infrastructure/kubernetes/
   ```

5. Access the application using the Ingress hostname or NodePort service.

## Backup and Restore

### Creating a Backup

To create a backup of your system:

```bash
python deploy.py backup
```

This will create a backup of:

- Docker volumes
- Configuration files
- Data directory

### Restoring from a Backup

To restore your system from a backup:

```bash
python deploy.py restore /path/to/backup_directory
```

## Troubleshooting

### Common Issues

1. **Database Connection Issues**:
   - Check that the database container is running: `docker compose ps`
   - Verify database credentials in `.env` file
   - Check database logs: `docker compose logs db`

2. **API Connection Issues**:
   - Check that the backend container is running: `docker compose ps`
   - Verify API URL in frontend environment
   - Check backend logs: `docker compose logs backend`

3. **Kubernetes Pod Issues**:
   - Check pod status: `kubectl get pods -n arabic-morphophon`
   - Check pod logs: `kubectl logs -n arabic-morphophon <pod-name>`
   - Check pod events: `kubectl describe pod -n arabic-morphophon <pod-name>`

### Getting Support

If you encounter issues not covered in this guide, please:

1. Check the project's GitHub issues page
2. Contact the project maintainers
3. Create a new issue with detailed information about your problem

## License

This project is licensed under [LICENSE NAME] - see the LICENSE file for details.
