# Arabic Morphophonology System - Infrastructure

This directory contains the deployment infrastructure for the Arabic Morphophonology System.

## Directory Structure

```text
infrastructure/
├── docker/                # Docker-related files
│   ├── backend.Dockerfile # Backend Docker image definition
│   └── frontend.Dockerfile # Frontend Docker image definition
├── kubernetes/            # Kubernetes manifests
│   ├── 00-namespace-config.yaml
│   ├── 01-secrets.yaml
│   ├── 02-backend-deployment.yaml
│   ├── 03-frontend-deployment.yaml
│   ├── 04-postgres-statefulset.yaml
│   ├── 05-redis-deployment.yaml
│   ├── 06-persistent-volumes.yaml
│   ├── 07-ingress.yaml
│   └── 08-nodeport-services.yaml
├── nginx/                 # Nginx configuration
│   ├── conf.d/            # Nginx site configuration
│   │   └── default.conf   # Default site config
│   ├── nginx.conf         # Main Nginx configuration
│   └── ssl/               # SSL certificates (not stored in Git)
└── scripts/               # Utility scripts
    ├── backend-entrypoint.sh # Backend container entrypoint
    └── setup_backend.py   # Backend directory structure setup
```

## Usage

The infrastructure is designed to support multiple deployment methods:

1. **Docker Compose**: For local development and single-server deployments
2. **Kubernetes**: For scalable production deployments

See the main [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md) for detailed instructions on how to use these files.

## Docker Configuration

The Docker setup includes:

- Multi-stage builds for optimized images
- Volume mounts for persistent data
- Network isolation for services
- Health checks for all services

## Kubernetes Configuration

The Kubernetes configuration includes:

- Namespace isolation
- ConfigMap and Secret management
- Deployment resources with health checks
- StatefulSet for the database
- Persistent volume claims for data
- Ingress for external access
- NodePort services for alternative access

## Development

To modify the infrastructure:

1. Update Docker files in `docker/` directory
2. Update Kubernetes manifests in `kubernetes/` directory
3. Test changes with the `deploy.py` script

## Best Practices

The infrastructure follows these best practices:

- Separation of code and configuration
- Environment variable management
- Proper resource allocation
- Health monitoring
- Security-focused configuration
- Scalable architecture
