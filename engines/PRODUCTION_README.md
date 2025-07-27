# Arabic Morphophonology System - Production Environment

This document provides information about the production environment for the Arabic Morphophonology System.

## Overview

The production environment is designed for high performance, reliability, and security. It includes:

- Optimized Docker container configurations
- Production-grade Nginx settings
- Database optimizations
- Monitoring and observability tools
- Enhanced security measures

## Directory Structure

```text
infrastructure/production/
├── db/                      # Database configuration
│   ├── conf/                # PostgreSQL configuration
│   └── init/                # Database initialization scripts
├── grafana/                 # Grafana configuration
│   └── provisioning/        # Grafana provisioning
├── nginx/                   # Nginx configuration
│   ├── conf.d/              # Site configurations
│   └── ssl/                 # SSL certificates (not stored in Git)
├── prometheus/              # Prometheus configuration
└── docker-compose.prod.yml  # Production Docker Compose override
```

## Getting Started

### Prerequisites

- Docker and Docker Compose installed
- SSL certificates (for production use)
- At least 4GB RAM and 2 CPU cores

### Setting Up the Production Environment

1. Run the setup command:

   ```bash
   python build_production.py setup
   ```

   For development/testing with self-signed certificates:

   ```bash
   python build_production.py setup --ssl
   ```

2. Build the Docker images:

   ```bash
   python build_production.py build
   ```

3. Deploy the system:

   ```bash
   python build_production.py deploy
   ```

   To build and deploy in one step:

   ```bash
   python build_production.py deploy --build
   ```

### Accessing the System

- Frontend: `https://arabic-morphophon.example.com`
- API: `https://api.arabic-morphophon.example.com`
- Grafana: `http://your-server-ip:3000`
- Prometheus: `http://your-server-ip:9090` (internal access only)

## Configuration

The production environment is configured through the `.env` file, which is copied from `.env.production` during setup.

Key configurations:

- `DB_USER`, `DB_PASSWORD`: Database credentials
- `JWT_SECRET`: Secret key for JWT authentication
- `BACKEND_REPLICAS`, `FRONTEND_REPLICAS`: Number of service replicas
- `*_CPU_LIMIT`, `*_MEM_LIMIT`: Resource limits for containers

## Monitoring

The production environment includes:

- Prometheus for metrics collection
- Grafana for visualization
- Container health checks
- Log aggregation

Default Grafana credentials:

- Username: admin
- Password: admin (change this immediately in production)

## Backup and Restore

### Creating a Backup

```bash
python build_production.py backup
```

This creates a backup in the `backup_prod_TIMESTAMP` directory, including:

- Docker volumes
- Database dumps
- Configuration files

### Restoring from a Backup

```bash
python build_production.py restore /path/to/backup_dir
```

## Security Considerations

The production environment includes several security enhancements:

- TLS encryption for all connections
- Secure HTTP headers
- Database connection encryption
- Restricted access to admin interfaces
- Container isolation

## Troubleshooting

Common issues:

1. **Cannot connect to services**
   - Check if containers are running: `docker compose ps`
   - Verify Nginx configuration
   - Check firewall settings

2. **Database connection issues**
   - Check database logs: `docker compose logs db`
   - Verify database credentials in `.env`

3. **SSL/TLS certificate errors**
   - Ensure certificates are properly installed in `nginx/ssl`
   - Check Nginx SSL configuration

## Additional Resources

- [Main Deployment Guide](./DEPLOYMENT_GUIDE.md)
- [Deployment Summary](./DEPLOYMENT_SUMMARY.md)
