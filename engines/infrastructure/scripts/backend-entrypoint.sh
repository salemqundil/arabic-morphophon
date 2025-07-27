#!/bin/bash
# Backend entrypoint script for Arabic Morphophonology System

set -e

# Wait for dependent services
if [ "${WAIT_FOR_SERVICES}" = "true" ]; then
    echo "Waiting for database to be ready..."
    while ! nc -z ${DB_HOST:-db} ${DB_PORT:-5432}; do
        sleep 1
    done
    echo "Database is ready!"

    echo "Waiting for Redis to be ready..."
    while ! nc -z ${REDIS_HOST:-redis} ${REDIS_PORT:-6379}; do
        sleep 1
    done
    echo "Redis is ready!"
fi

# Run database migrations if needed
if [ "${RUN_MIGRATIONS}" = "true" ]; then
    echo "Running database migrations..."
    python -m backend.db.migrations
fi

# Create initial data if needed
if [ "${CREATE_INITIAL_DATA}" = "true" ]; then
    echo "Creating initial data..."
    python -m backend.db.seed
fi

# Execute the command (likely the application server)
echo "Starting the application..."
exec "$@"
