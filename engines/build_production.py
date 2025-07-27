#!/usr/bin/env python3
"""
Production Build Script for Arabic Morphophonology System

This script automates the process of building and deploying the system
in production environment.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
import shutil
import datetime
import time

# Base paths
BASE_DIR = Path(__file__).parent
PRODUCTION_DIR = BASE_DIR / "infrastructure" / "production"


def run_command(command, cwd=None, env=None, check=True):
    """Run a shell command"""
    print(f"Running: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            cwd=cwd or BASE_DIR,
            env={**os.environ, **(env or {})},
            check=check,
            text=True,
            capture_output=True,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(e.stdout)
        print(e.stderr, file=sys.stderr)
        if check:
            sys.exit(e.returncode)
        return e


def generate_ssl_certificates():
    """Generate self-signed SSL certificates for development/testing"""
    ssl_dir = PRODUCTION_DIR / "nginx" / "ssl"
    os.makedirs(ssl_dir, exist_ok=True)

    print("Generating self-signed SSL certificates for development/testing...")

    # Generate dhparam.pem
    if not (ssl_dir / "dhparam.pem").exists():
        run_command(
            ["openssl", "dhparam", "-out", str(ssl_dir / "dhparam.pem"), "2048"]
        )

    # Generate frontend certificates
    if (
        not (ssl_dir / "fullchain.pem").exists()
        or not (ssl_dir / "privkey.pem").exists()
    ):
        run_command(
            [
                "openssl",
                "req",
                "-x509",
                "-nodes",
                "-days",
                "365",
                "-newkey",
                "rsa:2048",
                "-keyout",
                str(ssl_dir / "privkey.pem"),
                "-out",
                str(ssl_dir / "fullchain.pem"),
                "-subj",
                "/CN=arabic-morphophon.example.com",
            ]
        )

    # Generate API certificates
    if (
        not (ssl_dir / "api-fullchain.pem").exists()
        or not (ssl_dir / "api-privkey.pem").exists()
    ):
        run_command(
            [
                "openssl",
                "req",
                "-x509",
                "-nodes",
                "-days",
                "365",
                "-newkey",
                "rsa:2048",
                "-keyout",
                str(ssl_dir / "api-privkey.pem"),
                "-out",
                str(ssl_dir / "api-fullchain.pem"),
                "-subj",
                "/CN=api.arabic-morphophon.example.com",
            ]
        )

    print("SSL certificates generated successfully.")


def setup_production_env():
    """Set up production environment"""
    # Copy production .env file if it doesn't exist
    if not (BASE_DIR / ".env").exists():
        shutil.copy(PRODUCTION_DIR / ".env.production", BASE_DIR / ".env")
        print("Created production .env file")


def build_docker_images():
    """Build Docker images for production"""
    print("Building Docker images for production...")

    # Build backend
    run_command(
        [
            "docker",
            "build",
            "-t",
            "arabic-morphophon-backend:prod",
            "-f",
            "infrastructure/docker/backend.Dockerfile",
            ".",
        ]
    )

    # Build frontend
    frontend_dir = BASE_DIR / "frontend"
    if frontend_dir.exists():
        run_command(
            [
                "docker",
                "build",
                "-t",
                "arabic-morphophon-frontend:prod",
                "-f",
                "../infrastructure/docker/frontend.Dockerfile",
                ".",
            ],
            cwd=frontend_dir,
        )
    else:
        print("Warning: Frontend directory not found. Skipping frontend build.")

    print("Docker images built successfully.")


def deploy_production():
    """Deploy the system in production mode"""
    print("Deploying in production mode...")

    # Make sure the script is executable
    init_script = PRODUCTION_DIR / "db" / "init" / "00-init-db.sh"
    if init_script.exists():
        run_command(["chmod", "+x", str(init_script)])

    # Deploy with Docker Compose
    run_command(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.yml",
            "-f",
            str(PRODUCTION_DIR / "docker-compose.prod.yml"),
            "up",
            "-d",
        ]
    )

    print("Production deployment completed successfully.")


def stop_production():
    """Stop the production deployment"""
    print("Stopping production deployment...")

    run_command(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.yml",
            "-f",
            str(PRODUCTION_DIR / "docker-compose.prod.yml"),
            "down",
        ],
        check=False,
    )

    print("Production deployment stopped.")


def create_production_backup():
    """Create a backup of the production system"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BASE_DIR / f"backup_prod_{timestamp}"
    backup_dir.mkdir(exist_ok=True)

    print(f"Creating production backup in {backup_dir}")

    # Copy configuration files
    for config_file in [".env"]:
        if (BASE_DIR / config_file).exists():
            shutil.copy(BASE_DIR / config_file, backup_dir)
            print(f"Backed up {config_file}")

    # Backup Docker volumes
    try:
        # Get list of volumes
        result = run_command(
            ["docker", "volume", "ls", "--format", "{{.Name}}"], check=False
        )
        if result.returncode == 0:
            volumes = result.stdout.strip().split("\n")
            volumes = [
                v
                for v in volumes
                if v
                and (
                    v.startswith("arabic-morphophon")
                    or v.startswith("engines_postgres-data")
                    or v.startswith("engines_redis-data")
                )
            ]

            for volume in volumes:
                print(f"Backing up Docker volume: {volume}")
                backup_file = backup_dir / f"{volume}.tar"
                run_command(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "-v",
                        f"{volume}:/volume",
                        "-v",
                        f"{backup_dir.absolute()}:/backup",
                        "alpine",
                        "tar",
                        "cf",
                        f"/backup/{volume}.tar",
                        "-C",
                        "/volume",
                        ".",
                    ]
                )
    except (subprocess.SubprocessError, FileNotFoundError):
        print(
            "Warning: Could not backup Docker volumes. Make sure Docker is installed and running."
        )

    # Create database dumps
    try:
        print("Creating database dumps...")
        db_backup_dir = backup_dir / "db_dumps"
        db_backup_dir.mkdir(exist_ok=True)

        run_command(
            [
                "docker",
                "compose",
                "-f",
                "docker-compose.yml",
                "-f",
                str(PRODUCTION_DIR / "docker-compose.prod.yml"),
                "exec",
                "db",
                "pg_dump",
                "-U",
                "${DB_USER:-postgres}",
                "${DB_NAME:-arabic_morphophon}",
                "-f",
                "/tmp/db_dump.sql",
            ],
            check=False,
        )

        run_command(
            [
                "docker",
                "compose",
                "-f",
                "docker-compose.yml",
                "-f",
                str(PRODUCTION_DIR / "docker-compose.prod.yml"),
                "cp",
                "db:/tmp/db_dump.sql",
                str(db_backup_dir / "db_dump.sql"),
            ],
            check=False,
        )

    except (subprocess.SubprocessError, FileNotFoundError):
        print("Warning: Could not create database dumps.")

    print(f"Production backup completed: {backup_dir}")
    return backup_dir


def restore_production_backup(backup_dir):
    """Restore a production backup"""
    backup_dir = Path(backup_dir)
    if not backup_dir.exists() or not backup_dir.is_dir():
        print(f"Error: Backup directory {backup_dir} not found")
        sys.exit(1)

    print(f"Restoring from backup: {backup_dir}")

    # Stop containers if running
    stop_production()
    time.sleep(5)  # Give containers time to stop

    # Restore configuration files
    for config_file in [".env"]:
        backup_file = backup_dir / config_file
        if backup_file.exists():
            shutil.copy(backup_file, BASE_DIR / config_file)
            print(f"Restored {config_file}")

    # Restore Docker volumes
    try:
        volume_backups = list(backup_dir.glob("*.tar"))
        for volume_backup in volume_backups:
            volume_name = volume_backup.stem
            print(f"Restoring Docker volume: {volume_name}")

            # Check if volume exists, create if not
            result = run_command(
                ["docker", "volume", "inspect", volume_name], check=False
            )
            if result.returncode != 0:
                run_command(["docker", "volume", "create", volume_name])

            # Restore volume data
            run_command(
                [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{volume_name}:/volume",
                    "-v",
                    f"{volume_backup.absolute()}:/backup.tar",
                    "alpine",
                    "sh",
                    "-c",
                    "rm -rf /volume/* && tar xf /backup.tar -C /volume",
                ]
            )
    except (subprocess.SubprocessError, FileNotFoundError):
        print(
            "Warning: Could not restore Docker volumes. Make sure Docker is installed and running."
        )

    # Restore database if dump exists
    db_dump = backup_dir / "db_dumps" / "db_dump.sql"
    if db_dump.exists():
        print("Restoring database from dump...")
        # Start only the database container
        run_command(
            [
                "docker",
                "compose",
                "-f",
                "docker-compose.yml",
                "-f",
                str(PRODUCTION_DIR / "docker-compose.prod.yml"),
                "up",
                "-d",
                "db",
            ]
        )

        # Wait for database to be ready
        print("Waiting for database to be ready...")
        time.sleep(10)

        # Copy dump to container
        run_command(
            [
                "docker",
                "compose",
                "-f",
                "docker-compose.yml",
                "-f",
                str(PRODUCTION_DIR / "docker-compose.prod.yml"),
                "cp",
                str(db_dump),
                "db:/tmp/db_dump.sql",
            ]
        )

        # Restore dump
        run_command(
            [
                "docker",
                "compose",
                "-f",
                "docker-compose.yml",
                "-f",
                str(PRODUCTION_DIR / "docker-compose.prod.yml"),
                "exec",
                "db",
                "psql",
                "-U",
                "${DB_USER:-postgres}",
                "${DB_NAME:-arabic_morphophon}",
                "-f",
                "/tmp/db_dump.sql",
            ]
        )

        # Stop the database container
        run_command(
            [
                "docker",
                "compose",
                "-f",
                "docker-compose.yml",
                "-f",
                str(PRODUCTION_DIR / "docker-compose.prod.yml"),
                "stop",
                "db",
            ]
        )

    print("Backup restoration completed. You can now restart the system.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Build and deploy Arabic Morphophonology System in production"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands", required=True)

    # Setup command
    setup_parser = subparsers.add_parser(
        "setup", help="Set up the production environment"
    )
    setup_parser.add_argument(
        "--ssl", action="store_true", help="Generate self-signed SSL certificates"
    )

    # Build command
    build_parser = subparsers.add_parser(
        "build", help="Build Docker images for production"
    )

    # Deploy command
    deploy_parser = subparsers.add_parser(
        "deploy", help="Deploy the system in production mode"
    )
    deploy_parser.add_argument(
        "--build", action="store_true", help="Build Docker images before deploying"
    )

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the production deployment")

    # Backup command
    backup_parser = subparsers.add_parser(
        "backup", help="Create a backup of the production system"
    )

    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from a backup")
    restore_parser.add_argument("backup_dir", help="Backup directory to restore from")

    # Parse arguments
    args = parser.parse_args()

    # Execute the command
    if args.command == "setup":
        setup_production_env()
        if args.ssl:
            generate_ssl_certificates()
    elif args.command == "build":
        build_docker_images()
    elif args.command == "deploy":
        if args.build:
            build_docker_images()
        deploy_production()
    elif args.command == "stop":
        stop_production()
    elif args.command == "backup":
        create_production_backup()
    elif args.command == "restore":
        restore_production_backup(args.backup_dir)


if __name__ == "__main__":
    main()
