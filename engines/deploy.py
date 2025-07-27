#!/usr/bin/env python3
"""
Arabic Morphophonology System Deployment Script

This script helps deploy the Arabic Morphophonology System to various environments.
It supports local Docker Compose, Kubernetes, and cloud environments.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
import shutil
import json
import datetime

# Base paths
BASE_DIR = Path(__file__).parent
INFRASTRUCTURE_DIR = BASE_DIR / "infrastructure"
DOCKER_DIR = INFRASTRUCTURE_DIR / "docker"
KUBERNETES_DIR = INFRASTRUCTURE_DIR / "kubernetes"
SCRIPTS_DIR = INFRASTRUCTURE_DIR / "scripts"


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


def docker_compose_deploy(args):
    """Deploy using Docker Compose"""
    if not Path("docker-compose.yml").exists():
        print("Error: docker-compose.yml not found")
        sys.exit(1)

    # Create .env file if it doesn't exist
    if not Path(".env").exists():
        if Path(".env.example").exists():
            print("Creating .env file from .env.example")
            shutil.copy(".env.example", ".env")
        else:
            print("Warning: .env.example not found, creating empty .env file")
            Path(".env").touch()

    # Build and start containers
    cmd = ["docker", "compose"]

    if args.build:
        cmd.extend(["build", "--no-cache"])
        run_command(cmd)
        cmd = ["docker", "compose"]  # Reset command

    cmd.append("up")

    if args.detach:
        cmd.append("-d")

    run_command(cmd)


def kubernetes_deploy(args):
    """Deploy to Kubernetes"""
    if not Path(KUBERNETES_DIR).exists():
        print("Error: Kubernetes directory not found")
        sys.exit(1)

    # Check for kubectl
    try:
        run_command(["kubectl", "version", "--client"], check=False)
    except FileNotFoundError:
        print("Error: kubectl not found. Please install kubectl.")
        sys.exit(1)

    # Check for current context
    result = run_command(["kubectl", "config", "current-context"], check=False)
    if result.returncode != 0:
        print(
            "Error: Could not get Kubernetes context. Make sure you're connected to a cluster."
        )
        sys.exit(1)

    print(f"Deploying to Kubernetes context: {result.stdout.strip()}")

    # Apply Kubernetes manifests
    manifests = sorted(Path(KUBERNETES_DIR).glob("*.yaml"))
    if not manifests:
        print("Error: No Kubernetes manifests found")
        sys.exit(1)

    for manifest in manifests:
        print(f"Applying {manifest}")
        run_command(["kubectl", "apply", "-f", str(manifest)])

    # Wait for deployments
    print("Waiting for deployments to be ready...")
    run_command(["kubectl", "get", "deployments"])


def create_backup(args):
    """Create a backup of the system"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"backup_{timestamp}")
    backup_dir.mkdir(exist_ok=True)

    print(f"Creating backup in {backup_dir}")

    # Backup Docker volumes if Docker is available
    try:
        # Get list of volumes
        result = run_command(
            ["docker", "volume", "ls", "--format", "{{.Name}}"], check=False
        )
        if result.returncode == 0:
            volumes = result.stdout.strip().split("\n")
            volumes = [v for v in volumes if v.startswith("arabic-morphophon")]

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

    # Backup configuration files
    for config_file in [".env", "docker-compose.yml"]:
        if Path(config_file).exists():
            shutil.copy(config_file, backup_dir)
            print(f"Backed up {config_file}")

    # Backup data directory if it exists
    data_dir = Path("data")
    if data_dir.exists() and data_dir.is_dir():
        data_backup_dir = backup_dir / "data"
        shutil.copytree(data_dir, data_backup_dir)
        print(f"Backed up data directory to {data_backup_dir}")

    print(f"Backup completed: {backup_dir}")
    return backup_dir


def restore_backup(args):
    """Restore a backup of the system"""
    backup_dir = Path(args.backup_dir)
    if not backup_dir.exists() or not backup_dir.is_dir():
        print(f"Error: Backup directory {backup_dir} not found")
        sys.exit(1)

    print(f"Restoring from backup: {backup_dir}")

    # Stop containers if running
    try:
        if Path("docker-compose.yml").exists():
            run_command(["docker", "compose", "down"], check=False)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Restore configuration files
    for config_file in [".env", "docker-compose.yml"]:
        backup_file = backup_dir / config_file
        if backup_file.exists():
            shutil.copy(backup_file, config_file)
            print(f"Restored {config_file}")

    # Restore data directory
    data_backup_dir = backup_dir / "data"
    if data_backup_dir.exists() and data_backup_dir.is_dir():
        data_dir = Path("data")
        if data_dir.exists():
            shutil.rmtree(data_dir)
        shutil.copytree(data_backup_dir, data_dir)
        print(f"Restored data directory from {data_backup_dir}")

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

    print("Backup restoration completed. You can now restart the system.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Deploy Arabic Morphophonology System")
    subparsers = parser.add_subparsers(dest="command", help="Commands", required=True)

    # Docker Compose deployment
    docker_parser = subparsers.add_parser("docker", help="Deploy using Docker Compose")
    docker_parser.add_argument(
        "--build", action="store_true", help="Build containers before starting"
    )
    docker_parser.add_argument(
        "--detach", "-d", action="store_true", help="Run in detached mode"
    )
    docker_parser.set_defaults(func=docker_compose_deploy)

    # Kubernetes deployment
    k8s_parser = subparsers.add_parser("kubernetes", help="Deploy to Kubernetes")
    k8s_parser.add_argument(
        "--namespace", default="default", help="Kubernetes namespace"
    )
    k8s_parser.set_defaults(func=kubernetes_deploy)

    # Backup command
    backup_parser = subparsers.add_parser(
        "backup", help="Create a backup of the system"
    )
    backup_parser.set_defaults(func=create_backup)

    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from a backup")
    restore_parser.add_argument("backup_dir", help="Backup directory to restore from")
    restore_parser.set_defaults(func=restore_backup)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
