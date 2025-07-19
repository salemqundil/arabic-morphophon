"""
Expert deployment script for Arabic Phonological Analysis Engine
Handles multiple server configurations and health monitoring
"""

import subprocess
import sys
import os
import time
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional

class ArabicEngineDeployer:
    """Expert deployment manager for the Arabic phonology engine."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_exe = self.project_root / ".venv" / "Scripts" / "python.exe"
        self.servers = []
        
    def deploy_all_servers(self):
        """Deploy all available server configurations."""
        print("🚀 Arabic Phonological Analysis Engine - Expert Deployment")
        print("=" * 60)
        
        # Available server configurations
        server_configs = [
            {
                "name": "Production Web API",
                "file": "web_api.py",
                "port": 5000,
                "health_endpoint": "/api/health",
                "priority": 1
            },
            {
                "name": "Zero Tolerance Interface", 
                "file": "WEB_TEST.py",
                "port": 5001,
                "health_endpoint": "/health",
                "priority": 2
            },
            {
                "name": "Hafez Experimental Dashboard",
                "file": "hafez_123/app.py", 
                "port": 5002,
                "health_endpoint": "/api/health",
                "priority": 3
            }
        ]
        
        deployed_servers = []
        
        for config in server_configs:
            if self.deploy_server(config):
                deployed_servers.append(config)
                time.sleep(2)  # Allow server to start
        
        if deployed_servers:
            self.run_health_checks(deployed_servers)
            self.show_deployment_summary(deployed_servers)
        else:
            print("❌ No servers could be deployed")
            
    def deploy_server(self, config: Dict) -> bool:
        """Deploy a single server configuration."""
        server_file = self.project_root / config["file"]
        
        if not server_file.exists():
            print(f"❌ {config['name']}: File not found - {server_file}")
            return False
            
        print(f"🚀 Starting {config['name']}...")
        
        try:
            # Start server process
            process = subprocess.Popen(
                [str(self.python_exe), str(server_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.project_root)
            )
            
            config["process"] = process
            config["url"] = f"http://localhost:{config['port']}"
            
            # Wait for server to start
            if self.wait_for_server(config):
                print(f"✅ {config['name']} deployed successfully at {config['url']}")
                return True
            else:
                print(f"❌ {config['name']} failed to start")
                process.terminate()
                return False
                
        except Exception as e:
            print(f"❌ {config['name']} deployment error: {e}")
            return False
    
    def wait_for_server(self, config: Dict, timeout: int = 30) -> bool:
        """Wait for server to become responsive."""
        health_url = f"{config['url']}{config['health_endpoint']}"
        
        for attempt in range(timeout):
            try:
                response = requests.get(health_url, timeout=1)
                if response.status_code == 200:
                    return True
            except:
                pass
            
            time.sleep(1)
            if attempt % 5 == 0:
                print(f"   ⏳ Waiting for {config['name']} to start... ({attempt}s)")
        
        return False
    
    def run_health_checks(self, servers: List[Dict]):
        """Run comprehensive health checks on all servers."""
        print("
🔍 Running Health Checks...")
        print("-" * 40)
        
        for server in servers:
            health_url = f"{server['url']}{server['health_endpoint']}"
            
            try:
                start_time = time.time()
                response = requests.get(health_url, timeout=5)
                response_time = round((time.time() - start_time) * 1000, 2)
                
                if response.status_code == 200:
                    try:
                        health_data = response.json()
                        print(f"✅ {server['name']}: Healthy ({response_time}ms)")
                        if 'version' in health_data:
                            print(f"   📊 Version: {health_data['version']}")
                        if 'uptime' in health_data:
                            print(f"   ⏱️ Uptime: {health_data['uptime']}")
                    except:
                        print(f"✅ {server['name']}: Healthy ({response_time}ms) - Basic response")
                else:
                    print(f"⚠️ {server['name']}: Status {response.status_code}")
                    
            except Exception as e:
                print(f"❌ {server['name']}: Health check failed - {e}")
    
    def show_deployment_summary(self, servers: List[Dict]):
        """Show deployment summary with access information."""
        print("
🎯 Deployment Summary")
        print("=" * 50)
        
        for server in servers:
            print(f"🌐 {server['name']}")
            print(f"   URL: {server['url']}")
            print(f"   Health: {server['url']}{server['health_endpoint']}")
            print(f"   Priority: {server['priority']}")
            print()
        
        print("📋 Quick Access Commands:")
        print("   Primary Interface:", servers[0]['url'] if servers else "None available")
        print("   All endpoints accessible via browser or curl")
        print()
        print("⚡ The system is now running with multiple redundant interfaces!")


if __name__ == "__main__":
    deployer = ArabicEngineDeployer()
    deployer.deploy_all_servers()