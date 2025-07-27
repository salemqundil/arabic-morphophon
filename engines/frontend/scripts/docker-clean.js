/**
 * Docker Clean Utility
 *
 * This script provides utilities to clean up Docker resources:
 * - Remove stopped containers
 * - Remove unused images
 * - Remove unused volumes
 * - Remove unused networks
 */

const { exec } = require('child_process');

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
};

// Execute command and return promise
function execCommand(command) {
  return new Promise((resolve, reject) => {
    console.log(`${colors.blue}Executing: ${colors.cyan}${command}${colors.reset}`);

    exec(command, (error, stdout, stderr) => {
      if (error) {
        console.error(`${colors.red}Error: ${error.message}${colors.reset}`);
        reject(error);
        return;
      }

      if (stderr) {
        console.error(`${colors.yellow}Warning: ${stderr}${colors.reset}`);
      }

      console.log(`${colors.green}Output: ${stdout || 'No output'}${colors.reset}`);
      resolve(stdout);
    });
  });
}

// Clean Docker resources
async function cleanDocker() {
  try {
    console.log(`${colors.magenta}=======================================`);
    console.log(`       Docker Cleanup Utility`);
    console.log(`=======================================\n${colors.reset}`);

    // Step 1: Remove stopped containers
    console.log(`${colors.cyan}Step 1: Removing stopped containers...${colors.reset}`);
    await execCommand('docker container prune -f');

    // Step 2: Remove unused images
    console.log(`\n${colors.cyan}Step 2: Removing unused images...${colors.reset}`);
    await execCommand('docker image prune -f');

    // Step 3: Remove unused volumes
    console.log(`\n${colors.cyan}Step 3: Removing unused volumes...${colors.reset}`);
    await execCommand('docker volume prune -f');

    // Step 4: Remove unused networks
    console.log(`\n${colors.cyan}Step 4: Removing unused networks...${colors.reset}`);
    await execCommand('docker network prune -f');

    console.log(`\n${colors.green}=======================================`);
    console.log(`Docker cleanup completed successfully!`);
    console.log(`=======================================${colors.reset}`);

  } catch (error) {
    console.error(`${colors.red}=======================================`);
    console.error(`Docker cleanup failed!`);
    console.error(`=======================================${colors.reset}`);
    process.exit(1);
  }
}

// Run the cleanup
cleanDocker();
