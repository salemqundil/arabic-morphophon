# CI/CD Pipeline Setup Guide

## 📋 Overview

This Arabic Phonology Engine includes a comprehensive CI/CD pipeline with four specialized workflows:

- **CI Pipeline** (`ci.yml`) - Testing, quality checks, and deployment
- **Security Scanning** (`security.yml`) - Vulnerability detection and compliance
- **Performance Benchmarks** (`benchmark.yml`) - Performance monitoring and regression detection
- **Dependency Management** (`dependencies.yml`) - Automated dependency updates

## 🚀 Quick Start

### 1. Repository Setup

Ensure your repository has the following structure:

```text
.github/
├── workflows/
│   ├── ci.yml                 # Main CI/CD pipeline
│   ├── security.yml           # Security scanning
│   ├── benchmark.yml          # Performance tests
│   └── dependencies.yml       # Dependency updates
├── ISSUE_TEMPLATE/
│   ├── bug_report.yml
│   └── feature_request.yml
└── PULL_REQUEST_TEMPLATE.md
```

### 2. Required Secrets

Configure these secrets in GitHub Settings → Secrets and variables → Actions:

| Secret Name | Purpose | Required |
|-------------|---------|----------|
| `PYPI_API_TOKEN` | PyPI package publishing | ✅ |
| `DOCKERHUB_USERNAME` | Docker Hub authentication | ✅ |
| `DOCKERHUB_TOKEN` | Docker Hub token | ✅ |
| `CODECOV_TOKEN` | Code coverage reporting | ⚠️ Optional |

#### How to Configure Secrets

**PyPI API Token:**

1. Go to [PyPI Account Settings](https://pypi.org/manage/account/)
2. Create new API token with scope "Entire account"
3. Copy the token (starts with `pypi-`)
4. Add as `PYPI_API_TOKEN` secret in GitHub

**Docker Hub Token:**

1. Go to [Docker Hub Account Settings](https://hub.docker.com/settings/security)
2. Create new access token with "Read, Write, Delete" permissions
3. Copy the token (starts with `dckr_pat_`)
4. Add `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets

### 3. Branch Protection Rules

Set up branch protection for `main`:

1. Go to Settings → Branches
2. Add rule for `main` branch
3. Enable:
   - Require pull request reviews
   - Require status checks to pass
   - Require branches to be up to date
   - Include administrators

## 📋 Workflow Details

### CI Pipeline (`ci.yml`)

**Triggers:**

- Push to `main` branch
- Pull requests to `main`
- Manual dispatch

**Features:**

- Multi-platform testing (Ubuntu, Windows, macOS)
- Python version matrix (3.9, 3.10, 3.11, 3.12)
- Code quality checks (Black, Flake8, mypy)
- Security scanning (Bandit)
- Test coverage reporting
- Automated PyPI publishing (on tags)
- Docker image building and publishing

**Quality Gates:**

- All tests must pass
- Code coverage > 80%
- No security vulnerabilities
- Code style compliance

### Security Pipeline (`security.yml`)

**Triggers:**

- Weekly schedule
- Manual dispatch
- Security-related file changes

**Scans:**

- CodeQL analysis
- Dependency vulnerability scanning
- Docker image security scanning
- SAST (Static Application Security Testing)

### Benchmark Pipeline (`benchmark.yml`)

**Triggers:**

- Push to `main`
- Weekly schedule

**Metrics:**

- Performance regression detection
- Memory usage monitoring
- Execution time tracking
- Comparison with baseline

### Dependency Updates (`dependencies.yml`)

**Triggers:**

- Weekly schedule
- Manual dispatch

**Actions:**

- Automated dependency updates
- Security patch detection
- Compatibility testing
- Automated PR creation

## 🛠️ Configuration Files

### pyproject.toml

Essential configuration for modern Python packaging:

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "arabic-phonology-engine"
dynamic = ["version"]
description = "Professional Arabic phonological analysis system"
authors = [{name = "Your Name", email = "your.email@example.com"}]
license = {text = "MIT"}
requires-python = ">=3.9"
```

### requirements.txt

Production dependencies only:

```text
# Keep this minimal - core dependencies only
# Development dependencies go in requirements-dev.txt

# Production dependencies for Arabic Phonology Engine
flask>=2.3.0,<3.0.0
gunicorn>=21.0.0,<22.0.0
requests>=2.31.0,<3.0.0

# Optional: Pin specific versions for production stability
# flask==2.3.3
# gunicorn==21.2.0
```

### Docker Configuration

- `Dockerfile` - Production container
- `docker-compose.yml` - Development environment
- `.dockerignore` - Exclude unnecessary files

## 🔍 Monitoring and Alerts

### GitHub Actions Status

Monitor workflow status at: `https://github.com/YOUR_USERNAME/arabic-phonology-engine/actions`

### Key Metrics

- **Test Coverage**: Minimum 80% required
- **Build Time**: Target < 10 minutes
- **Security Score**: Zero high-severity vulnerabilities
- **Performance**: No regressions > 10%

### Troubleshooting

#### Common Issues

1. **Tests Failing**

   ```bash
   # Run tests locally
   python -m pytest -v
   ```

2. **Docker Build Failing**

   ```bash
   # Test Docker build locally
   docker build -t arabic-phonology-engine .
   ```

3. **Security Scan Issues**

   ```bash
   # Run security checks locally
   python -m bandit -r src/
   ```

4. **Secret not found errors**
   - Verify secrets are correctly named and set
   - Check environment protection rules

5. **PyPI publishing fails**
   - Ensure version number is incremented
   - Check package configuration in `pyproject.toml`

#### Getting Help

- Check workflow logs in GitHub Actions
- Review error messages in failed builds
- Consult documentation in `/docs` folder
- Create issue using provided templates

## 🔧 Local Testing

Test your pipeline locally before pushing:

```bash
# Quality checks
black --check src/ tests/
isort --check-only src/ tests/
flake8 src/ tests/
mypy src/
bandit -r src/

# Run tests
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/performance/ --benchmark-only

# Docker build
docker build -t arabic-phonology-engine:test .
docker run --rm -p 5000:5000 arabic-phonology-engine:test
```

## 💡 Best Practices

### Code Quality

- ✅ Use type hints
- ✅ Write comprehensive tests
- ✅ Follow PEP 8 style guide
- ✅ Document all public APIs
- ✅ Keep functions small and focused

### Security

- ✅ Regular dependency updates
- ✅ No secrets in code
- ✅ Input validation
- ✅ Error handling
- ✅ Security scanning

### Performance

- ✅ Profile critical paths
- ✅ Monitor memory usage
- ✅ Optimize hot spots
- ✅ Cache expensive operations
- ✅ Use efficient algorithms

### Deployment

- ✅ Automated testing
- ✅ Gradual rollouts
- ✅ Rollback capability
- ✅ Health checks
- ✅ Monitoring and alerting

### Development Workflow

1. **Branch Protection**: Require PR reviews and status checks
2. **Semantic Versioning**: Use conventional commits for releases
3. **Documentation**: Keep README and docs updated
4. **Monitoring**: Set up notification preferences
5. **Security**: Regularly review and rotate tokens

## 🚨 Emergency Procedures

### Hot Fix Deployment

For critical issues, use manual workflow dispatch:

1. Go to Actions → CI/CD Pipeline → Run workflow
2. Select branch with fix
3. Enable `skip_tests: true` if needed
4. Monitor deployment status

### Rollback Process

1. Identify last known good commit
2. Create rollback branch
3. Deploy through normal CI/CD process
4. Update monitoring and alerts

## 📊 Dashboard Links

After setup, monitor these dashboards:

- **GitHub Actions**: `https://github.com/YOUR_USERNAME/arabic-phonology-engine/actions`
- **Security Tab**: `https://github.com/YOUR_USERNAME/arabic-phonology-engine/security`
- **Codecov**: `https://codecov.io/gh/YOUR_USERNAME/arabic-phonology-engine`
- **Docker Hub**: `https://hub.docker.com/r/YOUR_USERNAME/arabic-phonology-engine`

## 🎯 Next Steps

1. **Push to GitHub** - Workflows will run automatically
2. **Configure secrets** - Enable publishing features
3. **Create first release** - Test deployment pipeline
4. **Monitor dashboards** - Track metrics and performance
5. **Iterate and improve** - Continuous enhancement

Your CI/CD pipeline is now enterprise-ready! 🚀
