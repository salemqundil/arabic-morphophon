# GitHub Actions Setup Guide

## 🔐 Required Secrets Configuration

To fully activate your CI/CD pipeline, configure these secrets in GitHub:

**Settings → Secrets and variables → Actions → New repository secret**

### 1. PyPI Publishing
```
Secret Name: PYPI_API_TOKEN
Value: pypi-AgE... (your PyPI API token)
```

**How to get PyPI token:**
1. Go to [PyPI Account Settings](https://pypi.org/manage/account/)
2. Create new API token with scope "Entire account"
3. Copy the token (starts with `pypi-`)

### 2. Docker Hub Publishing
```
Secret Name: DOCKERHUB_USERNAME
Value: your-dockerhub-username

Secret Name: DOCKERHUB_TOKEN  
Value: dckr_pat_... (Docker Hub access token)
```

**How to get Docker Hub token:**
1. Go to [Docker Hub Account Settings](https://hub.docker.com/settings/security)
2. Create new access token with "Read, Write, Delete" permissions
3. Copy the token (starts with `dckr_pat_`)

## 🚀 Environment Setup (Optional)

For additional protection, create a "production" environment:

**Settings → Environments → New environment**
- Name: `production`
- Protection rules:
  - ✅ Required reviewers (1-2 people)
  - ✅ Wait timer (5 minutes)
  - ✅ Restrict to protected branches

## 📋 Workflow Features

### ✅ **Triggers**
- **Push**: `main` and `develop` branches
- **Pull Request**: Against `main` and `develop`
- **Release**: Automatic deployment when published
- **Manual**: `workflow_dispatch` with skip options

### ✅ **Quality Checks**
- **Black**: Code formatting validation
- **isort**: Import sorting validation  
- **Flake8**: Linting with GitHub annotations
- **mypy**: Type checking with strict mode
- **Bandit**: Security vulnerability scanning

### ✅ **Testing Matrix**
- **Platforms**: Ubuntu, Windows, macOS
- **Python**: 3.9, 3.10, 3.11, 3.12
- **Coverage**: Codecov integration
- **Performance**: Benchmark tracking

### ✅ **Security**
- **Container Scanning**: Trivy security analysis
- **Dependency Check**: Automated vulnerability detection
- **SARIF Reports**: GitHub Security tab integration

### ✅ **Deployment**
- **PyPI**: Automatic package publishing on release
- **Docker Hub**: Multi-platform container publishing
- **GitHub Pages**: Documentation deployment

## 🔧 Manual Testing

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

## 📊 Monitoring

### **GitHub Actions Tab**
- View workflow runs and logs
- Download artifacts (test reports, security scans)
- Monitor performance trends

### **Security Tab**  
- Review security advisories
- Check dependency vulnerabilities
- Monitor code scanning alerts

### **Codecov Dashboard**
- Track coverage trends
- Review coverage reports
- Set coverage goals

## 🚨 Troubleshooting

### **Common Issues**

1. **Secret not found errors**
   - Verify secrets are correctly named and set
   - Check environment protection rules

2. **Test failures on specific platforms**
   - Review matrix configuration
   - Check platform-specific dependencies

3. **Docker build failures**
   - Verify Dockerfile syntax
   - Check multi-stage build compatibility

4. **PyPI publishing fails**
   - Ensure version number is incremented
   - Check package configuration in `pyproject.toml`

### **Emergency Deploy**
Use manual workflow dispatch with `skip_tests: true` for critical hotfixes.

## 📈 Best Practices

1. **Branch Protection**: Require PR reviews and status checks
2. **Semantic Versioning**: Use conventional commits for releases
3. **Documentation**: Keep README and docs updated
4. **Monitoring**: Set up notification preferences
5. **Security**: Regularly review and rotate tokens

Your CI/CD pipeline is now **production-ready** with enterprise-grade quality gates! 🎉
