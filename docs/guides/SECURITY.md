# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | ✅ Yes             |
| < 1.0   | ❌ No              |

## Reporting a Vulnerability

Please report security vulnerabilities via GitHub Security Advisories or by emailing security@your-org.com.

**Do not report security vulnerabilities in public issues.**

### What to include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fixes (if any)

### Response timeline:
- **Initial response**: Within 24 hours
- **Assessment**: Within 72 hours
- **Fix timeline**: Varies by severity

## Security Best Practices

When contributing to this project:
- Never commit secrets or credentials
- Use secure coding practices
- Validate all inputs
- Process errors gracefully
- Follow the principle of least privilege

## Dependencies

We regularly scan dependencies for security vulnerabilities using:
- `bandit` for Python security issues
- `safety` for dependency vulnerabilities
- `pip-audit` for package auditing
- GitHub Dependabot alerts

## Contact

For security-related questions: security@your-org.com
