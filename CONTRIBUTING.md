# Contributing to Arabic Phonology Engine

This document outlines the guidelines and steps for contributing to the Arabic Phonology Engine project.
## Welcome

Thank you for your interest in contributing to the Arabic Phonology Engine project! We value your contributions and look forward to collaborating with you to improve the project.

## Welcome

Thank you for your interest in contributing to the Arabic Phonology Engine project!

## Development Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/arabic-phonology-engine.git
   cd arabic-phonology-engine
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # "venv" stands for "virtual environment" folder. On Windows: venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards (Black, isort, flake8)
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests**
   ```bash
   python test_orchestrator.py --category all
   ```

4. **Run quality checks**
   ```bash
   python -m black .
   python -m isort .
   python -m flake8 .
   python -m mypy . --ignore-missing-imports
   ```

5. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: your descriptive commit message"
   git push origin feature/your-feature-name
   ```

6. **Create a pull request**

## Code Standards

### Python Style
- Use Black for code formatting
- Use isort for import sorting
- Follow PEP 8 guidelines
- Use type hints where appropriate

### Testing
- Write unit tests for all new functions
- Maintain test coverage above 85%
- Use descriptive test names
- Test edge cases and error conditions

### Documentation
- Use docstrings for all functions and classes
- Update README.md for significant changes
- Include examples in documentation

## Project Structure

```
arabic-phonology-engine/
├── new_engine/           # Core neural engine
├── phonology/           # Text analysis layer
├── unified_analyzer.py  # Unified analysis interface
├── web_*.py            # Web interfaces
├── test_*.py           # Test files
├── docs/               # Documentation
└── .github/            # GitHub workflows
```

## Commit Convention

Use conventional commits:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test changes
- `refactor:` for code refactoring
- `style:` for formatting changes

## Questions?

Feel free to open an issue for questions or suggestions!
