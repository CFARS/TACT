# Contributing to TACT

We welcome contributions to TACT! This guide will help you get started with the development process.

## Setting Up Development Environment

1. Fork the TACT repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/YOUR_USERNAME/TACT.git
cd TACT
```

3. Install development dependencies:
```bash
pip install -e .
pip install -r docs/requirements.txt
```

## Documentation Development

### Serving Documentation Locally

To preview documentation changes locally:

```bash
mkdocs serve
```

This will start a local server at `http://127.0.0.1:8000`. The documentation will automatically reload when you make changes to the source files.

### Documentation Structure

- `docs/`: Main documentation directory
  - `index.md`: Home page
  - `quickstart.md`: Getting started guide
  - `contributing.md`: This contribution guide
  - `api/`: API reference documentation

### Building Documentation

To build the documentation:

```bash
mkdocs build
```

This creates a `site` directory with the built HTML documentation.

## Making Contributions

1. Create a new branch for your feature:
```bash
git checkout -b feature-name
```

2. Make your changes and commit them:
```bash
git add .
git commit -m "Description of changes"
```

3. Push to your fork:
```bash
git push origin feature-name
```

4. Open a Pull Request on GitHub

### Code Style Guidelines

- Use type hints for all function parameters and return values
- Include docstrings for all functions and classes (NumPy style)
- Follow PEP 8 guidelines
- Keep functions focused and single-purpose
- Write clear commit messages

### Testing

Before submitting a PR:
1. Ensure all existing tests pass
2. Add tests for new functionality
3. Update documentation for any changes
4. Test documentation builds locally

## Getting Help

If you need help or have questions:
1. Open an issue on GitHub
2. Check existing documentation
3. Contact the maintainers

Thank you for contributing to TACT! 