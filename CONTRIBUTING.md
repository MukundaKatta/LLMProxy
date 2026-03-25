# Contributing to LLMProxy

Thank you for your interest in contributing to LLMProxy! This guide will help you get started.

## Development Setup

1. **Fork and clone** the repository:

   ```bash
   git clone https://github.com/your-username/LLMProxy.git
   cd LLMProxy
   ```

2. **Create a virtual environment** and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   make install
   ```

3. **Copy the environment file** and add your API keys (optional, needed for integration tests):

   ```bash
   cp .env.example .env
   ```

## Making Changes

1. Create a feature branch from `main`:

   ```bash
   git checkout -b feat/your-feature-name
   ```

2. Write your code with tests. All new features should include unit tests.

3. Run the quality checks:

   ```bash
   make lint
   make test
   ```

4. Commit with a clear message following [Conventional Commits](https://www.conventionalcommits.org/):

   ```
   feat: add Google Gemini provider
   fix: correct token counting for Anthropic models
   docs: update quickstart example
   ```

## Pull Request Guidelines

- Keep PRs focused on a single change.
- Include tests for new functionality.
- Update documentation if the public API changes.
- Ensure CI passes before requesting review.

## Adding a New Provider

1. Create a new class that inherits from `BaseProvider` in `src/llmproxy/core.py`.
2. Implement the `complete()`, `health_check()`, and `list_models()` methods.
3. Register the provider type in `PROVIDER_REGISTRY` inside `core.py`.
4. Add tests in `tests/test_core.py`.
5. Update the README with usage examples.

## Code Style

- We use **ruff** for linting and formatting.
- Type annotations are required for all public functions.
- Run `make format` before committing.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
