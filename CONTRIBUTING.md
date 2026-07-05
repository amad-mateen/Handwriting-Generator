# Contributing to Handwriting Synthesis

Thank you for your interest in improving this project! To maintain a clean and professional codebase, please follow these guidelines.

## Development Workflow

1. **Fork and Clone**: Fork the repository and clone it to your local environment.
2. **Setup Environment**:
   We recommend using a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Local Testing**:
   Before submitting changes, test both the command-line utility and the web server:
   ```bash
   # Test CLI
   python main.py
   
   # Test Flask Server
   python app.py
   ```

## Code Quality Standards

* **Modularity**: Keep core operations separated under the `src/` directory:
  - `src/config.py` - Global constants and parameters.
  - `src/model.py` - Core network modules and PyTorch code.
  - `src/dataset.py` - Training data loaders and preprocessors.
  - `src/inference.py` - Singletons, caching, and evaluation logic.
  - `src/visualization.py` - Drawing plots and compiler animations.
* **Thread Safety**: Never write request-specific state variables to PyTorch model attributes (`self.xyz = value`). Instead, leverage python's `threading.local` container context.
* **Type Hints**: Annotate public APIs and methods with proper type signatures.
* **Logging**: Use Python's built-in `logging` module (`logger.info`, `logger.error`) in servers rather than standard `print()` statements.

## Deploying to Hugging Face Spaces

This application deploys automatically on Hugging Face Spaces using the Docker SDK.
* If you modify dependencies, update `requirements.txt`.
* Ensure that any file creations (like outputs) write to folders initialized with loose access permissions (`chmod 777`) because Hugging Face executes the container runtime under user UID 1000.
* Do not change the exposed container port from `7860`.
