# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-07-05

### Added
- Introduced modular codebase layout under `src/` (`config`, `model`, `dataset`, `inference`, `visualization`).
- Implemented static dataset stats normalization variables in config to avoid re-parsing `strokes.npy` (46MB) at runtime.
- Added a `bias` slider control to the React interface to adjust handwriting neatness dynamically.
- Built-in front-end character validation with user-facing warnings for unsupported characters.
- Prepopulated click-to-fill preset example phrases.
- Added a direct "Download File" link for generated PNG/GIF assets.
- Implemented custom `.gitignore` and `.dockerignore` to streamline development and builds.

### Fixed
- Resolved a critical thread-safety concurrency bug in `HandWritingSynthesisNet` where shared `self.EOS` and `self._phi` state variables on the model instance caused generation truncation when multiple requests arrived simultaneously (resolved using `threading.local()`).
- Replaced absolute user machine directory paths with clean local paths inside `main.py`.
- Retrofitted `main.py` to serve as a backward-compatible CLI utility wrapper.
- Integrated Python's standard `logging` library instead of console print calls in `app.py`.
- Restricted Matplotlib to the non-interactive `Agg` backend to avoid threading GUI lock crashes on headless servers.

## [1.0.0] - 2025-05-12

### Added
- Initial deployment of handwriting synthesis neural network (trained model and dataset strokes).
- Flask app backend server and a simple client UI using React CDN.
