# Fazenda Workflow & Activators

## Overview
This document outlines the cooperation workflows and automated triggers for the **Fazenda** (Asset Generator Farm) project.

## Branch Strategy
- **`main`**: Production-ready code. Protected branch.
- **`pairduo`**: Development branch for Market Analysis tools. Synchronized with the external `Threndia` repository.

## GitHub Actions Triggers
The project uses `.github/workflows/main.yml` to automate quality checks and notifications.

### 1. Quality Check
**Trigger**: Push to `main` or `pairduo`, or Pull Request to `main`.
- **Linting**: Runs `ruff`.
- **Testing**: Runs `pytest`.

### 2. Sync Cooperation
**Trigger**: Push to `pairduo`.
- **Action**: Notifies that changes are ready for synchronization with `Threndia`.
- **Manual Step**: Run `scripts/sync_threndia.py` to push changes to the remote partner repo.

## Manual Activators
Scripts available in `scripts/`:
- `sync_dual_duo.py`: Bidirectional sync with `Threndia`. Use `--push` or `--pull`.
- `prepare_dummy_dataset.py`: Activates data prep testing.

## How to Contribute
1. Create a feature branch from `main` (or `pairduo` for market tools).
2. Open a Pull Request using the provided template.
3. Ensure CI checks pass.
