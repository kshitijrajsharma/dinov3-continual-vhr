set shell := ["bash", "-cu"]

# install training dependencies and the pre-commit hooks
setup:
    uv sync
    uv run pre-commit install --install-hooks --hook-type pre-commit --hook-type commit-msg

# add the optional evaluation dependencies (pulls the pinned pangaea benchmark)
setup-eval:
    uv sync --group eval

# run all pre-commit hooks (ruff fix, ruff format, ty, file hygiene)
lint:
    uv run pre-commit run --all-files

# run the test suite
test:
    uv run pytest

# continued pretraining entrypoint (pass extra flags after --)
run *args:
    uv run python -m dinov3_vhr.ssl.train {{args}}
