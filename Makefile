.DEFAULT_GOAL: help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
    match = re.match(r'^([^\s:]+):.*?## (.*)$$', line)
    if match:
        target, help = match.groups()
        print("%-20s: %s" % (target, help))
endef

export PRINT_HELP_PYSCRIPT

.PHONY: help
help: ## show help message
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: bootstrap
bootstrap: ## bootstrap this project (create venv, installall deps, ...)
	python3 -m venv venv
	. venv/bin/activate && make dep-dev && make dep
	@echo ""
	@echo "To activate created virtualenv, run the following command."
	@echo "    . venv/bin/activate"
	@[ -f .env ] || cp .env.sample .env && echo "(Edit your .env if you need.)"

.PHONY: test
test: ## run tests in tests/ directory
	python -m unittest discover

.PHONY: lint
lint: ## run linters
	black --diff --check --exclude venv .
	mypy ocr --ignore-missing-imports

.PHONY: format
format: ## format python scripts with black
	black --exclude venv .

.PHONY: dep
dep: requirements.txt  ## install dependencies
	pip install -r requirements.txt

.PHONY: dep-dev
dep-dev: requirements-dev.txt ## install development dependencies
	pip install -r requirements-dev.txt

.PHONY: dep-internal
dep-internal: requirements-internal.txt ## install internal dependencies
	pip install -r requirements-internal.txt

.PHONY: train
train: ## run training script and sync the models and data to s3
	python -m ocr.models.train
