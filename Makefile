.PHONY: install dev test test-unit test-integration test-evals lint proto clean docker-build docker-up docker-down help

PYTHON := python3
PIP := pip3

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	$(PIP) install -e .

dev: ## Install with dev dependencies and start services
	$(PIP) install -e ".[dev]"
	@echo "Starting ChainMind in development mode..."
	$(PYTHON) -m chainmind.api.app

dev-grpc: ## Start gRPC server
	$(PYTHON) -m chainmind.grpc_service.server

proto: ## Compile protobuf definitions
	$(PYTHON) -m grpc_tools.protoc \
		-I./chainmind/grpc_service/protos \
		--python_out=./chainmind/grpc_service \
		--grpc_python_out=./chainmind/grpc_service \
		--pyi_out=./chainmind/grpc_service \
		./chainmind/grpc_service/protos/*.proto

test: test-unit test-integration ## Run all tests

test-unit: ## Run unit tests
	$(PYTHON) -m pytest tests/unit -v --tb=short -m unit

test-integration: ## Run integration tests
	$(PYTHON) -m pytest tests/integration -v --tb=short -m integration

test-evals: ## Run evaluation pipeline
	$(PYTHON) -m pytest tests/evals -v --tb=short -m eval

lint: ## Run linter
	$(PYTHON) -m ruff check chainmind/ tests/
	$(PYTHON) -m mypy chainmind/

format: ## Format code
	$(PYTHON) -m ruff format chainmind/ tests/

seed: ## Seed knowledge base with sample data
	$(PYTHON) scripts/seed_knowledge_base.py

docker-build: ## Build Docker image
	docker build -f deploy/docker/Dockerfile -t chainmind:latest .

docker-up: ## Start all services with Docker Compose
	docker compose -f deploy/docker/docker-compose.yml up -d

docker-down: ## Stop Docker Compose services
	docker compose -f deploy/docker/docker-compose.yml down

clean: ## Clean build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ .pytest_cache/ .mypy_cache/ .ruff_cache/
