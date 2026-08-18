.PHONY: all build install clean test lint setup

# Go CLI
GO_MODULE = github.com/agent-fabric/dispatch
GO_BINARY = fabric
GO_CMD = cmd/fabric

# Python
PYTHON = python3
PIP = pip3

all: build

## Build the fabric CLI binary
build:
	go build -o dist/$(GO_BINARY) ./$(GO_CMD)

## Install fabric CLI to $GOPATH/bin
install: build
	go install ./$(GO_CMD)

## Set up Python dependencies
setup:
	$(PIP) install -e ".[dev]"

## Run Go + Python tests
test: test-go test-python

test-go:
	go test ./...

test-python:
	$(PYTHON) -m pytest tests/ -v

## Lint all code
lint: lint-go lint-python

lint-go:
	go vet ./...

lint-python:
	$(PYTHON) -m ruff check .

## Clean build artifacts
clean:
	rm -rf dist/ build/ *.egg-info/
	go clean

## Full development setup
dev-setup: setup
	go mod download
	@echo ""
	@echo "Development environment ready."
	@echo "  Build CLI:    make build"
	@echo "  Run tests:    make test"
	@echo "  Start worker: fabric worker up --local"

## Show available commands
help:
	@echo "Available targets:"
	@echo "  build       Build the fabric CLI binary"
	@echo "  install     Install fabric CLI to GOPATH/bin"
	@echo "  setup       Install Python dependencies"
	@echo "  test        Run all tests (Go + Python)"
	@echo "  lint        Lint all code"
	@echo "  clean       Clean build artifacts"
	@echo "  dev-setup   Full development environment setup"
