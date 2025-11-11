# Multi-stage Dockerfile for Engram cognitive graph database
# Target: <60MB total (6MB base + ~50MB binary)
# Security: Distroless runtime, non-root user, minimal attack surface

# Stage 1: Dependency caching layer
# This stage builds only dependencies to maximize Docker layer cache reuse
FROM rust:1.83-slim AS deps
WORKDIR /build

# Install required build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only manifests first for maximum cache reuse
COPY Cargo.toml Cargo.lock ./
COPY engram-core/Cargo.toml engram-core/
COPY engram-cli/Cargo.toml engram-cli/
COPY engram-proto/Cargo.toml engram-proto/
COPY engram-storage/Cargo.toml engram-storage/
COPY xtask/Cargo.toml xtask/

# Create dummy source files to build dependencies only
RUN mkdir -p engram-core/src engram-cli/src engram-proto/src engram-storage/src xtask/src && \
    echo "fn main() {}" > engram-cli/src/main.rs && \
    touch engram-core/src/lib.rs engram-proto/src/lib.rs engram-storage/src/lib.rs xtask/src/main.rs && \
    cargo build --release -p engram-cli && \
    rm -rf engram-*/src xtask/src

# Stage 2: Build the actual binary
FROM deps AS builder

# Copy actual source code
COPY . .

# Touch main.rs to ensure rebuild with actual source
RUN touch engram-cli/src/main.rs && \
    cargo build --release -p engram-cli && \
    ls -lh target/release/engram

# Stage 3: Minimal runtime with security hardening
FROM gcr.io/distroless/cc-debian12:nonroot

# Copy binary with proper permissions
COPY --from=builder --chown=nonroot:nonroot /build/target/release/engram /usr/local/bin/engram

# Use non-root user from distroless image
USER nonroot:nonroot
WORKDIR /data

# Expose ports (HTTP API and gRPC)
EXPOSE 7432 50051

# Health check using built-in CLI status command
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD ["/usr/local/bin/engram", "status", "--json"] || exit 1

# Signal handling for graceful shutdown
STOPSIGNAL SIGTERM

# Set memory allocator for better container performance
ENV MIMALLOC_SHOW_STATS=0

# Production logging and panic capture
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1

ENTRYPOINT ["/usr/local/bin/engram"]
CMD ["start", "--http-port", "7432", "--grpc-port", "50051"]
