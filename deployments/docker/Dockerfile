# Multi-stage Dockerfile for Engram cognitive graph database
# Target: <60MB total (6MB base + ~50MB binary)
# Security: Distroless runtime, non-root user, minimal attack surface

# Stage 1: Build the binary
FROM rust:1.91-slim AS builder
WORKDIR /build

# Install required build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libhwloc-dev \
    libudev-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Copy full workspace
COPY . .

RUN cargo build --release -p engram-cli && \
    ls -lh target/release/engram

# Stage 2: Minimal runtime with necessary shared libraries
FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libhwloc15 \
    libudev1 \
    libnuma1 \
    libpciaccess0 \
    libxml2 \
    libx11-6 \
    libxext6 \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 10001 engram

# Copy binary with proper permissions
COPY --from=builder --chown=engram:engram /build/target/release/engram /usr/local/bin/engram

USER engram:engram
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
