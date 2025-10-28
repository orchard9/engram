#!/bin/bash
# Certificate generation script for Engram TLS/mTLS configuration
#
# This script generates self-signed certificates for development and testing.
# For production, use certificates from a trusted CA (Let's Encrypt, DigiCert, etc.)

set -euo pipefail

# Configuration
CERTS_DIR="${CERTS_DIR:-./certs}"
CA_DAYS="${CA_DAYS:-3650}"
CERT_DAYS="${CERT_DAYS:-365}"
SERVER_CN="${SERVER_CN:-localhost}"
SERVER_IP="${SERVER_IP:-127.0.0.1}"
CLIENT_CN="${CLIENT_CN:-engram-client}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if openssl is available
if ! command -v openssl &> /dev/null; then
    log_error "openssl is not installed. Please install it first."
    exit 1
fi

# Create certificates directory
mkdir -p "$CERTS_DIR"
cd "$CERTS_DIR"

log_info "Generating certificates in $CERTS_DIR"

# Generate CA for development/testing
generate_ca() {
    log_info "Generating Certificate Authority (CA)..."

    openssl genrsa -out ca.key 4096
    openssl req -new -x509 -days "$CA_DAYS" -key ca.key \
        -out ca.crt \
        -subj "/C=US/ST=California/L=San Francisco/O=Engram/OU=Development/CN=Engram CA"

    log_info "CA certificate generated: ca.crt"
}

# Generate server certificate
generate_server_cert() {
    log_info "Generating server certificate for $SERVER_CN..."

    # Create server private key
    openssl genrsa -out server.key 2048

    # Create certificate signing request
    openssl req -new -key server.key \
        -out server.csr \
        -subj "/C=US/ST=California/L=San Francisco/O=Engram/OU=Server/CN=$SERVER_CN"

    # Create extension file for SAN
    cat > server_ext.cnf <<EOF
subjectAltName = DNS:$SERVER_CN,DNS:localhost,IP:$SERVER_IP,IP:127.0.0.1
extendedKeyUsage = serverAuth
EOF

    # Sign with CA
    openssl x509 -req -days "$CERT_DAYS" -in server.csr \
        -CA ca.crt -CAkey ca.key -CAcreateserial \
        -out server.crt \
        -extfile server_ext.cnf

    # Clean up
    rm server.csr server_ext.cnf

    log_info "Server certificate generated: server.crt"
    log_info "Server private key: server.key"
}

# Generate client certificate for mTLS
generate_client_cert() {
    log_info "Generating client certificate for $CLIENT_CN..."

    # Create client private key
    openssl genrsa -out client.key 2048

    # Create certificate signing request
    openssl req -new -key client.key \
        -out client.csr \
        -subj "/C=US/ST=California/L=San Francisco/O=Engram/OU=Client/CN=$CLIENT_CN"

    # Create extension file
    cat > client_ext.cnf <<EOF
extendedKeyUsage = clientAuth
EOF

    # Sign with CA
    openssl x509 -req -days "$CERT_DAYS" -in client.csr \
        -CA ca.crt -CAkey ca.key -CAcreateserial \
        -out client.crt \
        -extfile client_ext.cnf

    # Clean up
    rm client.csr client_ext.cnf

    log_info "Client certificate generated: client.crt"
    log_info "Client private key: client.key"
}

# Verify certificates
verify_certs() {
    log_info "Verifying certificates..."

    # Verify server certificate
    if openssl verify -CAfile ca.crt server.crt > /dev/null 2>&1; then
        log_info "Server certificate verification: OK"
    else
        log_error "Server certificate verification: FAILED"
        exit 1
    fi

    # Verify client certificate
    if openssl verify -CAfile ca.crt client.crt > /dev/null 2>&1; then
        log_info "Client certificate verification: OK"
    else
        log_error "Client certificate verification: FAILED"
        exit 1
    fi
}

# Display certificate information
display_cert_info() {
    log_info "Certificate Information:"
    echo
    echo "CA Certificate:"
    openssl x509 -in ca.crt -noout -subject -dates
    echo
    echo "Server Certificate:"
    openssl x509 -in server.crt -noout -subject -dates -ext subjectAltName
    echo
    echo "Client Certificate:"
    openssl x509 -in client.crt -noout -subject -dates
}

# Set proper permissions
set_permissions() {
    log_info "Setting secure file permissions..."

    chmod 600 *.key
    chmod 644 *.crt

    log_info "Permissions set: keys (600), certificates (644)"
}

# Main execution
main() {
    log_info "Starting certificate generation..."
    log_info "Configuration:"
    log_info "  Certificates directory: $CERTS_DIR"
    log_info "  Server CN: $SERVER_CN"
    log_info "  Server IP: $SERVER_IP"
    log_info "  Client CN: $CLIENT_CN"
    log_info "  CA validity: $CA_DAYS days"
    log_info "  Certificate validity: $CERT_DAYS days"
    echo

    # Check if certificates already exist
    if [ -f "ca.crt" ] || [ -f "server.crt" ] || [ -f "client.crt" ]; then
        log_warn "Certificates already exist in $CERTS_DIR"
        read -p "Do you want to regenerate them? This will overwrite existing certificates. (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Aborting certificate generation."
            exit 0
        fi
        log_warn "Regenerating certificates..."
    fi

    # Generate certificates
    generate_ca
    generate_server_cert
    generate_client_cert
    verify_certs
    set_permissions

    echo
    display_cert_info
    echo
    log_info "Certificate generation complete!"
    log_info "Files created:"
    log_info "  CA: ca.crt, ca.key"
    log_info "  Server: server.crt, server.key"
    log_info "  Client: client.crt, client.key"
    echo
    log_warn "These are self-signed certificates for development/testing only."
    log_warn "For production, use certificates from a trusted CA."
}

# Run main function
main "$@"
