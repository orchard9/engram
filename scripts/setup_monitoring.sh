#!/bin/bash
# Setup script for Engram monitoring stack
# Deploys Prometheus, Grafana, and Loki to Kubernetes or Docker Compose

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOYMENTS_DIR="${PROJECT_ROOT}/deployments"

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

check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi

    if ! command -v docker &> /dev/null; then
        log_warn "docker not found. Docker deployment will not be available."
    fi

    log_info "Prerequisites check passed"
}

deploy_kubernetes() {
    log_info "Deploying monitoring stack to Kubernetes..."

    # Apply monitoring stack manifest
    kubectl apply -f "${DEPLOYMENTS_DIR}/kubernetes/monitoring-stack.yaml"

    log_info "Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod -l app=prometheus -n monitoring --timeout=120s
    kubectl wait --for=condition=ready pod -l app=grafana -n monitoring --timeout=120s
    kubectl wait --for=condition=ready pod -l app=loki -n monitoring --timeout=120s

    log_info "Monitoring stack deployed successfully"

    # Get Grafana URL
    GRAFANA_IP=$(kubectl get svc grafana -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    if [ "$GRAFANA_IP" != "pending" ]; then
        log_info "Grafana available at: http://${GRAFANA_IP}:3000"
        log_info "Default credentials: admin/admin"
    else
        log_info "Grafana LoadBalancer IP pending. Check with: kubectl get svc grafana -n monitoring"
    fi

    # Get Prometheus URL
    log_info "Prometheus available at: kubectl port-forward -n monitoring svc/prometheus 9090:9090"

    # Get Loki URL
    log_info "Loki available at: kubectl port-forward -n monitoring svc/loki 3100:3100"
}

deploy_docker_compose() {
    log_info "Deploying monitoring stack with Docker Compose..."

    # Create docker-compose.yml
    cat > "${DEPLOYMENTS_DIR}/docker-compose.yml" <<'EOF'
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: engram-prometheus
    volumes:
      - ./prometheus:/etc/prometheus
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=15d'
    ports:
      - "9090:9090"
    networks:
      - monitoring
    restart: unless-stopped

  grafana:
    image: grafana/grafana:10.0.0
    container_name: engram-grafana
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_SERVER_ROOT_URL=http://localhost:3000
    ports:
      - "3000:3000"
    networks:
      - monitoring
    restart: unless-stopped
    depends_on:
      - prometheus
      - loki

  loki:
    image: grafana/loki:2.8.0
    container_name: engram-loki
    volumes:
      - ./loki:/etc/loki
      - loki-data:/loki
    command: -config.file=/etc/loki/loki-config.yml
    ports:
      - "3100:3100"
    networks:
      - monitoring
    restart: unless-stopped

  promtail:
    image: grafana/promtail:2.8.0
    container_name: engram-promtail
    volumes:
      - ./promtail:/etc/promtail
      - /var/log:/var/log
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
    command: -config.file=/etc/promtail/promtail-config.yml
    networks:
      - monitoring
    restart: unless-stopped
    depends_on:
      - loki

volumes:
  prometheus-data:
  grafana-data:
  loki-data:

networks:
  monitoring:
    driver: bridge
EOF

    log_info "Starting monitoring stack..."
    cd "${DEPLOYMENTS_DIR}"
    docker-compose up -d

    log_info "Monitoring stack deployed successfully"
    log_info "Grafana available at: http://localhost:3000 (admin/admin)"
    log_info "Prometheus available at: http://localhost:9090"
    log_info "Loki available at: http://localhost:3100"
}

validate_deployment() {
    log_info "Validating deployment..."

    if [ "$1" = "kubernetes" ]; then
        # Check Prometheus is scraping
        PROMETHEUS_POD=$(kubectl get pod -n monitoring -l app=prometheus -o jsonpath='{.items[0].metadata.name}')
        kubectl exec -n monitoring "$PROMETHEUS_POD" -- wget -q -O- http://localhost:9090/-/healthy

        # Check Grafana is running
        GRAFANA_POD=$(kubectl get pod -n monitoring -l app=grafana -o jsonpath='{.items[0].metadata.name}')
        kubectl exec -n monitoring "$GRAFANA_POD" -- wget -q -O- http://localhost:3000/api/health

        # Check Loki is running
        LOKI_POD=$(kubectl get pod -n monitoring -l app=loki -o jsonpath='{.items[0].metadata.name}')
        kubectl exec -n monitoring "$LOKI_POD" -- wget -q -O- http://localhost:3100/ready
    else
        # Docker validation
        docker exec engram-prometheus wget -q -O- http://localhost:9090/-/healthy
        docker exec engram-grafana wget -q -O- http://localhost:3000/api/health
        docker exec engram-loki wget -q -O- http://localhost:3100/ready
    fi

    log_info "Validation passed"
}

show_usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Deploy Engram monitoring stack (Prometheus, Grafana, Loki)

OPTIONS:
    -k, --kubernetes    Deploy to Kubernetes cluster
    -d, --docker        Deploy with Docker Compose
    -v, --validate      Validate deployment after setup
    -h, --help          Show this help message

EXAMPLES:
    $0 --kubernetes         # Deploy to Kubernetes
    $0 --docker --validate  # Deploy with Docker and validate
EOF
}

main() {
    local deployment_type=""
    local validate=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -k|--kubernetes)
                deployment_type="kubernetes"
                shift
                ;;
            -d|--docker)
                deployment_type="docker"
                shift
                ;;
            -v|--validate)
                validate=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    if [ -z "$deployment_type" ]; then
        log_error "Please specify deployment type: --kubernetes or --docker"
        show_usage
        exit 1
    fi

    check_prerequisites

    if [ "$deployment_type" = "kubernetes" ]; then
        deploy_kubernetes
    else
        deploy_docker_compose
    fi

    if [ "$validate" = true ]; then
        sleep 5  # Wait for services to fully start
        validate_deployment "$deployment_type"
    fi

    log_info "Setup complete!"
}

main "$@"
