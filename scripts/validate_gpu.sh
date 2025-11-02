#!/bin/bash
#
# GPU Validation Script for Engram
# 
# This script validates GPU functionality according to the requirements
# documented in docs/operations/gpu_testing_requirements.md
#
# Usage:
#   ./scripts/validate_gpu.sh [phase]
#
# Phases:
#   all         - Run all validation phases (default)
#   foundation  - Run foundation tests only
#   integration - Run integration tests only
#   sustained   - Run sustained load tests (60+ minutes)
#   production  - Run production workload validation
#   quick       - Run quick smoke tests only

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default phase
PHASE="${1:-all}"

# Log file
LOG_DIR="./tmp/gpu_validation"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/validation_$(date +%Y%m%d_%H%M%S).log"

# Function to print colored output
print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}" | tee -a "$LOG_FILE"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}" | tee -a "$LOG_FILE"
}

print_error() {
    echo -e "${RED}✗ $1${NC}" | tee -a "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}" | tee -a "$LOG_FILE"
}

print_info() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Function to run a test and check result
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    print_info "Running: $test_name"
    
    if eval "$test_command" >> "$LOG_FILE" 2>&1; then
        print_success "$test_name passed"
        return 0
    else
        print_error "$test_name failed"
        return 1
    fi
}

# Function to check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check for NVIDIA GPU
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found. NVIDIA drivers not installed."
        exit 1
    fi
    
    # Display GPU info
    print_info "GPU Information:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a "$LOG_FILE"
    
    # Check for CUDA
    if ! command -v nvcc &> /dev/null; then
        print_error "nvcc not found. CUDA toolkit not installed."
        exit 1
    fi
    
    # Display CUDA version
    print_info "CUDA Version:"
    nvcc --version | grep "release" | tee -a "$LOG_FILE"
    
    # Build with GPU support
    print_info "Building Engram with GPU support..."
    if cargo build --release --features gpu >> "$LOG_FILE" 2>&1; then
        print_success "Build successful"
    else
        print_error "Build failed. Check $LOG_FILE for details."
        exit 1
    fi
    
    # Verify GPU detection
    print_info "Verifying GPU detection..."
    if cargo test --features gpu test_cuda_device_detection -- --nocapture >> "$LOG_FILE" 2>&1; then
        print_success "GPU detection successful"
    else
        print_error "GPU detection failed"
        exit 1
    fi
}

# Phase 1: Foundation Tests
run_foundation_tests() {
    print_header "Phase 1: Foundation Tests (5-10 minutes)"
    
    local failed=0
    
    # GPU acceleration tests
    run_test "GPU acceleration foundation" \
        "cargo test --features gpu gpu_acceleration_test -- --nocapture" || ((failed++))
    
    # Differential tests
    run_test "CPU-GPU differential cosine similarity" \
        "cargo test --features gpu gpu_differential_cosine -- --nocapture" || ((failed++))
    
    run_test "CPU-GPU differential HNSW" \
        "cargo test --features gpu gpu_differential_hnsw -- --nocapture" || ((failed++))
    
    run_test "CPU-GPU differential spreading" \
        "cargo test --features gpu gpu_differential_spreading -- --nocapture" || ((failed++))
    
    if [ $failed -eq 0 ]; then
        print_success "All foundation tests passed"
    else
        print_error "$failed foundation tests failed"
        return 1
    fi
}

# Phase 2: Integration Tests
run_integration_tests() {
    print_header "Phase 2: Integration Tests (15-20 minutes)"
    
    local failed=0
    
    # Main integration test suite
    run_test "GPU integration tests" \
        "cargo test --features gpu gpu_integration -- --nocapture" || ((failed++))
    
    # Production readiness tests
    run_test "GPU production readiness" \
        "cargo test --features gpu gpu_production_readiness -- --nocapture" || ((failed++))
    
    if [ $failed -eq 0 ]; then
        print_success "All integration tests passed"
    else
        print_error "$failed integration tests failed"
        return 1
    fi
}

# Phase 3: Sustained Load Tests
run_sustained_tests() {
    print_header "Phase 3: Sustained Load Tests (60+ minutes)"
    print_warning "This phase takes approximately 60 minutes to complete"
    
    local failed=0
    
    # Monitor GPU during test
    print_info "Starting GPU monitoring in background..."
    nvidia-smi dmon -s pucvmet -i 0 -c 3600 > "$LOG_DIR/gpu_metrics_sustained.csv" 2>&1 &
    NVIDIA_SMI_PID=$!
    
    # Sustained throughput test
    run_test "Sustained throughput (60 minutes)" \
        "cargo test --features gpu test_sustained_throughput_fixed -- --ignored --nocapture" || ((failed++))
    
    # Confidence drift test
    run_test "Confidence drift over time" \
        "cargo test --features gpu test_confidence_drift_over_time -- --ignored --nocapture" || ((failed++))
    
    # Stop GPU monitoring
    kill $NVIDIA_SMI_PID 2>/dev/null || true
    
    # Analyze GPU metrics
    print_info "GPU metrics summary:"
    if [ -f "$LOG_DIR/gpu_metrics_sustained.csv" ]; then
        awk -F'[[:space:]]+' '
        NR > 2 {
            gpu_util += $3; 
            mem_util += $5; 
            temp += $9; 
            count++
        } 
        END {
            if (count > 0) {
                printf "  Average GPU Utilization: %.1f%%\n", gpu_util/count;
                printf "  Average Memory Utilization: %.1f%%\n", mem_util/count;
                printf "  Average Temperature: %.1f°C\n", temp/count;
            }
        }' "$LOG_DIR/gpu_metrics_sustained.csv" | tee -a "$LOG_FILE"
    fi
    
    if [ $failed -eq 0 ]; then
        print_success "All sustained load tests passed"
    else
        print_error "$failed sustained load tests failed"
        return 1
    fi
}

# Phase 4: Production Workload Validation
run_production_tests() {
    print_header "Phase 4: Production Workload Validation (30-60 minutes)"
    
    local failed=0
    
    # Production workload patterns
    run_test "Production workload - social graph" \
        "cargo test --features gpu test_production_workload_social_graph -- --nocapture" || ((failed++))
    
    run_test "Production workload - knowledge graph" \
        "cargo test --features gpu test_production_workload_knowledge_graph -- --nocapture" || ((failed++))
    
    # Multi-tenant security
    run_test "Multi-tenant resource exhaustion" \
        "cargo test --features gpu test_multi_tenant_resource_exhaustion -- --nocapture" || ((failed++))
    
    run_test "Multi-tenant security isolation" \
        "cargo test --features gpu test_multi_tenant_security_isolation -- --nocapture" || ((failed++))
    
    run_test "Multi-tenant GPU fairness" \
        "cargo test --features gpu test_multi_tenant_gpu_fairness_concurrent -- --nocapture" || ((failed++))
    
    # Confidence calibration
    run_test "Confidence calibration validation" \
        "cargo test --features gpu test_confidence_calibration_statistical_validation -- --nocapture" || ((failed++))
    
    # Chaos engineering
    run_test "Chaos - GPU OOM injection" \
        "cargo test --features gpu test_chaos_gpu_oom_injection -- --nocapture" || ((failed++))
    
    run_test "Chaos - Concurrent GPU access" \
        "cargo test --features gpu test_chaos_concurrent_gpu_access -- --nocapture" || ((failed++))
    
    if [ $failed -eq 0 ]; then
        print_success "All production workload tests passed"
    else
        print_error "$failed production workload tests failed"
        return 1
    fi
}

# Quick smoke tests
run_quick_tests() {
    print_header "Quick GPU Smoke Tests (2-5 minutes)"
    
    local failed=0
    
    # Basic GPU functionality
    run_test "GPU device detection" \
        "cargo test --features gpu test_cuda_device_detection -- --nocapture" || ((failed++))
    
    run_test "GPU memory allocation" \
        "cargo test --features gpu test_gpu_memory_allocation -- --nocapture" || ((failed++))
    
    run_test "GPU kernel launch" \
        "cargo test --features gpu test_gpu_kernel_launch -- --nocapture" || ((failed++))
    
    # Basic differential test
    run_test "Quick differential test" \
        "cargo test --features gpu test_gpu_cpu_consistency -- --nocapture" || ((failed++))
    
    if [ $failed -eq 0 ]; then
        print_success "All smoke tests passed"
    else
        print_error "$failed smoke tests failed"
        return 1
    fi
}

# Run benchmark suite
run_benchmarks() {
    print_header "GPU Performance Benchmarks"
    
    print_info "Running GPU benchmarks..."
    
    # GPU performance validation
    cargo bench --features gpu gpu_performance_validation -- --nocapture >> "$LOG_FILE" 2>&1
    
    # Individual component benchmarks
    cargo bench --features gpu gpu_cosine_similarity -- --nocapture >> "$LOG_FILE" 2>&1
    cargo bench --features gpu gpu_hnsw -- --nocapture >> "$LOG_FILE" 2>&1
    cargo bench --features gpu gpu_spreading -- --nocapture >> "$LOG_FILE" 2>&1
    
    print_success "Benchmarks completed"
}

# Generate final report
generate_report() {
    print_header "GPU Validation Report"
    
    # Run diagnostics
    print_info "Running Engram diagnostics..."
    if [ -f "./scripts/engram_diagnostics.sh" ]; then
        ./scripts/engram_diagnostics.sh >> "$LOG_FILE" 2>&1
    fi
    
    # Summary
    print_info "\nValidation Summary:"
    print_info "  Log file: $LOG_FILE"
    print_info "  GPU metrics: $LOG_DIR/gpu_metrics_sustained.csv"
    
    # Check for errors in log
    local error_count=$(grep -c "✗" "$LOG_FILE" || true)
    local warning_count=$(grep -c "⚠" "$LOG_FILE" || true)
    
    if [ $error_count -eq 0 ]; then
        print_success "GPU validation completed successfully!"
    else
        print_error "GPU validation completed with $error_count errors"
        print_warning "Review log file for details: $LOG_FILE"
    fi
    
    if [ $warning_count -gt 0 ]; then
        print_warning "Found $warning_count warnings - review before production deployment"
    fi
}

# Main execution
main() {
    print_header "Engram GPU Validation Script"
    print_info "Phase: $PHASE"
    print_info "Log file: $LOG_FILE"
    
    # Always check prerequisites
    check_prerequisites
    
    # Track overall success
    local overall_failed=0
    
    # Run requested phases
    case "$PHASE" in
        foundation)
            run_foundation_tests || ((overall_failed++))
            ;;
        integration)
            run_integration_tests || ((overall_failed++))
            ;;
        sustained)
            run_sustained_tests || ((overall_failed++))
            ;;
        production)
            run_production_tests || ((overall_failed++))
            ;;
        quick)
            run_quick_tests || ((overall_failed++))
            ;;
        benchmarks)
            run_benchmarks || ((overall_failed++))
            ;;
        all)
            run_foundation_tests || ((overall_failed++))
            run_integration_tests || ((overall_failed++))
            run_production_tests || ((overall_failed++))
            
            # Ask before running sustained tests
            print_warning "Sustained load tests take 60+ minutes. Run them? (y/N)"
            read -r response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                run_sustained_tests || ((overall_failed++))
            else
                print_info "Skipping sustained load tests"
            fi
            
            # Run benchmarks
            run_benchmarks || ((overall_failed++))
            ;;
        *)
            print_error "Unknown phase: $PHASE"
            print_info "Valid phases: all, foundation, integration, sustained, production, quick, benchmarks"
            exit 1
            ;;
    esac
    
    # Generate final report
    generate_report
    
    # Exit with appropriate code
    exit $overall_failed
}

# Run main function
main