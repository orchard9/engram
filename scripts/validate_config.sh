#!/usr/bin/env bash
#
# Configuration validation for Engram deployments
# Validates TOML syntax, parameter ranges, and deployment prerequisites
#
# Usage: ./validate_config.sh [config_file] [--deployment ENV]
# Exit codes: 0 = valid, 1 = invalid, 2 = file not found, 3 = missing dependencies

set -euo pipefail

# Configuration defaults
CONFIG_FILE="${1:-config.toml}"
DEPLOYMENT_ENV=""
STRICT_MODE=false
VERBOSE=false

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --deployment)
            DEPLOYMENT_ENV="$2"
            shift 2
            ;;
        --strict)
            STRICT_MODE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [config_file] [options]"
            echo ""
            echo "Options:"
            echo "  --deployment ENV    Validate for deployment environment (dev|staging|production)"
            echo "  --strict            Treat warnings as errors"
            echo "  --verbose           Show detailed validation output"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            CONFIG_FILE="$1"
            shift
            ;;
    esac
done

# Validation counters
ERRORS=0
WARNINGS=0
CHECKS_PASSED=0

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((CHECKS_PASSED++))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARNINGS++))
    if [[ "$STRICT_MODE" == "true" ]]; then
        ((ERRORS++))
    fi
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ((ERRORS++))
}

# Check for required tools
check_dependencies() {
    local missing_deps=()
    
    # Check for basic shell utilities
    for cmd in awk sed grep; do
        if ! command -v $cmd &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        exit 3
    fi
}

# Parse TOML value (simple parser for basic validation)
get_toml_value() {
    local key="$1"
    local file="$2"
    
    # Handle nested keys like "persistence.hot_capacity"
    local section="${key%%.*}"
    local field="${key#*.}"
    
    if [[ "$section" == "$field" ]]; then
        # Top-level key
        grep "^${key}\s*=" "$file" | sed 's/.*=\s*//' | tr -d '"' | tr -d "'" | xargs
    else
        # Nested key - find section first, then key
        awk "/^\[${section}\]/,/^\[/ {if (/^${field}\s*=/) print}" "$file" | \
            sed 's/.*=\s*//' | tr -d '"' | tr -d "'" | xargs
    fi
}

# Validate numeric range
validate_range() {
    local name="$1"
    local value="$2"
    local min="$3"
    local max="$4"
    
    if [[ ! "$value" =~ ^[0-9_]+$ ]]; then
        log_error "$name must be numeric (got: $value)"
        return 1
    fi
    
    # Remove underscores for comparison
    value="${value//_/}"
    
    if (( value < min )); then
        log_error "$name too low: $value (minimum: $min)"
        return 1
    fi
    
    if (( value > max )); then
        log_error "$name too high: $value (maximum: $max)"
        return 1
    fi
    
    log_success "$name in valid range: $value"
    return 0
}

# Validate float range
validate_float_range() {
    local name="$1"
    local value="$2"
    local min="$3"
    local max="$4"
    
    if ! awk "BEGIN {exit !($value >= $min && $value <= $max)}"; then
        log_error "$name out of range: $value (valid: $min-$max)"
        return 1
    fi
    
    log_success "$name in valid range: $value"
    return 0
}

# Validate memory space ID
validate_space_id() {
    local space_id="$1"
    
    # Check length (3-64 characters)
    local len=${#space_id}
    if (( len < 3 || len > 64 )); then
        log_error "Memory space ID length invalid: $len characters (valid: 3-64)"
        return 1
    fi
    
    # Check characters (alphanumeric, hyphens, underscores)
    if [[ ! "$space_id" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        log_error "Memory space ID contains invalid characters: $space_id"
        return 1
    fi
    
    # Check doesn't start with hyphen or underscore
    if [[ "$space_id" =~ ^[_-] ]]; then
        log_error "Memory space ID cannot start with hyphen or underscore: $space_id"
        return 1
    fi
    
    return 0
}

# Main validation
main() {
    log_info "Validating configuration: $CONFIG_FILE"
    echo ""
    
    # Check dependencies
    check_dependencies
    
    # Check file exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        exit 2
    fi
    
    log_success "Configuration file exists"
    
    # Validate TOML syntax (basic check)
    if ! grep -q '^\[' "$CONFIG_FILE"; then
        log_warning "No TOML sections found (may be empty config)"
    fi
    
    # Validate persistence.data_root
    data_root=$(get_toml_value "persistence.data_root" "$CONFIG_FILE")
    if [[ -n "$data_root" ]]; then
        if [[ ! "$data_root" =~ ^(/|~) ]]; then
            log_error "persistence.data_root must be absolute path or start with ~ (got: $data_root)"
        else
            log_success "persistence.data_root is valid: $data_root"
            
            # Expand tilde if present
            if [[ "$data_root" =~ ^~ ]]; then
                data_root="${data_root/#\~/$HOME}"
            fi
            
            # Check if directory exists or is creatable
            parent_dir=$(dirname "$data_root")
            if [[ -d "$parent_dir" ]]; then
                if [[ ! -w "$parent_dir" ]]; then
                    log_warning "Parent directory not writable: $parent_dir"
                fi
            else
                log_warning "Parent directory does not exist: $parent_dir"
            fi
        fi
    else
        log_warning "persistence.data_root not set (will use default)"
    fi
    
    # Validate persistence.hot_capacity
    hot_capacity=$(get_toml_value "persistence.hot_capacity" "$CONFIG_FILE")
    if [[ -n "$hot_capacity" ]]; then
        validate_range "persistence.hot_capacity" "$hot_capacity" 1000 10000000
    else
        log_warning "persistence.hot_capacity not set (will use default: 100,000)"
    fi
    
    # Validate persistence.warm_capacity
    warm_capacity=$(get_toml_value "persistence.warm_capacity" "$CONFIG_FILE")
    if [[ -n "$warm_capacity" ]]; then
        validate_range "persistence.warm_capacity" "$warm_capacity" 10000 100000000
    else
        log_warning "persistence.warm_capacity not set (will use default: 1,000,000)"
    fi
    
    # Validate persistence.cold_capacity
    cold_capacity=$(get_toml_value "persistence.cold_capacity" "$CONFIG_FILE")
    if [[ -n "$cold_capacity" ]]; then
        validate_range "persistence.cold_capacity" "$cold_capacity" 100000 1000000000
    else
        log_warning "persistence.cold_capacity not set (will use default: 10,000,000)"
    fi
    
    # Validate tier ordering
    if [[ -n "$hot_capacity" && -n "$warm_capacity" ]]; then
        hot_val="${hot_capacity//_/}"
        warm_val="${warm_capacity//_/}"
        if (( hot_val > warm_val )); then
            log_warning "hot_capacity ($hot_val) exceeds warm_capacity ($warm_val)"
        fi
    fi
    
    if [[ -n "$warm_capacity" && -n "$cold_capacity" ]]; then
        warm_val="${warm_capacity//_/}"
        cold_val="${cold_capacity//_/}"
        if (( warm_val > cold_val )); then
            log_warning "warm_capacity ($warm_val) exceeds cold_capacity ($cold_val)"
        fi
    fi
    
    # Validate memory_spaces.default_space
    default_space=$(get_toml_value "memory_spaces.default_space" "$CONFIG_FILE")
    if [[ -n "$default_space" ]]; then
        if validate_space_id "$default_space"; then
            log_success "memory_spaces.default_space is valid: $default_space"
        fi
    else
        log_warning "memory_spaces.default_space not set (will use default: 'default')"
    fi
    
    # Validate memory_spaces.bootstrap_spaces
    bootstrap_spaces=$(get_toml_value "memory_spaces.bootstrap_spaces" "$CONFIG_FILE")
    if [[ -n "$bootstrap_spaces" ]]; then
        # Parse array (simplified - assumes clean formatting)
        bootstrap_spaces=$(echo "$bootstrap_spaces" | tr -d '[]' | tr ',' '\n')
        space_count=0
        while IFS= read -r space; do
            space=$(echo "$space" | xargs)
            if [[ -n "$space" ]]; then
                if validate_space_id "$space"; then
                    ((space_count++))
                fi
            fi
        done <<< "$bootstrap_spaces"
        
        if (( space_count > 0 )); then
            log_success "memory_spaces.bootstrap_spaces validated: $space_count space(s)"
        fi
    else
        log_warning "memory_spaces.bootstrap_spaces not set (will use default: ['default'])"
    fi
    
    # Validate feature_flags.spreading_api_beta
    spreading_beta=$(get_toml_value "feature_flags.spreading_api_beta" "$CONFIG_FILE")
    if [[ -n "$spreading_beta" ]]; then
        if [[ "$spreading_beta" =~ ^(true|false)$ ]]; then
            log_success "feature_flags.spreading_api_beta is valid: $spreading_beta"
        else
            log_error "feature_flags.spreading_api_beta must be boolean (got: $spreading_beta)"
        fi
    else
        log_warning "feature_flags.spreading_api_beta not set (will use default: true)"
    fi
    
    # Deployment-specific validation
    if [[ -n "$DEPLOYMENT_ENV" ]]; then
        echo ""
        log_info "Deployment environment: $DEPLOYMENT_ENV"
        
        case "$DEPLOYMENT_ENV" in
            production|prod)
                # Production-specific checks
                if [[ -n "$hot_capacity" ]]; then
                    hot_val="${hot_capacity//_/}"
                    if (( hot_val < 100000 )); then
                        log_warning "Production hot_capacity is low: $hot_val (recommend: >=100,000)"
                    fi
                fi
                
                if [[ "$data_root" =~ ^~ ]]; then
                    log_warning "Production using home directory for data_root (recommend: /var/lib/engram)"
                fi
                
                log_info "Production checklist:"
                echo "  - Configure monitoring (Prometheus, Grafana)"
                echo "  - Enable TLS certificates"
                echo "  - Configure backup retention"
                echo "  - Set up alerting rules"
                echo "  - Configure authentication (JWT)"
                ;;
            
            staging)
                log_info "Staging checklist:"
                echo "  - Mirror production configuration at smaller scale"
                echo "  - Test backup/restore procedures"
                echo "  - Validate performance under load"
                ;;
            
            dev|development)
                log_info "Development environment detected"
                if [[ -n "$hot_capacity" ]]; then
                    hot_val="${hot_capacity//_/}"
                    if (( hot_val > 100000 )); then
                        log_warning "Development hot_capacity is high: $hot_val (may use excessive RAM)"
                    fi
                fi
                ;;
            
            *)
                log_warning "Unknown deployment environment: $DEPLOYMENT_ENV"
                ;;
        esac
    fi
    
    # Summary
    echo ""
    echo "=========================================="
    echo "Validation Summary"
    echo "=========================================="
    echo "Checks passed: $CHECKS_PASSED"
    echo "Warnings:      $WARNINGS"
    echo "Errors:        $ERRORS"
    echo "=========================================="
    
    if (( ERRORS > 0 )); then
        echo -e "${RED}Configuration validation FAILED${NC}"
        exit 1
    elif (( WARNINGS > 0 )); then
        echo -e "${YELLOW}Configuration validation PASSED with warnings${NC}"
        if [[ "$STRICT_MODE" == "true" ]]; then
            exit 1
        fi
        exit 0
    else
        echo -e "${GREEN}Configuration validation PASSED${NC}"
        exit 0
    fi
}

# Run main function
main
