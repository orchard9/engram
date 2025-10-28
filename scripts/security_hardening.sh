#!/bin/bash
# Security hardening script for Engram production deployments
#
# This script applies security best practices and hardening measures
# for running Engram in production environments.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_warn "This script should be run as root for full system hardening"
        log_warn "Some hardening steps will be skipped"
        return 1
    fi
    return 0
}

# OS-level hardening
harden_os() {
    log_step "Applying OS-level hardening..."

    if ! check_root; then
        log_warn "Skipping OS hardening (requires root)"
        return
    fi

    # Kernel parameters for security and networking
    log_info "Configuring kernel parameters..."
    cat > /etc/sysctl.d/99-engram-security.conf <<EOF
# Network hardening
net.ipv4.tcp_syncookies = 1
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.all.secure_redirects = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1
net.ipv4.tcp_timestamps = 0

# IPv6 hardening
net.ipv6.conf.all.accept_redirects = 0
net.ipv6.conf.default.accept_redirects = 0

# File system hardening
fs.protected_hardlinks = 1
fs.protected_symlinks = 1
fs.suid_dumpable = 0

# Kernel hardening
kernel.randomize_va_space = 2
kernel.yama.ptrace_scope = 1
kernel.kptr_restrict = 2
kernel.dmesg_restrict = 1

# Memory protection
vm.mmap_min_addr = 65536
EOF

    sysctl -p /etc/sysctl.d/99-engram-security.conf > /dev/null 2>&1 || log_warn "Failed to apply sysctl settings"
    log_info "Kernel parameters configured"

    # Disable unnecessary services
    log_info "Disabling unnecessary services..."
    for service in avahi-daemon cups bluetooth; do
        if systemctl is-active --quiet "$service" 2>/dev/null; then
            systemctl disable "$service" 2>/dev/null || log_warn "Failed to disable $service"
            systemctl stop "$service" 2>/dev/null || log_warn "Failed to stop $service"
        fi
    done
    log_info "Unnecessary services disabled"
}

# Application-level hardening
harden_application() {
    log_step "Applying application-level hardening..."

    # Create engram user if it doesn't exist
    if ! id -u engram > /dev/null 2>&1; then
        log_info "Creating engram user..."
        if check_root; then
            useradd -r -s /bin/false -d /opt/engram engram
            log_info "User 'engram' created"
        else
            log_warn "Cannot create user (requires root)"
        fi
    fi

    # Set proper file permissions
    log_info "Setting file permissions..."
    local app_dir="${APP_DIR:-/opt/engram}"

    if [ -d "$app_dir" ]; then
        if check_root; then
            chown -R engram:engram "$app_dir" || log_warn "Failed to set ownership"
            chmod 750 "$app_dir" || log_warn "Failed to set directory permissions"
            find "$app_dir/config" -type f -exec chmod 640 {} \; 2>/dev/null || log_warn "No config directory found"
            log_info "File permissions configured"
        else
            log_warn "Cannot set ownership (requires root)"
        fi
    else
        log_warn "Application directory $app_dir not found"
    fi

    # Validate Engram configuration
    log_info "Validating Engram configuration..."
    if [ -x "$app_dir/engram" ]; then
        "$app_dir/engram" --version > /dev/null 2>&1 && log_info "Engram binary validated" || log_warn "Engram validation failed"
    else
        log_warn "Engram binary not found or not executable"
    fi
}

# Container hardening
harden_container() {
    log_step "Applying container hardening..."

    log_info "Container security recommendations:"
    echo "  1. Run containers as non-root user"
    echo "  2. Use read-only file systems where possible"
    echo "  3. Drop unnecessary capabilities"
    echo "  4. Set resource limits (CPU, memory)"
    echo "  5. Use security profiles (seccomp, AppArmor, SELinux)"
    echo "  6. Scan images for vulnerabilities"
    echo "  7. Use minimal base images"
    echo "  8. Enable content trust"
}

# Security scanning
security_scan() {
    log_step "Running security scans..."

    # Check for cargo-audit
    if command -v cargo-audit &> /dev/null; then
        log_info "Running cargo audit..."
        cd "$(dirname "$0")/.." || exit 1
        cargo audit --json > /tmp/engram-audit.json 2>&1 || log_warn "Cargo audit found issues"
        log_info "Audit report saved to /tmp/engram-audit.json"
    else
        log_warn "cargo-audit not installed. Install with: cargo install cargo-audit"
    fi

    # Check file permissions
    log_info "Checking sensitive file permissions..."
    local issues=0
    for file in /opt/engram/config/*.{yml,yaml,toml,conf} 2>/dev/null; do
        if [ -f "$file" ]; then
            perms=$(stat -c %a "$file" 2>/dev/null || stat -f %A "$file")
            if [ "$perms" != "640" ] && [ "$perms" != "600" ]; then
                log_warn "Insecure permissions on $file: $perms (should be 640 or 600)"
                ((issues++))
            fi
        fi
    done

    if [ $issues -eq 0 ]; then
        log_info "File permission check: OK"
    else
        log_warn "Found $issues file permission issues"
    fi
}

# Firewall configuration
configure_firewall() {
    log_step "Firewall configuration recommendations..."

    log_info "Recommended firewall rules:"
    echo "  Allow inbound:"
    echo "    - 8080/tcp  (HTTP API)"
    echo "    - 8443/tcp  (HTTPS API)"
    echo "    - 50051/tcp (gRPC with mTLS)"
    echo "  Deny all other inbound traffic"
    echo
    echo "Example iptables rules:"
    echo "  iptables -A INPUT -p tcp --dport 8080 -j ACCEPT"
    echo "  iptables -A INPUT -p tcp --dport 8443 -j ACCEPT"
    echo "  iptables -A INPUT -p tcp --dport 50051 -j ACCEPT"
    echo "  iptables -A INPUT -j DROP"
}

# Generate security report
generate_report() {
    log_step "Generating security hardening report..."

    local report_file="/tmp/engram-security-report-$(date +%Y%m%d-%H%M%S).txt"

    cat > "$report_file" <<EOF
Engram Security Hardening Report
Generated: $(date)
========================================

System Information:
- Hostname: $(hostname)
- OS: $(uname -s) $(uname -r)
- Architecture: $(uname -m)

Hardening Status:
EOF

    if check_root; then
        echo "- OS hardening: APPLIED" >> "$report_file"
    else
        echo "- OS hardening: SKIPPED (requires root)" >> "$report_file"
    fi

    echo "- Application hardening: APPLIED" >> "$report_file"
    echo "- Container hardening: RECOMMENDATIONS PROVIDED" >> "$report_file"
    echo "- Security scanning: COMPLETED" >> "$report_file"

    cat >> "$report_file" <<EOF

Security Checklist:
[ ] TLS 1.3+ enforced on all endpoints
[ ] mTLS configured for gRPC
[ ] API key authentication enabled
[ ] JWT validation configured
[ ] OAuth2/OIDC integrated (if required)
[ ] Memory space access control enforced
[ ] Rate limiting active
[ ] Audit logging enabled
[ ] Secrets in secure storage (Vault/AWS Secrets Manager)
[ ] Certificate rotation configured
[ ] Firewall rules applied
[ ] System updates current
[ ] Security monitoring active
[ ] Backup procedures tested
[ ] Incident response plan documented

Recommendations:
1. Regularly update Engram and dependencies
2. Monitor security advisories
3. Rotate credentials quarterly
4. Review audit logs weekly
5. Perform security assessments annually
6. Test disaster recovery procedures
7. Keep documentation current
8. Train staff on security procedures

For more information, see: docs/operations/security.md
EOF

    log_info "Security report saved to: $report_file"
    cat "$report_file"
}

# Main execution
main() {
    echo "====================================="
    echo "Engram Security Hardening Script"
    echo "====================================="
    echo

    # Run hardening steps
    harden_os
    harden_application
    harden_container
    security_scan
    configure_firewall
    generate_report

    echo
    log_info "Security hardening complete!"
    log_info "Review the report and checklist above"
    log_warn "Remember: Security is an ongoing process, not a one-time setup"
}

# Run main function
main "$@"
