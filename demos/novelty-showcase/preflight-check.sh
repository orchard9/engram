#!/bin/bash
# Pre-flight check for Engram Novelty Showcase Demo
# Verifies all prerequisites are met before running the demo

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

CHECKS_PASSED=0
CHECKS_FAILED=0

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

check_pass() {
    echo -e "${GREEN}✓ $1${NC}"
    ((CHECKS_PASSED++))
}

check_fail() {
    echo -e "${RED}✗ $1${NC}"
    ((CHECKS_FAILED++))
}

check_warn() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_header "Engram Demo Pre-flight Check"

echo "This script verifies all prerequisites for the novelty showcase demo."
echo ""

# Check 1: curl installed
echo -n "Checking for curl... "
if command -v curl &> /dev/null; then
    CURL_VERSION=$(curl --version | head -n1)
    check_pass "curl installed ($CURL_VERSION)"
else
    check_fail "curl not found (required for API calls)"
    echo "  Install: apt-get install curl (Linux) or brew install curl (macOS)"
fi

# Check 2: jq installed
echo -n "Checking for jq... "
if command -v jq &> /dev/null; then
    JQ_VERSION=$(jq --version 2>&1)
    check_pass "jq installed ($JQ_VERSION)"
else
    check_fail "jq not found (required for JSON formatting)"
    echo "  Install: apt-get install jq (Linux) or brew install jq (macOS)"
fi

# Check 3: Bash version
echo -n "Checking Bash version... "
BASH_VERSION_NUM=${BASH_VERSION%%.*}
if [ "$BASH_VERSION_NUM" -ge 4 ]; then
    check_pass "Bash $BASH_VERSION (>= 4.0 required)"
else
    check_warn "Bash $BASH_VERSION (4.0+ recommended)"
fi

# Check 4: date command (macOS vs Linux compatibility)
echo -n "Checking date command... "
if date -u -v-1d +"%Y-%m-%dT%H:%M:%SZ" &> /dev/null; then
    check_pass "date command (macOS format)"
elif date -u -d '1 day ago' +"%Y-%m-%dT%H:%M:%SZ" &> /dev/null; then
    check_pass "date command (Linux format)"
else
    check_fail "date command format not recognized"
    echo "  The demo may fail on timestamp generation"
fi

# Check 5: Engram binary exists
echo -n "Checking for Engram binary... "
if [ -f "../../target/release/engram" ]; then
    check_pass "Engram binary found (release)"
elif [ -f "../../target/debug/engram" ]; then
    check_warn "Engram binary found (debug build)"
    echo "  Note: Release build recommended for accurate performance metrics"
else
    check_fail "Engram binary not found"
    echo "  Build with: cd ../../ && cargo build --release"
fi

# Check 6: Engram server running
echo -n "Checking if Engram server is running... "
if curl -s http://localhost:7432/health > /dev/null 2>&1; then
    SERVER_STATUS=$(curl -s http://localhost:7432/health | jq -r '.status // "unknown"' 2>/dev/null || echo "unknown")
    check_pass "Server running at http://localhost:7432 (status: $SERVER_STATUS)"
else
    check_fail "Server not running at http://localhost:7432"
    echo "  Start with: ../../target/release/engram start"
fi

# Check 7: Server API endpoints accessible
if curl -s http://localhost:7432/health > /dev/null 2>&1; then
    echo -n "Checking API endpoints... "

    ENDPOINTS_OK=true

    # Check memory API
    if ! curl -s -f http://localhost:7432/api/v1/memories/recall?query=test > /dev/null 2>&1; then
        ENDPOINTS_OK=false
    fi

    # Check episode API
    if ! curl -s -f http://localhost:7432/api/v1/episodes/list > /dev/null 2>&1; then
        ENDPOINTS_OK=false
    fi

    # Check consolidation API
    if ! curl -s -f http://localhost:7432/api/v1/consolidations > /dev/null 2>&1; then
        ENDPOINTS_OK=false
    fi

    if [ "$ENDPOINTS_OK" = true ]; then
        check_pass "All required API endpoints accessible"
    else
        check_warn "Some API endpoints may not be accessible"
        echo "  The demo may encounter errors"
    fi
fi

# Check 8: Demo script executable
echo -n "Checking demo script permissions... "
if [ -x "./demo.sh" ]; then
    check_pass "demo.sh is executable"
else
    check_warn "demo.sh is not executable"
    echo "  Fix with: chmod +x demo.sh"
fi

# Check 9: Network connectivity
echo -n "Checking network connectivity... "
if curl -s --max-time 2 http://localhost:7432/health > /dev/null 2>&1; then
    check_pass "Network connectivity to localhost:7432"
else
    check_fail "Cannot reach localhost:7432"
fi

# Check 10: Disk space for demo data
echo -n "Checking available disk space... "
if command -v df &> /dev/null; then
    AVAILABLE_KB=$(df -k . | tail -1 | awk '{print $4}')
    AVAILABLE_MB=$((AVAILABLE_KB / 1024))

    if [ $AVAILABLE_MB -gt 100 ]; then
        check_pass "Sufficient disk space (${AVAILABLE_MB}MB available)"
    else
        check_warn "Low disk space (${AVAILABLE_MB}MB available)"
        echo "  Demo requires minimal space, but may fail if disk is full"
    fi
else
    check_warn "Cannot check disk space (df command not available)"
fi

# Summary
print_header "Pre-flight Check Summary"

echo "Checks passed: $CHECKS_PASSED"
echo "Checks failed: $CHECKS_FAILED"
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! Ready to run demo.${NC}"
    echo ""
    echo "Start the demo with:"
    echo "  ./demo.sh"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some checks failed. Please fix the issues above before running the demo.${NC}"
    echo ""
    echo "Common fixes:"
    echo "  • Install missing tools: brew install curl jq (macOS) or apt-get install curl jq (Linux)"
    echo "  • Build Engram: cd ../../ && cargo build --release"
    echo "  • Start server: ../../target/release/engram start"
    echo "  • Make script executable: chmod +x demo.sh"
    echo ""
    exit 1
fi
