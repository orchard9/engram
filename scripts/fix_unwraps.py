#!/usr/bin/env python3
"""
Automated script to help identify and fix unsafe error handling patterns.

This script finds unwrap(), expect(), and panic!() calls in production code
and provides suggestions for fixing them using the new EngramError infrastructure.

Usage: python scripts/fix_unwraps.py src/
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict


class UnsafePattern:
    """Represents an unsafe error handling pattern found in code."""
    
    def __init__(self, file_path: str, line_num: int, pattern_type: str, 
                 match: str, context: str):
        self.file_path = file_path
        self.line_num = line_num
        self.pattern_type = pattern_type
        self.match = match
        self.context = context


def find_unsafe_patterns(file_path: Path) -> List[UnsafePattern]:
    """Find all unsafe error handling patterns in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except (UnicodeDecodeError, IOError):
        print(f"Warning: Could not read {file_path}")
        return []
    
    patterns = []
    
    # Find .unwrap() calls (but not unwrap_or or unwrap_or_else)
    unwraps = re.finditer(r'\.unwrap\(\)(?!\w)', content)
    for match in unwraps:
        line_num = content[:match.start()].count('\n') + 1
        context = get_line_context(content, match.start())
        patterns.append(UnsafePattern(
            str(file_path), line_num, 'unwrap', match.group(), context
        ))
    
    # Find .expect() calls
    expects = re.finditer(r'\.expect\([^)]+\)', content)
    for match in expects:
        line_num = content[:match.start()].count('\n') + 1
        context = get_line_context(content, match.start())
        patterns.append(UnsafePattern(
            str(file_path), line_num, 'expect', match.group(), context
        ))
    
    # Find panic! calls (excluding test functions)
    panics = re.finditer(r'panic!\([^)]+\)', content)
    for match in panics:
        line_num = content[:match.start()].count('\n') + 1
        context = get_line_context(content, match.start())
        
        # Skip if in test function
        if not is_in_test_function(content, match.start()):
            patterns.append(UnsafePattern(
                str(file_path), line_num, 'panic', match.group(), context
            ))
    
    return patterns


def get_line_context(content: str, pos: int) -> str:
    """Get the full line containing the position."""
    start = content.rfind('\n', 0, pos) + 1
    end = content.find('\n', pos)
    if end == -1:
        end = len(content)
    return content[start:end].strip()


def is_in_test_function(content: str, pos: int) -> bool:
    """Check if position is within a test function."""
    # Look backwards for function definition
    before_pos = content[:pos]
    
    # Find the most recent function definition
    func_matches = list(re.finditer(r'\n\s*#?\[?test\]?\s*\n\s*fn\s+test_|#\[test\]\s*\n\s*fn\s+', before_pos))
    if not func_matches:
        return False
    
    last_func = func_matches[-1]
    
    # Check if we're still inside that function (rough heuristic)
    after_func = content[last_func.end():pos]
    open_braces = after_func.count('{')
    close_braces = after_func.count('}')
    
    return open_braces > close_braces


def suggest_fix(pattern_type: str, context: str, match: str) -> str:
    """Suggest a fix for each type of unsafe pattern."""
    if pattern_type == 'unwrap':
        if 'Option' in context or 'Some(' in context or 'None' in context:
            return "Replace with '?' operator: 'value.ok_or_else(|| EngramError::...)?'"
        else:
            return "Replace with '?' operator: 'value.map_err(|e| EngramError::from(e))?'"
            
    elif pattern_type == 'expect':
        return ("Replace with proper error handling:\n"
               "    'value.map_err(|e| EngramError::memory_error(\"...\", e, RecoveryStrategy::...))?'")
               
    elif pattern_type == 'panic':
        return ("Replace with error return:\n"
               "    'return Err(EngramError::pattern_match_error(\"...\", RecoveryStrategy::...))'")
    
    return "Manual review required"


def categorize_by_severity(patterns: List[UnsafePattern]) -> Dict[str, List[UnsafePattern]]:
    """Categorize patterns by severity/priority."""
    categories = {
        'critical': [],    # Production panics
        'high': [],        # .unwrap() in production code
        'medium': [],      # .expect() calls
        'low': []          # Safe unwraps (tests, or with good reason)
    }
    
    for pattern in patterns:
        if pattern.pattern_type == 'panic':
            categories['critical'].append(pattern)
        elif pattern.pattern_type == 'unwrap':
            # Check if it's in a critical path
            if any(path in pattern.file_path.lower() for path in ['storage', 'wal', 'memory', 'query']):
                categories['high'].append(pattern)
            else:
                categories['medium'].append(pattern)
        elif pattern.pattern_type == 'expect':
            categories['medium'].append(pattern)
    
    return categories


def print_analysis_summary(categories: Dict[str, List[UnsafePattern]]):
    """Print a summary of the analysis."""
    total = sum(len(patterns) for patterns in categories.values())
    
    print(f"\n{'='*60}")
    print("UNSAFE ERROR HANDLING ANALYSIS")
    print(f"{'='*60}")
    print(f"Total unsafe patterns found: {total}")
    print()
    
    for severity, patterns in categories.items():
        if patterns:
            print(f"{severity.upper()} ({len(patterns)} patterns):")
            print("-" * 40)
            
            # Group by file
            by_file = {}
            for pattern in patterns:
                file_path = pattern.file_path
                if file_path not in by_file:
                    by_file[file_path] = []
                by_file[file_path].append(pattern)
            
            for file_path, file_patterns in by_file.items():
                print(f"\n📁 {file_path}:")
                for pattern in file_patterns[:5]:  # Show first 5
                    print(f"  Line {pattern.line_num}: {pattern.pattern_type} - {pattern.match}")
                    print(f"    Context: {pattern.context[:80]}...")
                    print(f"    Fix: {suggest_fix(pattern.pattern_type, pattern.context, pattern.match)}")
                
                if len(file_patterns) > 5:
                    print(f"    ... and {len(file_patterns) - 5} more")
            print()


def generate_migration_checklist(categories: Dict[str, List[UnsafePattern]]):
    """Generate a migration checklist ordered by priority."""
    print("\n📋 MIGRATION CHECKLIST")
    print("=" * 50)
    
    priority_order = ['critical', 'high', 'medium', 'low']
    task_num = 1
    
    for severity in priority_order:
        patterns = categories[severity]
        if not patterns:
            continue
            
        print(f"\n{severity.upper()} PRIORITY:")
        print("-" * 30)
        
        # Group by file for easier migration
        by_file = {}
        for pattern in patterns:
            file_path = pattern.file_path
            if file_path not in by_file:
                by_file[file_path] = []
            by_file[file_path].append(pattern)
        
        for file_path, file_patterns in by_file.items():
            print(f"  {task_num}. Fix {len(file_patterns)} patterns in {file_path}")
            
            # Show specific lines for critical/high priority
            if severity in ['critical', 'high']:
                for pattern in file_patterns:
                    print(f"     - Line {pattern.line_num}: {pattern.match}")
            
            task_num += 1


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_unwraps.py <directory>")
        print("Example: python fix_unwraps.py src/")
        sys.exit(1)
    
    directory = Path(sys.argv[1])
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)
    
    print("🔍 Scanning for unsafe error handling patterns...")
    print(f"Directory: {directory.absolute()}")
    
    all_patterns = []
    rust_files = []
    
    # Collect all Rust files
    for rust_file in directory.rglob("*.rs"):
        # Skip certain directories
        if any(skip in str(rust_file) for skip in ['target/', '.git/', 'tests/']):
            continue
        rust_files.append(rust_file)
    
    print(f"Found {len(rust_files)} Rust files to analyze")
    
    # Analyze each file
    for i, rust_file in enumerate(rust_files):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(rust_files)} files...")
        
        patterns = find_unsafe_patterns(rust_file)
        all_patterns.extend(patterns)
    
    # Categorize and analyze
    categories = categorize_by_severity(all_patterns)
    
    # Print results
    print_analysis_summary(categories)
    generate_migration_checklist(categories)
    
    # Final recommendations
    total_critical = len(categories['critical'])
    total_high = len(categories['high'])
    
    print("\n🎯 NEXT STEPS:")
    print("-" * 20)
    
    if total_critical > 0:
        print(f"❌ URGENT: Fix {total_critical} critical panic! patterns immediately")
    
    if total_high > 0:
        print(f"⚠️  HIGH: Fix {total_high} critical unwrap patterns in production paths")
    
    print("\n💡 RECOMMENDED TOOLS:")
    print("- Use the new EngramError types with RecoveryStrategy")
    print("- Apply try_unwrap! and try_expect! macros for gradual migration")
    print("- Run 'cargo clippy' after adding the safety lints")
    print("- Use ErrorRecovery::with_retry() for transient failures")
    print("- Use ErrorRecovery::with_fallback() for graceful degradation")
    
    if total_critical + total_high > 0:
        print(f"\n🚨 Found {total_critical + total_high} high-priority issues requiring immediate attention!")
        sys.exit(1)
    else:
        print(f"\n✅ No critical issues found. {sum(len(p) for p in categories.values())} lower-priority patterns remain.")
        sys.exit(0)


if __name__ == "__main__":
    main()