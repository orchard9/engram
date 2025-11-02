//! Interactive workflow helpers for CLI

#![allow(missing_docs)]

use anyhow::Result;
use std::io::{self, Write};

/// Prompt user for yes/no confirmation
pub fn confirm(message: &str, default_yes: bool) -> Result<bool> {
    let prompt = if default_yes {
        format!("{} [Y/n]: ", message)
    } else {
        format!("{} [y/N]: ", message)
    };

    print!("{}", prompt);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    let input = input.trim();
    if input.is_empty() {
        return Ok(default_yes);
    }

    Ok(input.eq_ignore_ascii_case("y") || input.eq_ignore_ascii_case("yes"))
}

/// Prompt user for text input
pub fn prompt(message: &str) -> Result<String> {
    print!("{}: ", message);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    Ok(input.trim().to_string())
}

/// Allow user to select from a list of options
pub fn select_from_list<T: std::fmt::Display>(message: &str, options: &[T]) -> Result<usize> {
    println!("{}", message);
    for (i, option) in options.iter().enumerate() {
        println!("  {}. {}", i + 1, option);
    }

    loop {
        print!("Select option [1-{}]: ", options.len());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if let Ok(choice) = input.trim().parse::<usize>()
            && choice > 0
            && choice <= options.len()
        {
            return Ok(choice - 1);
        }

        println!("Invalid selection. Please try again.");
    }
}

/// Context for executing operations with dry-run and verbose modes
pub struct OperationContext {
    pub dry_run: bool,
    pub verbose: bool,
    pub quiet: bool,
}

impl OperationContext {
    #[must_use]
    pub const fn new(dry_run: bool, verbose: bool, quiet: bool) -> Self {
        Self {
            dry_run,
            verbose,
            quiet,
        }
    }

    /// Execute an operation with context-aware logging
    pub fn execute<F, R>(&self, operation: &str, f: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
        R: Default,
    {
        if self.dry_run {
            if !self.quiet {
                println!("[DRY RUN] Would execute: {}", operation);
            }
            return Ok(R::default());
        }

        if self.verbose {
            println!("Executing: {}", operation);
        }

        f()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_context_dry_run() {
        let ctx = OperationContext::new(true, false, false);
        let result: Result<i32> = ctx.execute("test operation", || Ok(42));
        // In dry-run mode, should return default value (0 for i32)
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_operation_context_execute() {
        let ctx = OperationContext::new(false, false, false);
        let result: Result<i32> = ctx.execute("test operation", || Ok(42));
        assert_eq!(result.unwrap(), 42);
    }
}
