//! Progress bar and spinner utilities for long-running operations

#![allow(missing_docs)]

use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

pub struct OperationProgress {
    bar: ProgressBar,
    operation_name: String,
}

impl OperationProgress {
    #[must_use]
    pub fn new(operation: &str, total: u64) -> Self {
        let bar = ProgressBar::new(total);
        bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .expect("valid template")
                .progress_chars("=>-"),
        );
        bar.enable_steady_tick(Duration::from_millis(100));

        Self {
            bar,
            operation_name: operation.to_string(),
        }
    }

    pub fn set_message(&self, msg: &str) {
        self.bar
            .set_message(format!("{}: {}", self.operation_name, msg));
    }

    pub fn inc(&self, delta: u64) {
        self.bar.inc(delta);
    }

    pub fn finish(&self, msg: &str) {
        self.bar
            .finish_with_message(format!("{}: {}", self.operation_name, msg));
    }
}

#[must_use]
pub fn spinner(operation: &str) -> ProgressBar {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .expect("valid template"),
    );
    spinner.set_message(operation.to_string());
    spinner.enable_steady_tick(Duration::from_millis(80));
    spinner
}
