//! Rich table formatting utilities for CLI output

#![allow(missing_docs)]

use std::io::Write;

pub struct TableBuilder {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    column_widths: Vec<usize>,
    use_color: bool,
}

impl TableBuilder {
    #[must_use]
    pub fn new(headers: Vec<String>) -> Self {
        let use_color = std::env::var("NO_COLOR").is_err();
        let column_widths = headers.iter().map(std::string::String::len).collect();

        Self {
            headers,
            rows: Vec::new(),
            column_widths,
            use_color,
        }
    }

    pub fn add_row(&mut self, row: Vec<String>) {
        // Update column widths based on content
        for (i, cell) in row.iter().enumerate() {
            if let Some(width) = self.column_widths.get_mut(i) {
                *width = (*width).max(cell.len());
            }
        }
        self.rows.push(row);
    }

    pub fn render<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        // Top border
        self.write_border(writer, '┌', '┬', '┐')?;

        // Headers
        self.write_row(writer, &self.headers, true)?;

        // Header separator
        self.write_border(writer, '├', '┼', '┤')?;

        // Data rows
        for row in &self.rows {
            self.write_row(writer, row, false)?;
        }

        // Bottom border
        self.write_border(writer, '└', '┴', '┘')?;

        Ok(())
    }

    fn write_border<W: Write>(
        &self,
        writer: &mut W,
        left: char,
        mid: char,
        right: char,
    ) -> std::io::Result<()> {
        write!(writer, "{}", left)?;
        for (i, width) in self.column_widths.iter().enumerate() {
            for _ in 0..(*width + 2) {
                write!(writer, "─")?;
            }
            if i < self.column_widths.len() - 1 {
                write!(writer, "{}", mid)?;
            }
        }
        writeln!(writer, "{}", right)
    }

    fn write_row<W: Write>(
        &self,
        writer: &mut W,
        cells: &[String],
        is_header: bool,
    ) -> std::io::Result<()> {
        write!(writer, "│")?;
        for (cell, width) in cells.iter().zip(&self.column_widths) {
            if is_header && self.use_color {
                // Bold cyan headers when color is enabled
                write!(
                    writer,
                    " \x1b[1;36m{:<width$}\x1b[0m │",
                    cell,
                    width = width
                )?;
            } else {
                write!(writer, " {:<width$} │", cell, width = width)?;
            }
        }
        writeln!(writer)
    }
}

/// Format byte count into human-readable string
#[must_use]
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_idx])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_builder_basic() {
        let mut table = TableBuilder::new(vec!["Name".to_string(), "Value".to_string()]);
        table.add_row(vec!["foo".to_string(), "42".to_string()]);
        table.add_row(vec!["bar".to_string(), "100".to_string()]);

        let mut output = Vec::new();
        table.render(&mut output).unwrap();

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Name"));
        assert!(rendered.contains("Value"));
        assert!(rendered.contains("foo"));
        assert!(rendered.contains("42"));
        assert!(rendered.contains("bar"));
        assert!(rendered.contains("100"));
    }

    #[test]
    fn test_table_builder_column_width() {
        let mut table = TableBuilder::new(vec!["Short".to_string(), "X".to_string()]);
        table.add_row(vec!["small".to_string(), "y".to_string()]);
        table.add_row(vec!["much_longer_value".to_string(), "z".to_string()]);

        let mut output = Vec::new();
        table.render(&mut output).unwrap();

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("much_longer_value"));
        // Verify column widths expanded properly
        let lines: Vec<&str> = rendered.lines().collect();
        let first_data_row_len = lines.get(3).map_or(0, |s| s.len());
        let second_data_row_len = lines.get(4).map_or(0, |s| s.len());
        assert_eq!(
            first_data_row_len, second_data_row_len,
            "All rows should have same length"
        );
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1_048_576), "1.00 MB");
        assert_eq!(format_bytes(1_073_741_824), "1.00 GB");
        assert_eq!(format_bytes(500), "500.00 B");
        assert_eq!(format_bytes(2_560), "2.50 KB");
    }
}
