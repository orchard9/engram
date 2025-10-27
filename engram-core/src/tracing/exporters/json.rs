//! JSON exporter for cognitive events

use crate::tracing::event::{CognitiveEvent, EventType};
use serde_json::{Value, json};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Global output file for JSON export
static JSON_OUTPUT: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Set the output file for JSON export
#[allow(clippy::unwrap_used)] // Configuration API: panic on poisoned mutex is acceptable
pub fn set_output_file(path: PathBuf) {
    let mut output = JSON_OUTPUT.lock().unwrap();
    *output = Some(path);
}

/// Export events to JSON format
pub fn export_json(events: &[CognitiveEvent]) -> Result<(), Box<dyn std::error::Error>> {
    if events.is_empty() {
        return Ok(());
    }

    let json_events: Vec<Value> = events.iter().map(event_to_json).collect();

    let payload = json!({
        "events": json_events,
        "exported_at": SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs(),
        "count": events.len(),
    });

    // Get output file path
    let output_path = {
        let output = JSON_OUTPUT.lock().expect("JSON_OUTPUT mutex poisoned");
        output.clone()
    };

    if let Some(path) = output_path {
        // Write to file
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;

        writeln!(file, "{}", serde_json::to_string(&payload)?)?;
    } else {
        // Write to stdout if no file configured
        println!("{}", serde_json::to_string_pretty(&payload)?);
    }

    Ok(())
}

/// Convert a cognitive event to JSON
#[allow(unsafe_code)] // Required to access union fields in CognitiveEvent
fn event_to_json(event: &CognitiveEvent) -> Value {
    // Convert Instant timestamp to approximate SystemTime
    // Note: This is approximate since Instant is monotonic and doesn't
    // correspond to wall-clock time. For production, we'd need to track
    // the process start time as SystemTime.
    let timestamp_nanos = event.timestamp.elapsed().as_nanos();

    match event.event_type {
        EventType::Priming => {
            let data = unsafe { &event.data.priming };
            json!({
                "timestamp_ns": timestamp_nanos,
                "event_type": "Priming",
                "priming_type": format!("{:?}", data.priming_type),
                "strength": data.strength,
                "source_node": data.source_node,
                "target_node": data.target_node,
            })
        }
        EventType::Interference => {
            let data = unsafe { &event.data.interference };
            json!({
                "timestamp_ns": timestamp_nanos,
                "event_type": "Interference",
                "interference_type": format!("{:?}", data.interference_type),
                "magnitude": data.magnitude,
                "target_episode_id": data.target_episode_id,
                "competing_episode_count": data.competing_episode_count,
            })
        }
        EventType::Reconsolidation => {
            let data = unsafe { &event.data.reconsolidation };
            json!({
                "timestamp_ns": timestamp_nanos,
                "event_type": "Reconsolidation",
                "episode_id": data.episode_id,
                "window_position": data.window_position,
                "plasticity_factor": data.plasticity_factor,
                "modification_count": data.modification_count,
            })
        }
        EventType::FalseMemory => {
            let data = unsafe { &event.data.false_memory };
            json!({
                "timestamp_ns": timestamp_nanos,
                "event_type": "FalseMemory",
                "critical_lure_hash": format!("{:#x}", data.critical_lure_hash),
                "source_list_size": data.source_list_size,
                "reconstruction_confidence": data.reconstruction_confidence,
            })
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // Test code: panic on failure is acceptable
mod tests {
    use super::*;
    use crate::tracing::event::{InterferenceType, PrimingType};
    use tempfile::NamedTempFile;

    #[test]
    #[allow(clippy::float_cmp)] // Exact comparison intended for test validation
    fn test_event_to_json() {
        let event = CognitiveEvent::new_priming(PrimingType::Semantic, 0.75, 100, 200);
        let json = event_to_json(&event);

        assert_eq!(json["event_type"], "Priming");
        assert_eq!(json["priming_type"], "Semantic");
        assert_eq!(json["strength"], 0.75);
        assert_eq!(json["source_node"], 100);
        assert_eq!(json["target_node"], 200);
    }

    #[test]
    fn test_export_json_to_file() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();

        set_output_file(path.clone());

        let events = vec![
            CognitiveEvent::new_priming(PrimingType::Semantic, 0.5, 1, 2),
            CognitiveEvent::new_interference(InterferenceType::Proactive, 0.3, 999, 5),
        ];

        export_json(&events).unwrap();

        // Verify file was written
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("Priming"));
        assert!(contents.contains("Interference"));
        assert!(contents.contains("\"count\":2"));
    }

    #[test]
    fn test_export_empty_events() {
        let result = export_json(&[]);
        assert!(result.is_ok());
    }
}
