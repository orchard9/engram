use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get the path to the proto files
    let proto_root = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?)
        .parent()
        .unwrap()
        .join("proto");

    // Configure the protobuf compiler
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        // Generate descriptors for runtime reflection
        .file_descriptor_set_path(PathBuf::from(env::var("OUT_DIR")?).join("engram_descriptor.bin"))
        // Add attributes for better ergonomics - exclude Google types
        .type_attribute("engram", "#[derive(serde::Serialize, serde::Deserialize)]")
        .type_attribute("engram", "#[serde(rename_all = \"camelCase\")]")
        // Compile the proto files
        .compile(
            &[
                proto_root.join("engram/v1/memory.proto"),
                proto_root.join("engram/v1/service.proto"),
            ],
            &[proto_root],
        )?;

    Ok(())
}
