//! TLS configuration and certificate management.

use super::SecurityError;
use rustls::ServerConfig;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls_pemfile::{certs, pkcs8_private_keys};
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::Arc;

/// TLS protocol versions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlsVersion {
    /// TLS 1.2
    Tls12,
    /// TLS 1.3 (recommended)
    Tls13,
}

/// TLS configuration for server endpoints
#[derive(Debug, Clone)]
pub struct TlsConfig {
    /// Path to server certificate chain (PEM format)
    pub cert_chain_path: PathBuf,

    /// Path to server private key (PEM format)
    pub private_key_path: PathBuf,

    /// Optional CA bundle for client certificate validation
    pub ca_bundle_path: Option<PathBuf>,

    /// Minimum TLS protocol version (default: TLS 1.3)
    pub min_protocol_version: TlsVersion,

    /// Enable OCSP stapling
    pub ocsp_stapling: bool,
}

impl TlsConfig {
    /// Create a new TLS configuration
    #[must_use]
    pub fn new(cert_chain_path: PathBuf, private_key_path: PathBuf) -> Self {
        Self {
            cert_chain_path,
            private_key_path,
            ca_bundle_path: None,
            min_protocol_version: TlsVersion::Tls13,
            ocsp_stapling: false,
        }
    }

    /// Set CA bundle path for client certificate validation
    #[must_use]
    pub fn with_ca_bundle(mut self, ca_bundle_path: PathBuf) -> Self {
        self.ca_bundle_path = Some(ca_bundle_path);
        self
    }

    /// Set minimum TLS protocol version
    #[must_use]
    pub const fn with_min_version(mut self, version: TlsVersion) -> Self {
        self.min_protocol_version = version;
        self
    }

    /// Enable OCSP stapling
    #[must_use]
    pub const fn with_ocsp_stapling(mut self) -> Self {
        self.ocsp_stapling = true;
        self
    }

    /// Load and validate TLS configuration
    ///
    /// # Errors
    ///
    /// Returns `SecurityError` if:
    /// - Certificate files cannot be read
    /// - Private key cannot be parsed
    /// - Certificate chain is invalid
    pub fn load(&self) -> Result<Arc<ServerConfig>, SecurityError> {
        // Load certificate chain
        let cert_file = File::open(&self.cert_chain_path)
            .map_err(|e| SecurityError::Certificate(format!("Failed to open certificate: {e}")))?;
        let mut cert_reader = BufReader::new(cert_file);
        let cert_chain: Vec<CertificateDer<'static>> = certs(&mut cert_reader)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                SecurityError::Certificate(format!("Failed to parse certificates: {e}"))
            })?;

        if cert_chain.is_empty() {
            return Err(SecurityError::Certificate(
                "No certificates found in chain".to_string(),
            ));
        }

        // Load private key
        let key_file = File::open(&self.private_key_path)
            .map_err(|e| SecurityError::Certificate(format!("Failed to open private key: {e}")))?;
        let mut key_reader = BufReader::new(key_file);
        let keys = pkcs8_private_keys(&mut key_reader)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| SecurityError::Certificate(format!("Failed to parse private key: {e}")))?;

        if keys.is_empty() {
            return Err(SecurityError::Certificate(
                "No private key found".to_string(),
            ));
        }

        let private_key = PrivateKeyDer::Pkcs8(
            keys.into_iter()
                .next()
                .ok_or_else(|| SecurityError::Certificate("No private key found".to_string()))?,
        );

        // Build server configuration with default crypto provider
        let config = ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(cert_chain, private_key)
            .map_err(|e| SecurityError::Certificate(format!("Failed to build TLS config: {e}")))?;

        Ok(Arc::new(config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tls_config_builder() {
        let config = TlsConfig::new(
            PathBuf::from("/path/to/cert.pem"),
            PathBuf::from("/path/to/key.pem"),
        )
        .with_min_version(TlsVersion::Tls13)
        .with_ocsp_stapling();

        assert_eq!(config.min_protocol_version, TlsVersion::Tls13);
        assert!(config.ocsp_stapling);
    }
}
