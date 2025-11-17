use std::net::SocketAddr;
use std::sync::Arc;

use bincode::{deserialize, serialize};
use tokio::net::UdpSocket;

use super::{error::ClusterError, membership::SwimMessage};

const DEFAULT_MAX_PACKET_SIZE: usize = 1_400;

/// UDP-based transport for SWIM gossip traffic.
pub struct SwimTransport {
    socket: Arc<UdpSocket>,
    max_packet_size: usize,
}

impl SwimTransport {
    /// Bind a UDP socket for SWIM traffic.
    pub async fn bind(bind_addr: SocketAddr) -> Result<Self, ClusterError> {
        let socket = UdpSocket::bind(bind_addr).await?;
        Ok(Self {
            socket: Arc::new(socket),
            max_packet_size: DEFAULT_MAX_PACKET_SIZE,
        })
    }

    /// Returns the socket address assigned to the underlying UDP socket.
    pub fn local_addr(&self) -> Result<SocketAddr, ClusterError> {
        Ok(self.socket.local_addr()?)
    }

    /// Send a serialized SWIM message to the provided peer.
    pub async fn send(&self, message: &SwimMessage, addr: SocketAddr) -> Result<(), ClusterError> {
        let payload =
            serialize(message).map_err(|err| ClusterError::Serialization(err.to_string()))?;
        if payload.len() > self.max_packet_size {
            return Err(ClusterError::MessageTooLarge(payload.len()));
        }
        self.socket.send_to(&payload, addr).await?;
        Ok(())
    }

    /// Receive the next SWIM message from the UDP socket.
    pub async fn recv(&self) -> Result<(SwimMessage, SocketAddr), ClusterError> {
        let mut buf = vec![0u8; self.max_packet_size];
        let (len, addr) = self.socket.recv_from(&mut buf).await?;
        let message =
            deserialize(&buf[..len]).map_err(|err| ClusterError::Serialization(err.to_string()))?;
        Ok((message, addr))
    }
}
