# Create WebSocket endpoint for real-time activation monitoring

## Status: PENDING

## Description
Build WebSocket endpoint enabling real-time monitoring of activation spreading and memory dynamics for debugging and visualization.

## Requirements
- WebSocket server at /ws/monitor
- Real-time activation level updates
- Memory formation event stream
- Consolidation progress updates
- Selective subscription to event types
- Binary and text frame support

## Acceptance Criteria
- [ ] WebSocket connects and maintains connection
- [ ] <100ms latency for activation updates
- [ ] Handles 100+ concurrent monitoring clients
- [ ] Graceful reconnection support
- [ ] Structured event format (JSON)

## Dependencies
- Task 017 (HTTP API)

## Notes
- Use tokio-tungstenite for WebSocket
- Consider binary frames for performance
- Implement event filtering server-side
- Support compression for large updates