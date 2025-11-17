# Engram cluster with docker-compose

This example brings up a three-node Engram cluster using the SWIM membership + static discovery configuration added in Milestone 14.

## Prerequisites
- Docker with BuildKit
- docker-compose v2+

## Quick start
```bash
cd deployments/docker/cluster

# Build the local image (first run takes ~2 minutes)
docker compose build

# Launch the three-node cluster
docker compose up -d

# Inspect status once the containers settle
docker compose ps
```

Exposed host ports:
- HTTP: 7432 (node1), 7433 (node2), 7434 (node3)
- gRPC: 50051 (node1), 50052 (node2), 50053 (node3)

Each node mounts a node-specific `engram.toml` under `node*/config/engram/config.toml`, sets `XDG_CONFIG_HOME=/config`, and binds its SWIM port to `0.0.0.0:7946`. The static seed list references the three service hostnames on the private `engram-net` bridge network.

### Advertise address
The CLI now refuses to gossip `0.0.0.0`, so every container must publish a routable address. The compose topology relies on the default auto-detection logic (connecting to each seed hostname to learn the bridge IP), so you do **not** need to set `[cluster.network].advertise_addr` manually. If you customize the network or remove static seeds, set `ENGRAM_CLUSTER_ADVERTISE_ADDR` for each node or provide an explicit `advertise_addr` in the config before starting the cluster.

## Health checks
Each container runs the built-in `engram status --json` healthcheck. You can also query it manually:
```bash
docker exec engram-node1 /usr/local/bin/engram status --json
docker exec engram-node2 /usr/local/bin/engram status --json
docker exec engram-node3 /usr/local/bin/engram status --json
```

HTTP health endpoints are exposed on the mapped ports (e.g., `http://localhost:7432/health`), but the CLI command is a simple way to verify readiness from inside the container namespace.

## Data paths
Per-node named volumes store data under the container path `/data`:
- `engram-node1-data`
- `engram-node2-data`
- `engram-node3-data`

Adjust capacities or feature flags in the per-node config files as needed.

## Monitoring & Dashboards

The cluster example reuses the production Grafana dashboards and Prometheus alerts from Milestone 16. Monitoring is optional and gated behind the `monitoring` profile so that the cluster can run on hosts without Grafana/Prometheus installed.

1. Tweak `.env` (in this folder) if you need a different cluster name, environment label, or retention window.
2. Regenerate the Prometheus config to pick up any `.env` changes:
   ```bash
   ./monitoring/render_prometheus_config.py
   ```
3. Start the monitoring profile alongside the cluster:
   ```bash
   docker compose --profile monitoring up -d
   ```
4. Access the tooling:
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (default admin/admin)

The Grafana container automatically loads the dashboards from `deployments/grafana/dashboards/` and provisioning under `deployments/grafana/provisioning/`. Prometheus ingests alerts and recording rules from `deployments/prometheus/*.yml` while using `monitoring/prometheus.generated.yml` for cluster-specific targets. Regenerate the config whenever you change `.env` values.

## Tear-down
```bash
docker compose down
```
