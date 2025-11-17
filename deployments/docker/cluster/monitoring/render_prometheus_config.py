#!/usr/bin/env python3
"""Render the cluster Prometheus configuration using .env overrides."""

from __future__ import annotations

import os
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
TEMPLATE_PATH = Path(__file__).resolve().parent / "prometheus.yml.tmpl"
OUTPUT_PATH = Path(__file__).resolve().parent / "prometheus.generated.yml"

DEFAULTS = {
    "CLUSTER_NAME": "demo-cluster",
    "PROMETHEUS_ENVIRONMENT": "local-dev",
    "PROMETHEUS_SCRAPE_INTERVAL": "15s",
    "PROMETHEUS_EVALUATION_INTERVAL": "15s",
    "PROMETHEUS_RETENTION": "7d",
}


def load_env() -> dict[str, str]:
    values = DEFAULTS.copy()
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def main() -> None:
    values = load_env()
    template = Template(TEMPLATE_PATH.read_text())
    rendered = template.substitute(values)
    OUTPUT_PATH.write_text(rendered)
    print(f"Rendered Prometheus config to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
