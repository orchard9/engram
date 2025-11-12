# Engram Reference Documentation

Complete technical reference for Engram's APIs, configuration, and performance characteristics.

## Performance and Benchmarking

- [Competitive Baselines](./competitive_baselines.md) - Comparison vs Qdrant, Neo4j, Milvus
- [Performance Baselines](./performance-baselines.md) - Engram single-node benchmarks
- [Benchmark Results](./benchmark-results.md) - Historical benchmark data
- [Resource Requirements](./resource-requirements.md) - Hardware sizing guide
- [System Requirements](./system-requirements.md) - Minimum and recommended specs

## API Documentation

- [REST API](./rest-api.md) - HTTP endpoints and examples
- [gRPC API](./grpc-api.md) - Protocol Buffers definitions
- [Query Language](./query-language.md) - RECALL/SPREAD/CONSOLIDATE syntax
- [Spreading Activation API](./spreading_api.md) - Advanced spreading parameters

## Configuration

- [Configuration Reference](./configuration.md) - All TOML settings documented
- [GPU Architecture](./gpu-architecture.md) - CUDA/ROCm configuration
- [Error Codes](./error-codes.md) - Complete error catalog
- [Error Catalog](./error-catalog.md) - Error handling guide

## Cognitive Patterns

- [Cognitive Patterns](./cognitive_patterns.md) - Psychological validation studies
- [API Versioning](./api-versioning.md) - Breaking change policy

## Advanced Topics

- [Streaming Performance Analysis](./streaming-performance-analysis.md) - Real-time streaming optimization
- [CLI Reference](./cli.md) - Command-line tool documentation

## Related Documentation

- **Tutorials**: [docs/tutorials/](../tutorials/) - Step-by-step learning guides
- **How-To Guides**: [docs/howto/](../howto/) - Problem-solving recipes
- **Explanation**: [docs/explanation/](../explanation/) - Understanding concepts
- **Operations**: [docs/operations/](../operations/) - Production deployment guides

## Contributing

Reference documentation should be comprehensive, accurate, and example-driven. When updating:

1. Follow [Diataxis framework](https://diataxis.fr/) - reference is information-oriented
2. Include working code examples for every API endpoint
3. Document all parameters, return values, and error conditions
4. Keep competitive baselines updated quarterly
5. Link to related tutorials/how-tos for practical usage

## Document Status

Last updated: 2025-11-11
Version: M17.1 (Competitive Baseline Framework)
