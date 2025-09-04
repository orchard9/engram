# Document HTTP API with OpenAPI specification

## Status: PENDING

## Description
Create complete OpenAPI 3.0 specification for the HTTP REST API, enabling client generation and interactive documentation.

## Requirements
- OpenAPI 3.0 specification file
- All endpoints documented
- Request/response schemas defined
- Error responses specified
- Authentication documented
- Interactive Swagger UI

## Acceptance Criteria
- [ ] Valid OpenAPI 3.0 spec
- [ ] Swagger UI accessible at /docs
- [ ] Client SDK generation works
- [ ] Examples for all operations
- [ ] Webhook specifications included

## Dependencies
- Task 017 (HTTP API)

## Notes
- Use utoipa for Rust integration
- Include rate limit headers
- Document Server-Sent Events endpoints
- Consider ReDoc for documentation