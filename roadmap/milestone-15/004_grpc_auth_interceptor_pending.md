# Task: gRPC Authentication Interceptor

## Objective
Implement a Tonic interceptor that validates API keys for gRPC requests, providing the same authentication as the HTTP middleware.

## Context
The gRPC server in `engram-cli/src/grpc.rs` has no authentication. We need to add an interceptor that validates API keys from metadata headers.

## Requirements

### 1. Create gRPC Interceptor
Implement interceptor in `engram-cli/src/auth/grpc.rs`:
```rust
use tonic::{Request, Response, Status, service::Interceptor};
use engram_core::auth::{ApiKeyValidator, AuthContext};

pub struct AuthInterceptor {
    validator: Arc<ApiKeyValidator>,
    auth_mode: AuthMode,
}

impl Interceptor for AuthInterceptor {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        // Skip if auth disabled
        if matches!(self.auth_mode, AuthMode::None) {
            return Ok(request);
        }
        
        // Extract authorization from metadata
        let metadata = request.metadata();
        let auth_header = metadata
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| {
                Status::unauthenticated("Missing authorization header")
            })?;
        
        // Validate API key synchronously (use block_on carefully)
        let auth_context = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.validator.validate(auth_header).await
            })
        })
        .map_err(|e| match e {
            AuthError::InvalidApiKey => Status::unauthenticated("Invalid API key"),
            AuthError::ExpiredKey => Status::unauthenticated("API key expired"),
            _ => Status::internal(e.to_string()),
        })?;
        
        // Add auth context to request extensions
        request.extensions_mut().insert(auth_context);
        
        Ok(request)
    }
}
```

### 2. Async Interceptor Alternative
For better async handling:
```rust
pub async fn auth_interceptor(
    validator: Arc<ApiKeyValidator>,
    mut req: Request<()>,
) -> Result<Request<()>, Status> {
    let auth_header = req
        .metadata()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| Status::unauthenticated("Missing authorization"))?;
    
    let auth_context = validator
        .validate(auth_header)
        .await
        .map_err(|_| Status::unauthenticated("Invalid credentials"))?;
    
    req.extensions_mut().insert(auth_context);
    Ok(req)
}
```

### 3. Update MemoryService
Add auth context handling:
```rust
impl MemoryService {
    /// Extract auth context from request
    fn get_auth_context<T>(&self, request: &Request<T>) -> Result<&AuthContext, Status> {
        request
            .extensions()
            .get::<AuthContext>()
            .ok_or_else(|| Status::internal("Missing auth context"))
    }
    
    /// Check space access
    fn check_space_access(
        &self,
        auth: &AuthContext,
        space_id: &MemorySpaceId,
    ) -> Result<(), Status> {
        if !auth.allowed_spaces.contains(space_id) {
            return Err(Status::permission_denied(
                format!("Access denied to space: {}", space_id)
            ));
        }
        Ok(())
    }
}
```

### 4. Update Service Methods
Add auth checks to service methods:
```rust
#[tonic::async_trait]
impl proto::memory_service_server::MemoryService for MemoryService {
    async fn store(
        &self,
        request: Request<StoreRequest>,
    ) -> Result<Response<StoreResponse>, Status> {
        // Get auth context
        let auth = self.get_auth_context(&request)?;
        
        // Extract space from request
        let space_id = self.extract_space_id(&request)?;
        
        // Check access
        self.check_space_access(auth, &space_id)?;
        
        // Check write permission
        if !auth.permissions.contains(&Permission::MemoryWrite) {
            return Err(Status::permission_denied("Missing write permission"));
        }
        
        // Existing implementation...
    }
}
```

### 5. Apply Interceptor to Server
Update server creation in `main.rs`:
```rust
// Create interceptor if auth enabled
let interceptor = match config.security.auth_mode {
    AuthMode::None => None,
    AuthMode::ApiKey => {
        let validator = Arc::new(
            ApiKeyValidator::new(api_key_store)
        );
        Some(AuthInterceptor { validator, auth_mode: config.security.auth_mode })
    }
};

// Create service with optional interceptor
let service = if let Some(interceptor) = interceptor {
    MemoryServiceServer::with_interceptor(memory_service, interceptor)
} else {
    MemoryServiceServer::new(memory_service)
};

// Start server
Server::builder()
    .add_service(service)
    .serve(addr)
    .await?;
```

### 6. Client Authentication Support
Update client to send auth headers:
```rust
pub async fn get_grpc_client(
    addr: &str,
    api_key: Option<&str>,
) -> Result<MemoryServiceClient<Channel>> {
    let channel = Channel::from_shared(addr.to_string())?
        .connect()
        .await?;
    
    let mut client = MemoryServiceClient::new(channel);
    
    // Add auth interceptor if key provided
    if let Some(key) = api_key {
        let auth_header = format!("Bearer {}", key);
        client = client.with_interceptor(move |mut req: Request<()>| {
            req.metadata_mut().insert(
                "authorization",
                auth_header.parse().unwrap(),
            );
            Ok(req)
        });
    }
    
    Ok(client)
}
```

### 7. Metadata Propagation
Ensure auth metadata propagates through streaming calls:
```rust
async fn observe_stream(
    &self,
    request: Request<Streaming<ObserveRequest>>,
) -> Result<Response<ObserveResponse>, Status> {
    let auth = self.get_auth_context(&request)?;
    let mut stream = request.into_inner();
    
    // Validate each message in stream
    while let Some(msg) = stream.message().await? {
        let space_id = MemorySpaceId::try_from(msg.space.as_str())?;
        self.check_space_access(auth, &space_id)?;
        // Process message...
    }
}
```

## Testing Requirements
1. Test interceptor with valid/invalid keys
2. Test permission checking in service methods
3. Test streaming calls with auth
4. Test client with authentication
5. Test auth bypass when disabled
6. Load test with concurrent authenticated requests

## Error Handling
- Return appropriate gRPC status codes
- Include helpful error messages
- Don't leak sensitive information
- Log authentication failures

## Acceptance Criteria
1. gRPC calls require valid API key when enabled
2. Invalid keys return UNAUTHENTICATED status
3. Forbidden access returns PERMISSION_DENIED
4. Auth can be disabled via config
5. Streaming calls maintain auth context
6. Client can authenticate successfully
7. Same auth behavior as HTTP API