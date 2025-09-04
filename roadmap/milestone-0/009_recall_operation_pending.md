# Create infallible recall() operation returning (Episode, Confidence) tuples

## Status: PENDING

## Description
Implement recall() operation that always returns results with confidence scores, never failing even when no exact matches exist.

## Requirements
- Recall returns Vec<(Episode, Confidence)>, never Result
- Empty results distinguished from low-confidence results
- Partial pattern matching with confidence degradation
- Spreading activation for associative recall
- Concurrent recalls without blocking
- Reconstruction of missing details with low confidence

## Acceptance Criteria
- [ ] recall() signature: fn recall(&self, cue: Cue) -> Vec<(Episode, Confidence)>
- [ ] Always returns, even if empty vector
- [ ] Confidence scores properly normalized [0,1]
- [ ] Supports multiple cue types (embedding, context, temporal)
- [ ] Non-blocking concurrent execution

## Dependencies
- Task 006 (Memory types)
- Task 008 (store operation)

## Notes
- Use activation spreading from vision.md
- Consider async for long-running recalls
- Empty vec != error condition
- Implement basic similarity first, spreading later