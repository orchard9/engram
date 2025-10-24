# Research: Hand-Written Recursive Descent Parsing

## Topic Overview

Recursive descent parsing is the most straightforward parsing technique: one function per grammar rule, calling each other recursively. While parser generators (nom, pest, lalrpop) are popular, hand-written parsers offer unmatched control over error messages, performance, and debuggability.

## Key Research Areas

### 1. Recursive Descent vs. Parser Generators

**Parser Generator Approaches**:

**nom (Parser combinators)**:
```rust
use nom::{
    bytes::complete::tag,
    sequence::tuple,
    IResult,
};

fn parse_recall(input: &str) -> IResult<&str, RecallQuery> {
    let (input, _) = tag("RECALL")(input)?;
    let (input, pattern) = parse_pattern(input)?;
    // ...
}
```

Pro: Composable, zero-copy by default
Con: Cryptic error messages, steep learning curve, slow compile times

**pest (PEG generator)**:
```pest
recall = { "RECALL" ~ pattern ~ ("WHERE" ~ constraints)? }
pattern = { identifier | embedding | string }
```

Pro: Declarative grammar, decent errors
Con: Generated code harder to debug, added build step

**lalrpop (LR parser generator)**:
```rust
grammar;
pub Query: Query = {
    "RECALL" <p:Pattern> => Query::Recall(p),
};
```

Pro: Powerful, handles ambiguity well
Con: Complex grammar syntax, slow compile times, shift-reduce conflicts

**Hand-written recursive descent**:
```rust
impl<'a> Parser<'a> {
    fn parse_recall(&mut self) -> ParseResult<RecallQuery> {
        self.expect(Token::Recall)?;
        let pattern = self.parse_pattern()?;
        // ...
    }
}
```

Pro: Easy to debug, fast compile, full error control
Con: More initial work, manual error recovery

**Performance comparison** (typical query parse time):
- Hand-written: 50-100ns
- nom: 200-500ns
- pest: 500-1000ns
- lalrpop: 300-700ns

**Reference**: "Parser Comparison" - Geal (nom author)
https://github.com/Geal/nom/blob/main/doc/compared_parsers.md

### 2. LL(k) Grammar Design

Recursive descent parsers work best with LL(k) grammars (Left-to-right, Leftmost derivation, k token lookahead).

**LL(1) grammar** (one token lookahead):
```
Query     → RECALL RecallBody | SPREAD SpreadBody | ...
RecallBody → Pattern WhereCl? ConfidenceCl?
Pattern    → Identifier | Embedding | String
```

Key property: At each decision point, one token of lookahead suffices to determine which rule to apply.

**Why LL(1) is ideal**:
- Minimal lookahead = faster parsing
- No backtracking = predictable performance
- Direct mapping to recursive functions

**Non-LL(1) example** (requires backtracking):
```
Pattern → Identifier Threshold?  // Could be node ID or start of "THRESHOLD 0.8"
```

Ambiguous: Is "confidence" an identifier or the start of "CONFIDENCE > 0.7"?

**Solution**: Use keywords (Token::Confidence) to disambiguate:
```
Identifier → [a-z]+ but not in {RECALL, SPREAD, CONFIDENCE, ...}
```

**Reference**: "Compilers: Principles, Techniques, and Tools" (Dragon Book)
Chapter 4: Syntax Analysis

### 3. Error Recovery Strategies

**Panic mode recovery**:
When error occurs, skip tokens until synchronization point (e.g., semicolon, keyword).

```rust
fn recover_from_error(&mut self) -> Result<(), ParseError> {
    while !self.is_at_sync_point() {
        self.advance()?;
    }
    Ok(())
}
```

Pro: Simple
Con: Skips potentially recoverable input

**Phrase-level recovery**:
Insert/delete/replace tokens to continue parsing:

```rust
fn parse_recall(&mut self) -> Result<RecallQuery> {
    if self.current != Token::Recall {
        // Try to recover: maybe they misspelled it
        if levenshtein(self.current, "RECALL") <= 2 {
            self.emit_warning("Did you mean RECALL?");
            self.advance();  // Skip and continue
        } else {
            return Err(ParseError::UnexpectedToken);
        }
    }
    // Continue parsing
}
```

Pro: Better error messages
Con: Complex, risk of cascading errors

**Error productions**:
Add grammar rules for common errors:

```rust
Pattern → Identifier
        | Embedding
        | WHERE  // Common error: forgot pattern before WHERE
          { error("Missing pattern before WHERE") }
```

Pro: Catches specific errors
Con: Bloats grammar

**Our approach**: Panic mode + phrase-level suggestions (no recovery)
- Emit best error message possible
- Don't try to continue parsing (avoid cascading errors)
- Use Levenshtein distance for keyword suggestions

**Reference**: "Error Recovery in Parsing Expression Grammars" - Maidl et al. (2016)

### 4. Operator Precedence Parsing

For expression parsing (e.g., `confidence > 0.7 AND created < "2024-01-01"`), precedence matters:

**Pratt parsing** (operator precedence):
```rust
fn parse_expr(&mut self, min_bp: u8) -> Result<Expr> {
    let mut lhs = self.parse_atom()?;

    loop {
        let op = match self.current {
            Token::And => (10, 11),  // (left_bp, right_bp)
            Token::Or => (5, 6),
            _ => break,
        };

        if op.0 < min_bp {
            break;
        }

        self.advance();
        let rhs = self.parse_expr(op.1)?;
        lhs = Expr::Binary(lhs, op, rhs);
    }

    Ok(lhs)
}
```

**Binding power** determines precedence:
- AND: bp=10 (higher priority)
- OR: bp=5 (lower priority)

Parse `a OR b AND c` as `a OR (b AND c)`, not `(a OR b) AND c`.

**Reference**: "Simple but Powerful Pratt Parsing" - matklad
https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html

**Our approach**: Constraints don't have operators (all implicit AND), so we use simple sequential parsing:

```rust
fn parse_constraints(&mut self) -> Result<Vec<Constraint>> {
    let mut constraints = Vec::new();
    while self.is_constraint_start() {
        constraints.push(self.parse_single_constraint()?);
    }
    Ok(constraints)
}
```

### 5. Left Recursion Elimination

Recursive descent can't handle left recursion:

```
Expr → Expr + Term  // Left recursive!
     | Term
```

This creates infinite loop: `parse_expr()` calls itself before consuming input.

**Solution**: Transform to right recursion or iteration:

```
Expr → Term ExprRest
ExprRest → + Term ExprRest
         | ε
```

Or use iteration:
```rust
fn parse_expr(&mut self) -> Result<Expr> {
    let mut expr = self.parse_term()?;
    while self.current == Token::Plus {
        self.advance();
        let term = self.parse_term()?;
        expr = Expr::Add(expr, term);
    }
    Ok(expr)
}
```

**Our grammar**: No left recursion by design
- All operations are prefix (RECALL, SPREAD, not "episode RECALL")
- Clauses are optional suffixes (WHERE, CONFIDENCE)

**Reference**: "Introduction to Compilers and Language Design" - Thain
Chapter 4: Parsing

### 6. Backtracking vs. Predictive Parsing

**Backtracking parsers** (PEG):
Try alternatives, backtrack on failure:

```rust
fn parse_pattern(&mut self) -> Result<Pattern> {
    if let Ok(id) = self.try_parse_node_id() {
        return Ok(Pattern::NodeId(id));
    }
    if let Ok(emb) = self.try_parse_embedding() {
        return Ok(Pattern::Embedding(emb));
    }
    Err(ParseError::NoPatternMatch)
}
```

Pro: Simple to implement
Con: Expensive (may re-parse same input multiple times)

**Predictive parsers** (LL):
Use lookahead to choose rule:

```rust
fn parse_pattern(&mut self) -> Result<Pattern> {
    match self.current {
        Token::Identifier(_) => self.parse_node_id(),
        Token::LeftBracket => self.parse_embedding(),
        Token::StringLiteral(_) => self.parse_content_match(),
        _ => Err(ParseError::UnexpectedToken),
    }
}
```

Pro: O(1) decision, no backtracking
Con: Requires unambiguous grammar

**Our approach**: Predictive with minimal backtracking
- Use token type to choose production
- Only backtrack for optional clauses (WHERE, CONFIDENCE)

### 7. Performance Optimization Techniques

**Inline hot paths**:
```rust
#[inline]
fn advance(&mut self) -> Result<Token> {
    // Called for every token - must be inlined
}

#[inline(never)]
fn report_error(&self, error: ParseError) {
    // Rarely called - save code size by not inlining
}
```

**Reduce allocations**:
```rust
// Bad: Allocates Vec for every parse
fn parse_constraints(&mut self) -> Vec<Constraint> {
    let mut constraints = Vec::new();  // Allocation!
    // ...
}

// Better: Pre-allocate with capacity
fn parse_constraints(&mut self) -> Vec<Constraint> {
    let mut constraints = Vec::with_capacity(4);  // Typical case
    // ...
}

// Best: Reuse buffer
struct Parser<'a> {
    tokenizer: Tokenizer<'a>,
    constraint_buffer: Vec<Constraint>,  // Reused across parses
}
```

**Branch prediction**:
Order match arms by frequency:

```rust
match token {
    Token::Identifier(_) => { /* Most common */ }
    Token::IntegerLiteral(_) => { /* Second most */ }
    Token::StringLiteral(_) => { /* Rare */ }
    _ => { /* Very rare */ }
}
```

**Loop unrolling** (for embedding parsing):
```rust
// Tight loop parsing [0.1, 0.2, 0.3, ...]
while !self.check(Token::RightBracket) {
    values.push(self.parse_float()?);
    if self.check(Token::Comma) {
        self.advance()?;
    }
}
```

Compiler unrolls for small iterations, improving cache locality.

**Reference**: Rust Performance Book - https://nnethercote.github.io/perf-book/

### 8. Error Message Quality

**Poor error**:
```
Parse error at line 3
```

**Better error**:
```
Parse error at line 3, column 12: unexpected token
```

**Good error**:
```
Parse error at line 3, column 12:
  RECALL episode WHERE confidnece > 0.7
                       ^^^^^^^^^
  Unexpected identifier 'confidnece'
  Did you mean: confidence
```

**Great error**:
```
Parse error at line 3, column 12:
  RECALL episode WHERE confidnece > 0.7
                       ^^^^^^^^^
  Unexpected identifier 'confidnece'
  Expected: constraint (confidence, created, content)
  Suggestion: Did you mean 'confidence'? (edit distance: 1)
  Example: WHERE confidence > 0.7
```

**Elements of great errors**:
1. Location (line, column)
2. Context (source snippet with caret)
3. What was found (actual token)
4. What was expected (valid tokens)
5. Suggestion (likely fix with edit distance)
6. Example (correct usage)

**Implementation**:
```rust
struct ParseError {
    position: Position,
    found: String,
    expected: Vec<String>,
    suggestion: Option<String>,
    example: String,
}

impl Display for ParseError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "Parse error at line {}, column {}:", self.position.line, self.position.column)?;
        writeln!(f, "  {}", self.source_context())?;
        writeln!(f, "  {}^", " ".repeat(self.position.column - 1))?;
        writeln!(f, "  Found: {}", self.found)?;
        writeln!(f, "  Expected: {}", self.expected.join(" or "))?;
        if let Some(suggestion) = &self.suggestion {
            writeln!(f, "  Suggestion: {}", suggestion)?;
        }
        writeln!(f, "  Example: {}", self.example)?;
        Ok(())
    }
}
```

**Reference**: "Crafting Interpreters" - Bob Nystrom
Chapter 6: Parsing Expressions (error handling)

### 9. Levenshtein Distance for Suggestions

**Algorithm**: Minimum edit distance between strings

```rust
fn levenshtein(a: &str, b: &str) -> usize {
    let m = a.len();
    let n = b.len();
    let mut dp = vec![vec![0; n + 1]; m + 1];

    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a.as_bytes()[i - 1] == b.as_bytes()[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)        // deletion
                .min(dp[i][j - 1] + 1)           // insertion
                .min(dp[i - 1][j - 1] + cost);   // substitution
        }
    }

    dp[m][n]
}
```

**Usage**:
```rust
fn suggest_keyword(&self, typo: &str) -> Option<&'static str> {
    KEYWORDS.iter()
        .filter(|(k, _)| levenshtein(typo, k) <= 2)  // Max distance 2
        .min_by_key(|(k, _)| levenshtein(typo, k))
        .map(|(k, _)| *k)
}
```

**Performance**: O(mn) where m, n are string lengths
For keywords (<20 chars), <100ns per comparison

**Reference**: "A Guided Tour of Dynamic Programming" - Wagner & Fischer (1974)

### 10. Testing Strategies

**Unit tests**: One function at a time
```rust
#[test]
fn test_parse_pattern_node_id() {
    let mut parser = Parser::new("episode_123");
    let pattern = parser.parse_pattern().unwrap();
    assert_eq!(pattern, Pattern::NodeId("episode_123".into()));
}
```

**Integration tests**: Full queries
```rust
#[test]
fn test_parse_recall_query() {
    let query = "RECALL episode WHERE confidence > 0.7";
    let ast = Parser::parse(query).unwrap();
    // Assert full AST structure
}
```

**Fuzzing**: Random input
```rust
#[test]
fn test_fuzz_parser() {
    for _ in 0..10000 {
        let input = generate_random_string();
        let result = Parser::parse(&input);
        // Should not crash, regardless of input
        let _ = result;
    }
}
```

**Property-based testing** (with proptest):
```rust
proptest! {
    #[test]
    fn parse_then_serialize_roundtrips(query: Query) {
        let serialized = query.to_string();
        let parsed = Parser::parse(&serialized)?;
        assert_eq!(query, parsed);
    }
}
```

**Snapshot testing** (for error messages):
```rust
#[test]
fn test_error_message_snapshot() {
    let input = "RCALL episode";  // Typo
    let error = Parser::parse(input).unwrap_err();
    insta::assert_snapshot!(error.to_string());
}
```

**Reference**: "Effective Testing" - Rust Testing Guide
https://doc.rust-lang.org/book/ch11-00-testing.html

## Performance Benchmarks from Literature

**Comparison with production parsers**:

| Parser Type | Parse Time (typical query) | Grammar Size |
|-------------|---------------------------|--------------|
| SQL (PostgreSQL) | 10-50μs | ~100 keywords, complex |
| GraphQL | 50-200μs | ~30 keywords, moderate |
| JSON (serde_json) | 5-20μs | Simple grammar |
| Protobuf | 1-5μs | Binary format, no parsing |
| **Engram (target)** | **<100μs** | ~20 keywords, moderate |

Target is achievable because:
- Simpler grammar than SQL (no JOINs, subqueries)
- Prefix operations (easier to predict)
- Smaller token vocabulary

## Open Questions

1. **Should we support query composition?**
   ```sql
   RECALL episode WHERE episode IN (SPREAD FROM node)
   ```
   Pro: Expressiveness
   Con: Complexity, harder to optimize

2. **Should constraints support arbitrary expressions?**
   ```sql
   WHERE confidence > 0.7 AND (created < "2024" OR updated > "2025")
   ```
   Pro: More powerful
   Con: Operator precedence, performance

3. **Should we generate a parse tree before AST?**
   Pro: Easier error recovery, preserves all syntax
   Con: Extra allocation, two passes

## References

### Classic Texts
1. "Compilers: Principles, Techniques, and Tools" (Dragon Book) - Aho et al.
2. "Engineering a Compiler" - Cooper & Torczon
3. "Parsing Techniques" - Grune & Jacobs

### Modern Approaches
1. "Crafting Interpreters" - Bob Nystrom: https://craftinginterpreters.com/
2. "Simple but Powerful Pratt Parsing" - matklad: https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html
3. "Recursive Descent Revisited" - Frost & Szydlowski (1996)

### Rust-Specific
1. Rust compiler parser: https://github.com/rust-lang/rust/tree/master/compiler/rustc_parse
2. rustc error messages guide: https://rustc-dev-guide.rust-lang.org/diagnostics.html
3. pest parser: https://pest.rs/

### Performance
1. Rust Performance Book: https://nnethercote.github.io/perf-book/
2. "Faster Parsing with Packrat Parsing" - Ford (2002)
3. "Parser Comparison" (nom): https://github.com/Geal/nom/blob/main/doc/compared_parsers.md
