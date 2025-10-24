# Error Messages That Teach: Why "Did You Mean RECALL?" Beats "Syntax Error"

Picture this: It's 3am, your production cognitive system is throwing errors, and you're staring at:

```
Parse error at position 12
```

Now picture this instead:

```
Parse error at line 1, column 1:
  Unknown keyword: 'RECAL'
  Did you mean: 'RECALL'?

Suggestion: Use RECALL for episodic memory retrieval
Example: RECALL episode WHERE confidence > 0.7
```

Which error message would you want at 3am?

This isn't just about being nice to developers. Error message quality directly impacts system adoption, debugging efficiency, and developer confidence. Let's explore what makes error messages actually helpful, and how we're building this into Engram's query parser.

---

## The Problem: Most Error Messages Are Terrible

Let's be honest: most parsers give terrible error messages. You've seen them:

```python
# Python
SyntaxError: invalid syntax

# Generic parser
Error: Unexpected token at line 3

# Cryptic compiler
error: expected `;`, found identifier
```

These messages fail the **"tiredness test"**: Would a developer at 3am, under pressure, sleep-deprived, understand what to do?

The answer is usually no. And that's a problem, because that's exactly when developers need error messages most.

---

## What Makes an Error Message Good?

After studying error messages across dozens of compilers, parsers, and developer tools, we've identified five essential qualities:

### 1. Precision: Tell Me Exactly Where

Bad:
```
Error: Parse error
```

Good:
```
Parse error at line 3, column 15:
  WHERE confidence >> 0.7
                   ^
```

Line and column numbers aren't optional. They're essential for quickly locating the problem, especially in multi-line queries.

### 2. Clarity: Explain What's Wrong

Bad:
```
Error: Unexpected token
```

Good:
```
Found: '>>' (not a valid operator)
Expected: >, <, >=, <=, or =
```

Tell me what you found AND what you expected. Don't make me guess what "unexpected" means in this context.

### 3. Actionability: Tell Me How to Fix It

Bad:
```
Syntax error in query
```

Good:
```
Suggestion: Use single comparison operators for constraints
Example: WHERE confidence > 0.7
```

Every error should include a suggestion and an example. Show me what correct syntax looks like.

### 4. Context: Understand Where I Am

Bad:
```
Expected: identifier
```

Good:
```
SPREAD requires FROM keyword followed by node identifier
Example: SPREAD FROM node_123 MAX_HOPS 5
```

The error message should reflect where I am in the grammar. What's valid after SPREAD is different from what's valid after RECALL.

### 5. Intelligence: Catch My Typos

Bad:
```
Unknown keyword: RECAL
```

Good:
```
Unknown keyword: 'RECAL'
Did you mean: 'RECALL'?
```

Typos are the most common error. Catching them with "did you mean" suggestions is table stakes for developer experience.

---

## The Secret Weapon: Levenshtein Distance

How do we catch typos automatically? Enter **Levenshtein distance** - a measure of how many single-character edits separate two strings.

```rust
"RECAL" → "RECALL" = 1 edit (insert 'L')
"SPRED" → "SPREAD" = 1 edit (insert 'A')
"IMAGIN" → "IMAGINE" = 1 edit (insert 'E')
"XYZ" → "RECALL" = 5 edits (too different, no suggestion)
```

The algorithm is simple: build a matrix of edit costs and find the minimum path. For keyword matching, it's perfect because:

1. Keywords are short (5-15 characters)
2. Keyword lists are small (10-20 keywords)
3. Error path can be slower than hot path (we only compute this on errors)

**Performance**: Computing Levenshtein distance for one keyword pair takes about 150 nanoseconds. Checking all 10 Engram keywords takes 1.5 microseconds. That's acceptable overhead for the error path.

**Threshold**: We use distance ≤2 as our cutoff. One or two typos? We'll suggest a fix. More than that? The user probably didn't intend that keyword, so we don't suggest.

---

## Context-Aware Error Messages

Generic error messages are easy to write. Context-aware error messages require architecture.

Here's how we track context in Engram's parser:

```rust
enum ParserContext {
    QueryStart,        // Beginning of query
    AfterRecall,       // After RECALL keyword
    InConstraints,     // Inside WHERE clause
    AfterSpread,       // After SPREAD keyword
}
```

Each context knows what's valid next:

**QueryStart**: Valid tokens are RECALL, PREDICT, IMAGINE, CONSOLIDATE, SPREAD
```
Query must start with a cognitive operation keyword
Example: RECALL episode WHERE confidence > 0.7
```

**AfterRecall**: Valid tokens are patterns (node IDs, embeddings, content matches)
```
RECALL requires a pattern (node ID, embedding, or content match)
Example: RECALL episode_123
```

**InConstraints**: Valid tokens are field names and operators
```
WHERE clause requires field name followed by operator and value
Example: WHERE confidence > 0.7
```

**AfterSpread**: Valid token is FROM keyword
```
SPREAD requires FROM keyword followed by node identifier
Example: SPREAD FROM node_123
```

This context flows through the parser, so when an error occurs, we know exactly what we were expecting and can give specific guidance.

---

## Real Examples: Bad vs Good

Let's compare actual error messages for common mistakes:

### Scenario 1: Typo in Keyword

Query:
```
RECAL episode
```

Bad error:
```
Parse error at position 0: unexpected identifier
```

Good error:
```
Parse error at line 1, column 1:
  Unknown keyword: 'RECAL'
  Did you mean: 'RECALL'?

Suggestion: Use RECALL for episodic memory retrieval
Example: RECALL episode WHERE confidence > 0.7
```

**Why it's better**: Immediately identifies the typo, suggests the fix, shows correct usage. Developer can fix it in seconds.

---

### Scenario 2: Missing Required Keyword

Query:
```
SPREAD node_123
```

Bad error:
```
Error: Expected FROM at position 7
```

Good error:
```
Parse error at line 1, column 8:
  Found: 'node_123'
  Expected: FROM keyword

Suggestion: SPREAD requires FROM keyword followed by node identifier
Example: SPREAD FROM node_123 MAX_HOPS 5
```

**Why it's better**: Explains the grammar rule, shows what's missing, provides complete example. Developer understands the pattern.

---

### Scenario 3: Invalid Operator

Query:
```
RECALL episode WHERE confidence >> 0.7
```

Bad error:
```
Error: Invalid operator
```

Good error:
```
Parse error at line 1, column 35:
  Found: '>>' (not a valid operator)
  Expected: >, <, >=, <=, or =

Suggestion: Use single comparison operators for constraints
Example: WHERE confidence > 0.7
```

**Why it's better**: Shows exactly what was wrong with the operator, lists all valid alternatives, demonstrates correct usage.

---

### Scenario 4: Confidence Out of Range

Query:
```
RECALL episode WHERE confidence > 1.5
```

Bad error:
```
Validation error: invalid confidence value
```

Good error:
```
Validation error at line 1, column 37:
  Confidence value: 1.5 (out of range)
  Valid range: 0.0 to 1.0

Suggestion: Confidence must be between 0.0 and 1.0
Example: WHERE confidence > 0.7
```

**Why it's better**: Shows the problematic value, explains the constraint, provides valid example. Developer knows exactly what to change.

---

## The Architecture of Good Error Messages

Good error messages don't happen by accident. They require architecture:

### Layer 1: Error Detection
```rust
impl Parser {
    fn expect_keyword(&mut self, keyword: &str) -> Result<(), ParseError> {
        let token = self.current_token();
        if !matches!(token, Token::Keyword(k) if k == keyword) {
            return Err(self.unexpected_token_error(token));
        }
        Ok(())
    }
}
```

The parser detects errors and captures context.

### Layer 2: Error Enrichment
```rust
impl Parser {
    fn unexpected_token_error(&self, found: &Token) -> ParseError {
        let (expected, suggestion, example) = match self.context {
            ParserContext::QueryStart => (
                vec!["RECALL", "PREDICT", "IMAGINE"],
                "Query must start with a cognitive operation keyword",
                "RECALL episode WHERE confidence > 0.7",
            ),
            // ... context-specific details
        };

        ParseError {
            kind: ErrorKind::UnexpectedToken { found, expected },
            position: self.position(),
            suggestion: suggestion.to_string(),
            example: example.to_string(),
        }
    }
}
```

Context determines what suggestions and examples to include.

### Layer 3: Error Presentation
```rust
impl Display for ParseError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "Parse error at line {}, column {}:",
                 self.position.line, self.position.column)?;

        match &self.kind {
            ErrorKind::UnknownKeyword { found, did_you_mean } => {
                writeln!(f, "  Unknown keyword: '{}'", found)?;
                if let Some(suggestion) = did_you_mean {
                    writeln!(f, "  Did you mean: '{}'?", suggestion)?;
                }
            }
            // ... format other error kinds
        }

        if !self.suggestion.is_empty() {
            writeln!(f, "\nSuggestion: {}", self.suggestion)?;
        }

        if !self.example.is_empty() {
            writeln!(f, "Example: {}", self.example)?;
        }

        Ok(())
    }
}
```

Structured formatting makes errors scannable and actionable.

---

## Testing Error Message Quality

How do we ensure every error message is good? Comprehensive testing:

### Test 1: Every Error Has Required Fields
```rust
#[test]
fn all_errors_have_actionable_messages() {
    for test_case in INVALID_QUERY_CORPUS {
        let error = Parser::parse(test_case.query).unwrap_err();

        // Verify structure
        assert!(error.position.line > 0);
        assert!(error.position.column > 0);
        assert!(!error.suggestion.is_empty());
        assert!(!error.example.is_empty());
    }
}
```

### Test 2: Typo Detection Works
```rust
#[test]
fn typo_detection_suggests_correct_keywords() {
    let test_cases = vec![
        ("RECAL", "RECALL"),
        ("SPRED", "SPREAD"),
        ("IMAGIN", "IMAGINE"),
    ];

    for (typo, expected) in test_cases {
        let query = format!("{} episode", typo);
        let error = Parser::parse(&query).unwrap_err();

        assert!(error.to_string().contains("Did you mean"));
        assert!(error.to_string().contains(expected));
    }
}
```

### Test 3: Position Tracking Is Accurate
```rust
#[test]
fn error_positions_are_accurate() {
    let query = "RECALL\n  episode\n  INVALID";
    let error = Parser::parse(query).unwrap_err();

    assert_eq!(error.position.line, 3);
    // Error should point to INVALID token
}
```

### Test 4: Context-Aware Expectations
```rust
#[test]
fn errors_reflect_parser_context() {
    let query = "SPREAD node_123"; // Missing FROM
    let error = Parser::parse(query).unwrap_err();

    assert!(error.to_string().contains("FROM"));
    assert!(error.to_string().contains("SPREAD requires FROM"));
}
```

We test 75+ invalid queries to ensure comprehensive error coverage.

---

## Performance: Error Path vs Hot Path

A common objection: "Won't all this error checking slow down parsing?"

No, because we distinguish **hot path** (valid queries) from **error path** (invalid queries):

**Hot Path** (99% of queries):
- Target: <100 microseconds
- Zero allocations (use string slices)
- No Levenshtein distance computation
- Fast token matching

**Error Path** (1% of queries):
- Target: <1 millisecond (10x slower is fine)
- Allocations allowed (error construction)
- Compute Levenshtein distance (~1.5μs for all keywords)
- Generate suggestions and examples

Developers encounter errors rarely. When they do, spending an extra millisecond to construct a helpful error message is worth it if it saves them 30 seconds of debugging.

---

## The Tiredness Test: A Design Principle

Let's revisit our opening scenario. It's 3am, production is down, you're tired and stressed. Which error message helps you?

```
Parse error at position 12
```

vs

```
Parse error at line 1, column 1:
  Unknown keyword: 'RECAL'
  Did you mean: 'RECALL'?

Suggestion: Use RECALL for episodic memory retrieval
Example: RECALL episode WHERE confidence > 0.7
```

The second one. Obviously.

That's the **tiredness test**: Would this error message make sense to a tired, stressed developer who just wants to fix the problem and get back to bed?

Every error message in Engram's parser must pass this test:
- No parser jargon ("unexpected token", "lookahead", "AST")
- Clear explanation of what's wrong
- Specific suggestion for how to fix it
- Example showing correct syntax
- Precise location (line, column)

---

## Lessons from Great Error Messages

Let's look at what other systems get right:

### Elm Compiler: Gold Standard
```
-- TYPE MISMATCH ---------------------------------------------------------------

The 1st argument to `drop` is not what I expect:

8|   List.drop (String.toInt userInput) [1,2,3,4,5,6]
                ^^^^^^^^^^^^^^^^^^^^^^
This `toInt` call produces:

    Maybe Int

But `drop` needs the 1st argument to be:

    Int

Hint: Use Maybe.withDefault to handle possible errors.
```

**What's great**:
- Clear section headers
- Visual highlighting of problem
- Explains actual vs expected
- Actionable hint with specific function

### Rust Compiler: Excellent
```
error[E0425]: cannot find value `x` in this scope
 --> src/main.rs:4:20
  |
4 |     println!("{}", x);
  |                    ^ help: a local variable with a similar name exists: `y`
```

**What's great**:
- Error code for documentation lookup
- Precise position with visual caret
- Helpful suggestion (did you mean `y`?)

### Python: Problematic
```
SyntaxError: invalid syntax
```

**What's missing**:
- No explanation of what syntax was invalid
- No suggestion for fix
- Position often imprecise

---

## Implementation in Engram

Here's how Engram's query parser implements these principles:

### Typo Detection
```rust
pub fn find_closest_keyword(input: &str, keywords: &[&str]) -> Option<String> {
    let input_lower = input.to_lowercase();
    let mut closest = None;
    let mut min_distance = usize::MAX;

    for keyword in keywords {
        let distance = levenshtein_distance(&input_lower, &keyword.to_lowercase());

        // Only suggest if distance ≤2 (1-2 typos)
        if distance <= 2 && distance < min_distance {
            min_distance = distance;
            closest = Some((*keyword).to_string());
        }
    }

    closest
}
```

### Context Tracking
```rust
impl Parser {
    fn get_suggestion_for_context(&self) -> String {
        match self.context {
            ParserContext::QueryStart =>
                "Query must start with a cognitive operation keyword",
            ParserContext::AfterRecall =>
                "RECALL requires a pattern (node ID, embedding, or content match)",
            ParserContext::InConstraints =>
                "WHERE clause requires field name followed by operator and value",
            ParserContext::AfterSpread =>
                "SPREAD requires FROM keyword followed by node identifier",
        }.to_string()
    }

    fn get_example_for_context(&self) -> String {
        match self.context {
            ParserContext::QueryStart =>
                "RECALL episode WHERE confidence > 0.7",
            ParserContext::AfterRecall =>
                "RECALL episode_123",
            ParserContext::InConstraints =>
                "WHERE confidence > 0.7",
            ParserContext::AfterSpread =>
                "SPREAD FROM node_123 MAX_HOPS 5",
        }.to_string()
    }
}
```

### Error Presentation
```rust
impl Display for ParseError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "Parse error at line {}, column {}:",
                 self.position.line, self.position.column)?;

        match &self.kind {
            ErrorKind::UnknownKeyword { found, did_you_mean } => {
                writeln!(f, "  Unknown keyword: '{}'", found)?;
                if let Some(suggestion) = did_you_mean {
                    writeln!(f, "  Did you mean: '{}'?", suggestion)?;
                }
            }
            ErrorKind::UnexpectedToken { found, expected } => {
                writeln!(f, "  Found: {}", found)?;
                writeln!(f, "  Expected: {}", expected.join(" or "))?;
            }
            ErrorKind::ValidationError { message } => {
                writeln!(f, "  Validation error: {}", message)?;
            }
        }

        if !self.suggestion.is_empty() {
            writeln!(f, "\nSuggestion: {}", self.suggestion)?;
        }

        if !self.example.is_empty() {
            writeln!(f, "Example: {}", self.example)?;
        }

        Ok(())
    }
}
```

---

## Measuring Success

How do we know if our error messages are good? Metrics:

**Quantitative**:
- 100% of errors have non-empty suggestion and example
- Typo detection covers all keywords (10/10 = 100%)
- Position tracking accurate within ±1 character
- Error construction time <1ms

**Qualitative**:
- All errors pass the "tiredness test"
- Examples show realistic usage (not toy queries)
- Suggestions are actionable (specific steps)
- No parser jargon in user-facing messages

**User-Facing**:
- >80% of errors fixed on first attempt
- <5% of support tickets about "unclear error messages"
- Positive developer sentiment on error quality

---

## Conclusion: Error Messages Are an Investment

Good error messages aren't free. They require:
- Architecture (error detection, enrichment, presentation)
- Context tracking (parser state management)
- Typo detection (Levenshtein distance implementation)
- Comprehensive testing (75+ test cases)
- Performance optimization (hot path vs error path)

But the payoff is enormous:
- **Faster debugging**: Fix errors in seconds, not minutes
- **Better learning**: Examples teach grammar through demonstration
- **Higher confidence**: Developers trust the system to catch mistakes
- **Lower support cost**: Fewer "how do I fix this?" questions

Error messages are not a secondary concern. They're a primary interface between your system and its users. Invest in them accordingly.

At 3am, when production is down and stress is high, your users will thank you for error messages that actually help.

---

## Further Reading

1. Elm Compiler Error Messages: https://elm-lang.org/news/compiler-errors-for-humans
2. Rust Error Handling: https://doc.rust-lang.org/book/ch09-00-error-handling.html
3. Levenshtein Distance: https://en.wikipedia.org/wiki/Levenshtein_distance
4. "Compiler Error Messages Considered Unhelpful" (Becker et al., 2019)
5. "Mind Your Language: On Novices' Interactions with Error Messages" (Marceau et al., 2011)
