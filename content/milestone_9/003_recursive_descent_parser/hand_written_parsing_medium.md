# Why We Ditched Parser Generators and Wrote 500 Lines of Recursive Descent

"Just use nom," they said. "Parser generators are production-ready," they said.

Six months into building Engram's cognitive query language, we ripped out our carefully-crafted nom-based parser and replaced it with 500 lines of hand-written recursive descent.

Parse time dropped from 500ns to 80ns. Error messages went from cryptic to actually helpful. And debugging went from "stare at generated code" to "step through my own functions."

Here's why parser generators aren't always the answer - and when hand-written code wins.

## The Seduction of Parser Generators

Parser generators promise a beautiful world: write your grammar declaratively, get a working parser for free.

**nom (combinator library)**:
```rust
use nom::{
    bytes::complete::tag,
    sequence::tuple,
    IResult,
};

fn parse_recall(input: &str) -> IResult<&str, RecallQuery> {
    let (input, _) = tag("RECALL")(input)?;
    let (input, pattern) = parse_pattern(input)?;
    let (input, constraints) = opt(preceded(tag("WHERE"), parse_constraints))(input)?;
    Ok((input, RecallQuery { pattern, constraints }))
}
```

Looks elegant, right? Composable, zero-copy by default, fast.

**pest (PEG generator)**:
```pest
recall = { "RECALL" ~ pattern ~ ("WHERE" ~ constraints)? }
pattern = { identifier | embedding | string }
constraints = { constraint ~ ("AND" ~ constraint)* }
```

Even cleaner! Declarative grammar, generated parser.

So what's the problem?

## Problem 1: Error Messages That Make You Cry

Here's a real error from our nom-based parser:

```
Error: Parsing Error: Error { input: "RCALL episode WHERE confidence > 0.7", code: Tag }
```

What does this tell the user?
- Something went wrong. Somewhere.
- Good luck figuring out where.

Now here's the hand-written version:

```
Parse error at line 1, column 1:
  RCALL episode WHERE confidence > 0.7
  ^^^^^
  Unexpected identifier 'RCALL'
  Expected: RECALL, PREDICT, IMAGINE, CONSOLIDATE, or SPREAD
  Suggestion: Did you mean 'RECALL'? (edit distance: 1)
  Example: RECALL episode WHERE confidence > 0.7
```

The difference between a frustrated user and a productive one.

Parser generators can be configured for better errors, but it takes as much work as writing the parser yourself - without the control.

## Problem 2: Performance You Can't Optimize

nom is fast - for a combinator library. But you're at the mercy of its abstraction layers.

**nom-based parse time**: 500ns for simple query
**Hand-written parse time**: 80ns for simple query

What changed? Not the algorithm (both are recursive descent). The difference:

**Inlining control**:
```rust
#[inline]  // Force inlining on hot path
fn advance(&mut self) -> Option<Token> {
    // Called for every token - inlining critical
}
```

**Allocation control**:
```rust
let mut constraints = Vec::with_capacity(4);  // Pre-allocate typical size
```

**Branch prediction**:
```rust
match token {
    Token::Identifier(_) => { /* Most common - first branch */ }
    Token::Float(_) => { /* Second most */ }
    _ => { /* Rare cases last */ }
}
```

These micro-optimizations matter when your target is sub-100 microsecond query execution.

## Problem 3: Debugging Generated Code

You write this pest grammar:

```pest
pattern = { identifier | embedding | string }
```

pest generates ~200 lines of Rust code to handle this. When parsing fails mysteriously, you're reading generated code like:

```rust
fn parse_pattern(input: &str) -> Result<Node> {
    let state = ParserState::new(input);
    let result = match state.rule(Rule::pattern) {
        Ok(pair) => /* ... generated matching logic ... */
        Err(e) => return Err(transform_error(e, input)),
    };
    // ... 150 more lines ...
}
```

Want to add a print statement to debug? Good luck.

Hand-written version:

```rust
fn parse_pattern(&mut self) -> Result<Pattern> {
    match self.current {
        Token::Identifier(name) => {
            println!("DEBUG: parsing identifier {}", name);  // Easy!
            self.parse_node_id()
        }
        Token::LeftBracket => self.parse_embedding(),
        _ => Err(...)
    }
}
```

Step through with a debugger. Add prints. Understand exactly what's happening.

## The Architecture of a Hand-Written Parser

Here's the entire structure:

```rust
pub struct Parser<'a> {
    tokenizer: Tokenizer<'a>,  // Token stream
    current: Option<Spanned<Token<'a>>>,  // Current token
    previous: Option<Spanned<Token<'a>>>,  // For error recovery
}
```

112 bytes total. Fits in 2 cache lines. Always L1-resident.

**Core methods**:

```rust
impl<'a> Parser<'a> {
    // Entry point
    pub fn parse(source: &'a str) -> Result<Query, ParseError> {
        let mut parser = Self::new(source)?;
        parser.parse_query()
    }

    // Recursive descent functions (one per grammar rule)
    fn parse_query(&mut self) -> Result<Query> { ... }
    fn parse_recall(&mut self) -> Result<RecallQuery> { ... }
    fn parse_pattern(&mut self) -> Result<Pattern> { ... }
    fn parse_constraints(&mut self) -> Result<Vec<Constraint>> { ... }

    // Helper functions
    fn advance(&mut self) -> Result<Spanned<Token>> { ... }
    fn expect(&mut self, token: Token) -> Result<Spanned<Token>> { ... }
    fn check(&self, token: Token) -> bool { ... }
}
```

Each grammar rule becomes a function. Functions call each other recursively. That's it.

## Parsing as Pattern Recognition

Here's the beautiful part: recursive descent mirrors how your brain parses language.

When you hear "Recall the meeting," your brain doesn't use a lookup table. It recognizes hierarchical patterns:

**Level 1**: Phonemes → Sounds
**Level 2**: Morphemes → Word parts
**Level 3**: Words → "Recall", "the", "meeting"
**Level 4**: Phrases → Verb phrase
**Level 5**: Semantics → Memory retrieval operation

Each level feeds the next. Recursive descent does the same:

```rust
parse_query()           // Level 5: Understand full query
  ├─ parse_recall()     // Level 4: Recognize operation type
  │   ├─ parse_pattern()    // Level 3: Extract arguments
  │   └─ parse_constraints() // Level 3: Extract filters
  └─ ...
```

This isn't just elegant - it's debuggable. Want to understand how pattern parsing works? Read one function.

## Predictive Parsing: Making the Compiler Do the Work

The grammar is LL(1): one token of lookahead suffices to decide which rule applies.

```rust
fn parse_pattern(&mut self) -> Result<Pattern> {
    match self.current {
        Token::Identifier(_) => self.parse_node_id(),
        Token::LeftBracket => self.parse_embedding(),
        Token::StringLiteral(_) => self.parse_content_match(),
        _ => Err(ParseError::unexpected_token(...)),
    }
}
```

No backtracking. No trying multiple alternatives. Look at one token, know what to do.

This is O(1) decision-making. For a query with N tokens, we make N decisions. Total time: O(N).

Compare to backtracking PEG parsers:
- Try first alternative
- Fail
- Backtrack
- Try second alternative
- Succeed

Worst case: O(N × M) where M = number of alternatives.

## Error Recovery Through Pattern Completion

When you mistype "RCALL," your brain automatically suggests "RECALL." This is pattern completion: the partial cue activates similar patterns.

Our parser does the same:

```rust
fn suggest_keyword(typo: &str) -> Option<&'static str> {
    KEYWORDS.iter()
        .filter(|(k, _)| levenshtein(typo, k) <= 2)  // Edit distance ≤2
        .min_by_key(|(k, _)| levenshtein(typo, k))
        .map(|(k, _)| *k)
}
```

Levenshtein distance measures similarity in edit-space. "RCALL" is distance 1 from "RECALL" (one deletion).

For each parse error, we:
1. Find closest valid keyword (if within distance 2)
2. Include it in error message as suggestion
3. Show example of correct usage

Result: Errors become learning opportunities, not roadblocks.

## Parsing Embeddings: The Tight Loop

The most performance-critical part: parsing embedding vectors `[0.1, 0.2, 0.3, ...]`.

For a 768-dimensional embedding (typical for language models), we need to parse 768 floats fast.

```rust
fn parse_embedding(&mut self) -> Result<Vec<f32>> {
    self.expect(Token::LeftBracket)?;

    let mut values = Vec::with_capacity(768);  // Pre-allocate!

    while !self.check(Token::RightBracket) {
        values.push(self.parse_float()?);  // Inlined!

        if self.check(Token::Comma) {
            self.advance()?;
        }
    }

    self.expect(Token::RightBracket)?;
    Ok(values)
}

#[inline]  // Force inlining
fn parse_float(&mut self) -> Result<f32> {
    match self.current {
        Token::FloatLiteral(f) => {
            self.advance()?;
            Ok(f)
        }
        _ => Err(...)
    }
}
```

**Optimizations**:
1. Pre-allocate Vec with capacity 768 (no reallocation)
2. Inline `parse_float()` (no function call overhead)
3. Tight loop that the compiler can optimize aggressively

**Result**: Parse 768 floats in ~30μs (39ns per float).

## Measuring Success: Before and After

**Before (nom-based)**:
- Parse time: 500ns typical, 2μs worst-case
- Error messages: Cryptic ("Error: Tag")
- Debugging: Read generated code
- Compilation: 12 seconds (nom's proc macros)

**After (hand-written)**:
- Parse time: 80ns typical, 150ns worst-case
- Error messages: Contextual with suggestions
- Debugging: Step through own code
- Compilation: 4 seconds (no codegen)

6x faster parsing. Infinitely better errors. 3x faster compilation.

## When Parser Generators Still Win

Hand-written isn't always better. Parser generators win when:

**Complex grammars**: If you have 50+ production rules, hand-writing is tedious
**Ambiguous grammars**: LR/LALR parsers handle ambiguity better than LL
**Rapid prototyping**: Changing a pest grammar is faster than refactoring code
**Team expertise**: If your team knows ANTLR but not parsers, use ANTLR

We chose hand-written because:
- Small grammar (20 keywords, 10 production rules)
- Unambiguous syntax (LL(1) grammar)
- Performance critical (sub-100μs target)
- Error messages matter (cognitive interface)

## The Implementation: 500 Lines

Here's what those 500 lines include:

**parser.rs** (300 lines):
- Parser struct (20 lines)
- Helper methods (50 lines)
- Per-operation parsers (200 lines)
- Literal parsers (30 lines)

**error.rs** (100 lines):
- ParseError type (30 lines)
- Display implementation (40 lines)
- Suggestion logic (30 lines)

**tests.rs** (100 lines):
- Unit tests for each parsing function
- Integration tests for full queries
- Error message tests

That's it. No generated code. No build scripts. Just readable Rust.

## Lessons for Your Project

If you're building a parser, ask yourself:

**1. How complex is your grammar?**
- Simple (<20 rules): Consider hand-written
- Complex (>50 rules): Consider generator

**2. How important are error messages?**
- Critical (user-facing): Hand-written wins
- Nice to have: Generators are fine

**3. What's your performance budget?**
- Sub-microsecond: Hand-written probably needed
- Milliseconds: Generators are fast enough

**4. How often will the grammar change?**
- Rarely: Hand-written is fine
- Constantly: Generators ease refactoring

**5. Can your team debug it?**
- Comfortable with parsers: Hand-written
- Parser novices: Generators provide structure

For Engram, the answers were clear: simple grammar, critical errors, tight performance budget, stable grammar, expert team.

Hand-written recursive descent was the obvious choice.

## The Cognitive Parallel

Why does recursive descent feel natural?

Because it mirrors how your brain parses language:

**Hierarchical**: Recognition happens in layers (sounds → words → phrases → meaning)
**Predictive**: Context predicts what comes next (top-down processing)
**Recursive**: Complex structures are built from simpler ones
**Error-correcting**: Mistakes trigger pattern completion (suggestions)

When you design a parser for a cognitive memory system, the parser itself should work like cognition.

Recursive descent does. Parser generators... less so.

## Putting It All Together

Our final parser:
- 500 lines of readable Rust
- 80ns parse time for typical queries
- Helpful error messages with suggestions
- Zero external parser dependencies
- Easy to debug and maintain

Could we have gotten there with a parser generator? Maybe, with enough configuration and custom error handling.

But at that point, you're writing as much code as a hand-written parser - just in a less debuggable form.

We chose simplicity, performance, and control.

The result: a parser that's fast enough to be invisible, helpful enough to guide users, and maintainable enough to extend.

Just like how you don't notice your brain parsing sentences - you just understand them.

---

**About Engram**: Engram is an open-source cognitive memory system that brings biological memory principles to software. The query language parser is part of Milestone 9. Learn more at github.com/engram-memory.

**Technical details**: All benchmarks measured on Apple M1, compiled with `--release` and LTO. Parser comparison includes nom 7.1, pest 2.7, and lalrpop 0.20.
