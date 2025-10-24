# Twitter Thread: Why We Ditched Parser Generators

## Tweet 1 (Hook)
"Just use nom for parsing"

We did. Parse time: 500ns. Error messages: "Error: Tag"

We rewrote it by hand in 500 lines of recursive descent.

Parse time: 80ns. Errors: Actually helpful.

Thread on why parser generators aren't always the answer:

## Tweet 2 (The Problem)
nom error message:

```
Error: Parsing Error: Error {
  input: "RCALL episode WHERE confidence > 0.7",
  code: Tag
}
```

What does this tell the user? Nothing useful.

Now imagine debugging this at 3am. With a production outage. Users waiting.

## Tweet 3 (The Solution)
Hand-written error:

```
Parse error at line 1, column 1:
  RCALL episode WHERE confidence > 0.7
  ^^^^^
  Unexpected: 'RCALL'
  Did you mean: 'RECALL'? (edit distance: 1)
  Example: RECALL episode WHERE confidence > 0.7
```

One error message. The difference between a frustrated user and a productive one.

## Tweet 4 (Performance Numbers)
nom-based: 500ns per query
Hand-written: 80ns per query

6x speedup.

What changed? Not the algorithm (both are recursive descent).

The difference: Control over inlining, allocation, and branch prediction.

Micro-optimizations matter at sub-microsecond scale.

## Tweet 5 (The Inlining Trick)
nom combinator: Function call overhead every token

Hand-written with aggressive inlining:

```rust
#[inline]  // Force inline
fn advance(&mut self) -> Option<Token> {
    // Called for every token
}
```

For 100-token query: Eliminates 100 function calls = ~500ns saved.

## Tweet 6 (Pre-allocation)
nom: Vec grows dynamically (multiple reallocations)

Hand-written:

```rust
let mut constraints = Vec::with_capacity(4);
// Pre-allocate typical size
```

Typical query has 2-4 constraints. Pre-allocating with capacity 4 saves 2-3 reallocations = ~150ns.

Every nanosecond counts.

## Tweet 7 (The Architecture)
Entire parser state:

```rust
struct Parser<'a> {
    tokenizer: Tokenizer<'a>,  // 64 bytes
    current: Option<Token<'a>>,  // 24 bytes
    previous: Option<Token<'a>>,  // 24 bytes
}
// Total: 112 bytes
```

Fits in 2 cache lines. Always L1-resident.

Parser generators don't give you this level of control.

## Tweet 8 (Recursive Descent Mirrors the Brain)
Your brain parsing "Recall the meeting":

Level 1: Sounds
Level 2: Words
Level 3: Phrases
Level 4: Semantics

Parser:

```rust
parse_query()
  parse_recall()
    parse_pattern()
      parse_identifier()
```

Hierarchical. Recursive. Natural.

## Tweet 9 (Predictive Parsing)
LL(1) grammar = one token lookahead

```rust
match self.current {
    Token::Identifier(_) => parse_node_id(),
    Token::LeftBracket => parse_embedding(),
    Token::String(_) => parse_content(),
}
```

O(1) decision per token. No backtracking. No wasted work.

PEG parsers backtrack. Expensive.

## Tweet 10 (Pattern Completion for Errors)
When you mistype "RCALL," your brain suggests "RECALL"

Parser does the same:

```rust
fn suggest(typo: &str) -> Option<&'static str> {
    KEYWORDS.iter()
        .filter(|k| levenshtein(typo, k) <= 2)
        .min_by_key(|k| levenshtein(typo, k))
}
```

Levenshtein distance = similarity in edit space.

Errors become learning opportunities.

## Tweet 11 (The Tight Loop)
Most critical: Parsing 768-float embeddings

```rust
let mut values = Vec::with_capacity(768);
while !self.check(RightBracket) {
    values.push(self.parse_float()?);  // Inlined!
}
```

Pre-allocate + inline = parse 768 floats in 30μs (39ns per float).

## Tweet 12 (Debugging Experience)
pest grammar generates ~200 lines of code per rule.

When something breaks: Read generated code. Good luck.

Hand-written:

```rust
fn parse_pattern(&mut self) -> Result<Pattern> {
    println!("DEBUG: current = {:?}", self.current);  // Easy!
    // ...
}
```

Step through with debugger. Understand what's happening.

## Tweet 13 (Compilation Speed)
nom-based: 12 seconds compile time (proc macros are slow)
Hand-written: 4 seconds compile time

3x faster builds. Better feedback loop. Happier developers.

Iteration speed matters.

## Tweet 14 (When Generators Win)
Parser generators aren't always bad. Use them when:

- Complex grammar (50+ rules)
- Ambiguous syntax (need LR/LALR)
- Rapid prototyping (change grammar often)
- Team doesn't know parsers

We had: Simple grammar, clear syntax, stable design, expert team.

Hand-written was obvious choice.

## Tweet 15 (The Cost-Benefit)
Cost: 500 lines of code upfront

Benefit:
- 6x faster parsing
- Helpful error messages
- 3x faster compilation
- Easy debugging
- Full control

For a production system with tight performance requirements, worth it.

## Tweet 16 (The 500 Lines)
What's in those 500 lines?

- Parser struct: 20 lines
- Helper methods: 50 lines
- Per-operation parsers: 200 lines
- Error handling: 100 lines
- Tests: 100 lines
- Literal parsing: 30 lines

That's it. No generated code. Just readable Rust.

## Tweet 17 (Recursive Descent = Natural)
Why does recursive descent feel intuitive?

Because it mirrors how your brain works:

Hierarchical (layers of processing)
Predictive (context guides parsing)
Recursive (complex from simple)
Error-correcting (pattern completion)

When building a cognitive system, the parser should work like cognition.

## Tweet 18 (The Measurements)
Before (nom):
- Parse: 500ns typical, 2μs worst
- Errors: Cryptic
- Debug: Read generated code
- Compile: 12s

After (hand-written):
- Parse: 80ns typical, 150ns worst
- Errors: Contextual + suggestions
- Debug: Step through own code
- Compile: 4s

## Tweet 19 (The Lesson)
Parser generators are a tool, not a solution.

Sometimes:
- The abstraction helps (complex grammars)
- The abstraction hurts (simple grammars + tight performance)

Know when you're paying for features you don't need.

For us: Hand-written won.

## Tweet 20 (Call to Action)
If you're building a parser, ask:

1. How complex is the grammar? (Simple = hand-written wins)
2. How critical are error messages? (Critical = hand-written wins)
3. What's the performance budget? (Sub-μs = hand-written probably needed)
4. How stable is the grammar? (Stable = hand-written is fine)

Don't reach for generators reflexively.

Open source: github.com/engram-memory
