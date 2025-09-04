# Probabilistic Types and Cognitive Architecture Research

## Research Areas

### 1. Human Intuitions About Uncertainty and Probability
**Research Focus**: How developers naturally think about uncertainty and confidence, and common cognitive biases in probabilistic reasoning

**Key Findings**:
- **Probability Matching vs Maximizing**: Herrnstein (1961) showed humans exhibit "probability matching" behavior—choosing options proportional to their success rate rather than always choosing the best option. This suggests confidence types should expose probability distributions rather than binary decisions.
- **Base Rate Neglect**: Kahneman & Tversky (1973) demonstrated that people systematically ignore base rates when making probabilistic judgments. APIs should make base rates explicit in confidence calculations.
- **Overconfidence Bias**: Fischhoff et al. (1977) research shows people consistently overestimate their certainty. Confidence types should include calibration mechanisms to counteract this bias.
- **Conjunction Fallacy**: Tversky & Kahneman (1983) proved people estimate conjunction probabilities higher than constituent probabilities. Type systems should prevent impossible confidence combinations at compile time.
- **Anchoring Effects**: Strack & Mussweiler (1997) show initial values strongly influence subsequent probability estimates. Confidence APIs should avoid providing misleading default values.

### 2. Cognitive Load and Probabilistic Reasoning
**Research Focus**: How uncertainty affects working memory and decision-making under cognitive load

**Key Findings**:
- **Dual-Process Theory**: Sloman (1996) research shows System 1 (automatic) thinking handles simple probabilities intuitively, while System 2 (controlled) thinking is required for complex probabilistic reasoning. Simple confidence operations should feel automatic.
- **Cognitive Load Effects**: Sweller (1988) cognitive load theory shows that probabilistic reasoning degrades more rapidly than deterministic reasoning under mental fatigue. Confidence types should minimize cognitive overhead through clear invariants.
- **Working Memory Constraints**: Research by Miller (1956) and modern studies by Cowan (2001) show working memory capacity limits affect probabilistic reasoning. Confidence intervals and ranges should fit within 3-7 chunk limitations.
- **Uncertainty Aversion**: Ellsberg (1961) paradox demonstrates people prefer known risks over unknown uncertainties. Type systems should distinguish between different sources of uncertainty (measurement error vs model uncertainty).

### 3. Type Theory and Probabilistic Programming
**Research Focus**: How type systems can encode probabilistic reasoning and uncertainty propagation

**Key Findings**:
- **Dependent Types for Ranges**: Research by Xi & Pfenning (1999) shows dependent types can encode numeric ranges at compile time. Confidence types can leverage this for [0,1] range enforcement.
- **Linear Type Systems**: Wadler (1990) linear logic research shows how resources can be tracked through computation. Confidence values have similar properties—they can't be duplicated without explicit combination rules.
- **Effect Systems**: Research by Gifford & Lucassen (1986) demonstrates how type systems can track computational effects. Probabilistic operations have effect-like properties that should be tracked by the type system.
- **Refinement Types**: Rondon et al. (2008) research on liquid types shows how refinement predicates can enforce complex invariants. Confidence types benefit from refinement predicates for validity checking.
- **Phantom Types**: Leijen & Meijer (1999) research shows phantom types can encode compile-time information without runtime cost. Confidence types can use phantoms to track uncertainty sources.

### 4. Bayesian Cognition and Mental Models
**Research Focus**: How humans naturally perform Bayesian reasoning and update beliefs with new evidence

**Key Findings**:
- **Natural Bayesian Reasoning**: Research by Gigerenzer & Hoffrage (1995) shows humans perform Bayesian reasoning naturally when information is presented as frequencies rather than probabilities. Confidence APIs should expose frequency-based interfaces.
- **Mental Model Theory**: Johnson-Laird (1983) research on mental models shows people reason about possibilities rather than probabilities. Confidence types should support scenario-based reasoning.
- **Belief Revision**: Research by Harman (1986) on belief revision shows humans use coherence rather than probabilistic updating. Confidence systems should support coherence checking.
- **Causal Reasoning**: Pearl (2000) research on causal inference shows humans naturally think causally rather than probabilistically. Confidence types should distinguish causal from correlational confidence.

### 5. Numerical Cognition and Floating-Point Psychology
**Research Focus**: How developers understand and work with floating-point numbers, particularly in probabilistic contexts

**Key Findings**:
- **Floating-Point Mental Models**: Research by Siegler & Opfer (2003) shows people have logarithmic intuitions about number magnitude. Confidence values near 0 and 1 are psychologically different from middle values.
- **Precision Illusions**: Studies by Klayman & Ha (1987) show people overestimate the precision of numerical calculations. Confidence types should expose precision limitations explicitly.
- **Decimal vs Binary Intuitions**: Research by Resnick et al. (1989) shows people think in decimal but computers calculate in binary. Confidence types should handle decimal-binary conversion transparently.
- **Comparison Difficulties**: Studies by Moyer & Landauer (1967) show numerical comparison time increases as numbers get closer. Confidence comparison operations should account for perceptual thresholds.

### 6. Error Handling in Probabilistic Systems
**Research Focus**: How developers handle and recover from errors in probabilistic computations

**Key Findings**:
- **Error Propagation Psychology**: Research by Norman (1981) on human error shows people have systematic blind spots in error propagation. Confidence types should make error propagation explicit and automatic.
- **Graceful Degradation**: Studies by Reason (1990) on human reliability show people prefer systems that degrade gracefully rather than failing catastrophically. Confidence systems should avoid Option<Confidence> in favor of degraded confidence values.
- **Error Detection**: Research by Card et al. (1983) shows visual error detection is more effective than numerical. Confidence types should support visual debugging and validation.
- **Recovery Strategies**: Studies by Lewis & Norman (1986) show people use systematic recovery strategies when errors are detected. Confidence APIs should provide clear recovery mechanisms.

### 7. Cognitive Biases in Software Development
**Research Focus**: How cognitive biases affect probabilistic reasoning in software systems

**Key Findings**:
- **Confirmation Bias**: Research by Wason (1960) shows people seek confirming rather than disconfirming evidence. Confidence systems should present contradictory evidence prominently.
- **Availability Heuristic**: Tversky & Kahneman (1974) research shows people overweight easily recalled examples. Confidence calibration should account for availability bias in training data.
- **Representativeness Heuristic**: Research shows people judge probability by similarity to mental prototypes. Confidence types should validate against actual base rates, not similarity judgments.
- **Hindsight Bias**: Fischhoff (1975) research shows people overestimate how predictable past events were. Confidence systems should maintain audit trails of historical confidence estimates.

### 8. Rust-Specific Type System Research
**Research Focus**: How Rust's ownership and type system can be leveraged for probabilistic programming

**Key Findings**:
- **Affine Types**: Research by O'Hearn (2003) on separation logic shows affine types can prevent resource leaks. Confidence values have similar properties—they shouldn't be accidentally duplicated.
- **Borrow Checker Applications**: Studies on Rust's borrow checker show it can enforce more than memory safety. Confidence types can use borrowing to enforce temporal consistency.
- **Zero-Cost Abstractions**: Research by Stroustrup (1994) and Rust's design principles show abstractions can have zero runtime cost. Confidence types should compile to raw f32 in release builds.
- **Trait System Leverage**: Research on Rust's trait system shows coherence rules prevent conflicting implementations. Confidence traits should use coherence to prevent invalid operations.

## Scientific Citations

1. Herrnstein, R. J. (1961). Relative and absolute strength of response as a function of frequency of reinforcement. Journal of the Experimental Analysis of Behavior, 4(3), 267-272.
2. Kahneman, D., & Tversky, A. (1973). On the psychology of prediction. Psychological Review, 80(4), 237-251.
3. Fischhoff, B., Slovic, P., & Lichtenstein, S. (1977). Knowing with certainty: The appropriateness of extreme confidence. Journal of Experimental Psychology: Human Perception and Performance, 3(4), 552-564.
4. Tversky, A., & Kahneman, D. (1983). Extensional versus intuitive reasoning: The conjunction fallacy in probability judgment. Psychological Review, 90(4), 293-315.
5. Strack, F., & Mussweiler, T. (1997). Explaining the enigmatic anchoring effect: Mechanisms of selective accessibility. Journal of Personality and Social Psychology, 73(3), 437-446.
6. Sloman, S. A. (1996). The empirical case for two systems of reasoning. Psychological Bulletin, 119(1), 3-22.
7. Sweller, J. (1988). Cognitive load during problem solving: Effects on learning. Cognitive Science, 12(2), 257-285.
8. Miller, G. A. (1956). The magical number seven, plus or minus two: Some limits on our capacity for processing information. Psychological Review, 63(2), 81-97.
9. Cowan, N. (2001). The magical number 4 in short-term memory: A reconsideration of mental storage capacity. Behavioral and Brain Sciences, 24(1), 87-114.
10. Ellsberg, D. (1961). Risk, ambiguity, and the Savage axioms. The Quarterly Journal of Economics, 75(4), 643-669.
11. Xi, H., & Pfenning, F. (1999). Dependent types in practical programming. ACM SIGPLAN Notices, 34(1), 214-227.
12. Wadler, P. (1990). Linear types can change the world. Programming Concepts and Methods, 2(3), 347-359.
13. Gifford, D. K., & Lucassen, J. M. (1986). Integrating functional and imperative programming. ACM Conference on LISP and Functional Programming, 28-38.
14. Rondon, P. M., Kawaguci, M., & Jhala, R. (2008). Liquid types. ACM SIGPLAN Notices, 43(6), 159-169.
15. Leijen, D., & Meijer, E. (1999). Domain specific embedded compilers. ACM SIGPLAN Notices, 34(12), 109-122.
16. Gigerenzer, G., & Hoffrage, U. (1995). How to improve Bayesian reasoning without instruction: frequency formats. Psychological Review, 102(4), 684-704.
17. Johnson-Laird, P. N. (1983). Mental Models: Towards a Cognitive Science of Language, Inference, and Consciousness. Harvard University Press.
18. Harman, G. (1986). Change in View: Principles of Reasoning. MIT Press.
19. Pearl, J. (2000). Causality: Models, Reasoning, and Inference. Cambridge University Press.
20. Siegler, R. S., & Opfer, J. E. (2003). The development of numerical estimation: Evidence for multiple representations of numerical quantity. Psychological Science, 14(3), 237-250.
21. Klayman, J., & Ha, Y. W. (1987). Confirmation, disconfirmation, and information in hypothesis testing. Psychological Review, 94(2), 211-228.
22. Resnick, L. B., Nesher, P., Leonard, F., Magone, M., Omanson, S., & Peled, I. (1989). Conceptual bases of arithmetic errors: The case of decimal fractions. Journal for Research in Mathematics Education, 20(1), 8-27.
23. Moyer, R. S., & Landauer, T. K. (1967). Time required for judgements of numerical inequality. Nature, 215(5109), 1519-1520.
24. Norman, D. A. (1981). Categorization of action slips. Psychological Review, 88(1), 1-15.
25. Reason, J. (1990). Human Error. Cambridge University Press.
26. Card, S. K., Moran, T. P., & Newell, A. (1983). The Psychology of Human-Computer Interaction. Lawrence Erlbaum Associates.
27. Lewis, C., & Norman, D. A. (1986). Designing for error. User Centered System Design, 411-432.
28. Wason, P. C. (1960). On the failure to eliminate hypotheses in a conceptual task. Quarterly Journal of Experimental Psychology, 12(3), 129-140.
29. Tversky, A., & Kahneman, D. (1974). Judgment under uncertainty: Heuristics and biases. Science, 185(4157), 1124-1131.
30. Fischhoff, B. (1975). Hindsight ≠ foresight: The effect of outcome knowledge on judgment under uncertainty. Journal of Experimental Psychology: Human Perception and Performance, 1(3), 288-299.
31. O'Hearn, P. (2003). Resources, concurrency, and local reasoning. Theoretical Computer Science, 375(1-3), 271-307.
32. Stroustrup, B. (1994). The Design and Evolution of C++. Addison-Wesley Professional.