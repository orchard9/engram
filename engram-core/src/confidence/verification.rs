//! SMT-backed verification helpers for dual-memory confidence rules.

#[cfg(feature = "smt_verification")]
use z3::ast::Real;
#[cfg(feature = "smt_verification")]
use z3::{Context, SatResult, Solver};

#[cfg(feature = "smt_verification")]
fn real_from_f32(context: &Context, value: f32) -> Real<'_> {
    // Use three decimal digits of precision for deterministic rational encoding.
    let scaled = (value * 1000.0).round() as i32;
    Real::from_real(context, scaled, 1000)
}

/// Attempt to violate monotonic propagation by proving propagated > source.
#[cfg(feature = "smt_verification")]
#[must_use]
pub fn propagation_monotonicity_result(context: &Context, binding_decay: f32) -> SatResult {
    let solver = Solver::new(context);

    let source = Real::new_const(context, "source_conf");
    let binding = Real::new_const(context, "binding_strength");
    let zero = Real::from_real(context, 0, 1);
    let one = Real::from_real(context, 1, 1);

    solver.assert(&source.ge(&zero));
    solver.assert(&source.le(&one));
    solver.assert(&binding.ge(&zero));
    solver.assert(&binding.le(&one));

    let decay = real_from_f32(context, binding_decay);
    let propagated = Real::mul(context, &[&source, &binding, &decay]);
    solver.assert(&propagated.gt(&source));
    solver.check()
}

/// Attempt to violate blend bonus bounds by exceeding max(input) * bonus.
#[cfg(feature = "smt_verification")]
#[must_use]
pub fn blend_bonus_bounds_result(
    context: &Context,
    blend_bonus: f32,
    episodic_weight: f32,
    semantic_weight: f32,
) -> SatResult {
    let solver = Solver::new(context);

    let episodic = Real::new_const(context, "episodic");
    let semantic = Real::new_const(context, "semantic");
    let threshold = Real::from_real(context, 1, 2);
    let one = Real::from_real(context, 1, 1);

    solver.assert(&episodic.gt(&threshold));
    solver.assert(&semantic.gt(&threshold));
    solver.assert(&episodic.le(&one));
    solver.assert(&semantic.le(&one));

    let ep_weight = real_from_f32(context, episodic_weight);
    let sem_weight = real_from_f32(context, semantic_weight);
    let weighted_sum = Real::add(
        context,
        &[
            &Real::mul(context, &[&episodic, &ep_weight]),
            &Real::mul(context, &[&semantic, &sem_weight]),
        ],
    );
    let total_weight = Real::add(context, &[&ep_weight, &sem_weight]);
    let base = Real::div(&weighted_sum, &total_weight);

    let bonus = real_from_f32(context, blend_bonus);
    let with_bonus = Real::mul(context, &[&base, &bonus]);

    let cond = episodic.ge(&semantic);
    let max_input = cond.ite(&episodic, &semantic);
    let max_allowed = Real::mul(context, &[&max_input, &bonus]);

    solver.assert(&with_bonus.gt(&max_allowed));
    solver.check()
}

/// Attempt to violate the requirement that combined decay is < 1.0.
#[cfg(feature = "smt_verification")]
#[must_use]
pub fn cycle_convergence_result(
    context: &Context,
    binding_decay: f32,
    binding_strength: f32,
) -> SatResult {
    let solver = Solver::new(context);
    let decay = real_from_f32(context, binding_decay);
    let binding = real_from_f32(context, binding_strength);
    let combined = Real::mul(context, &[&decay, &binding]);
    let one = Real::from_real(context, 1, 1);

    solver.assert(&combined.ge(&one));
    solver.check()
}
