use std::time::Duration;

use engram_core::decay::TwoComponentModel;

use super::statistical_analysis::{cohens_d, mean};

pub(crate) fn run_bjork_1992_spacing_effect() {
    let massed = simulate_recall(false);
    let spaced = simulate_recall(true);

    let massed_mean = mean(&massed);
    let spaced_mean = mean(&spaced);
    let improvement = (spaced_mean - massed_mean) / massed_mean;

    assert!(
        improvement > 0.25,
        "Spacing should improve retention by >25%, observed {:.1}%",
        improvement * 100.0
    );

    let effect_size = cohens_d(&spaced, &massed).abs();
    assert!(
        effect_size > 0.4,
        "Spacing effect size should exceed 0.4, observed {:.2}",
        effect_size
    );
}

fn simulate_recall(spaced: bool) -> Vec<f32> {
    let mut retentions = Vec::new();
    let profiles = if spaced {
        vec![1.0, 1.1, 1.2, 1.3]
    } else {
        vec![1.0, 1.02, 0.98, 1.05]
    };

    for profile in profiles {
        let mut model = TwoComponentModel::new();
        model.update_on_retrieval(true, Duration::from_millis(500), 0.95);

        if spaced {
            model.update_on_retrieval(true, Duration::from_millis((1200.0 * profile) as u64), 0.84);
            model.update_on_retrieval(true, Duration::from_millis((1100.0 * profile) as u64), 0.88);
        } else {
            model.update_on_retrieval(true, Duration::from_millis((400.0 * profile) as u64), 0.96);
            model.update_on_retrieval(true, Duration::from_millis((420.0 * profile) as u64), 0.95);
        }

        let retention = (-24.0 / model.stability()).exp();
        retentions.push(retention);
    }

    retentions
}
