use std::time::Duration;

use engram_core::decay::TwoComponentModel;

use super::statistical_analysis::{eta_squared, mean};

pub(crate) fn run_craik_lockhart_levels_of_processing() {
    let semantic = simulate_level(ProcessingLevel::Semantic);
    let phonemic = simulate_level(ProcessingLevel::Phonemic);
    let shallow = simulate_level(ProcessingLevel::Shallow);

    let semantic_mean = mean(&semantic);
    let phonemic_mean = mean(&phonemic);
    let shallow_mean = mean(&shallow);

    assert!(
        semantic_mean > phonemic_mean && phonemic_mean > shallow_mean,
        "Retention ordering should be Semantic > Phonemic > Shallow"
    );

    let depth_ratio = semantic_mean / shallow_mean.max(0.01);
    assert!(
        depth_ratio > 3.0,
        "Semantic/shallow ratio should exceed 3.0, observed {:.2}",
        depth_ratio
    );

    let effect_size = eta_squared(&[&semantic, &phonemic, &shallow]);
    assert!(
        effect_size > 0.3,
        "Levels-of-processing effect should be large (η² > 0.3), observed {:.2}",
        effect_size
    );
}

#[derive(Clone, Copy)]
enum ProcessingLevel {
    Shallow,
    Phonemic,
    Semantic,
}

fn simulate_level(level: ProcessingLevel) -> Vec<f32> {
    let mut retentions = Vec::new();
    let profiles = [0.8, 1.0, 1.1, 1.2];

    for profile in profiles {
        let mut model = TwoComponentModel::with_parameters(0.85, 1.0, 1.0, 2.0 * profile);
        match level {
            ProcessingLevel::Shallow => {
                model.update_on_retrieval(true, Duration::from_millis(350), 0.98);
            }
            ProcessingLevel::Phonemic => {
                model.update_on_retrieval(
                    true,
                    Duration::from_millis((650.0 * profile) as u64),
                    0.92,
                );
                model.update_on_retrieval(
                    true,
                    Duration::from_millis((720.0 * profile) as u64),
                    0.88,
                );
            }
            ProcessingLevel::Semantic => {
                model.update_on_retrieval(
                    true,
                    Duration::from_millis((900.0 * profile) as u64),
                    0.90,
                );
                model.update_on_retrieval(
                    true,
                    Duration::from_millis((1100.0 * profile) as u64),
                    0.82,
                );
                model.update_on_retrieval(
                    true,
                    Duration::from_millis((1250.0 * profile) as u64),
                    0.78,
                );
            }
        }

        let retention = (-24.0 / model.stability()).exp();
        retentions.push(retention);
    }

    retentions
}
