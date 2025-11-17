//! Validation suite that ensures Engram's dual-memory configuration can
//! replicate canonical psychological experiments such as spacing effects
//! or semantic priming. These tests mirror published studies to guard
//! against regressions in behavior-sensitive logic.

#![cfg(feature = "dual_memory_types")]
#![allow(missing_docs)]

#[path = "psychological/category_formation_tests.rs"]
mod category_formation_tests;
#[path = "psychological/context_dependent_memory_tests.rs"]
mod context_dependent_memory_tests;
#[path = "psychological/fan_effect_replication.rs"]
mod fan_effect_replication;
#[path = "psychological/levels_of_processing_tests.rs"]
mod levels_of_processing_tests;
#[path = "psychological/semantic_priming_tests.rs"]
mod semantic_priming_tests;
#[path = "psychological/spacing_effect_tests.rs"]
mod spacing_effect_tests;
#[path = "psychological/statistical_analysis.rs"]
mod statistical_analysis;
#[path = "psychological/test_datasets.rs"]
mod test_datasets;

#[test]
fn test_anderson_1974_person_location_paradigm() {
    fan_effect_replication::run_anderson_1974_person_location_paradigm();
}

#[test]
fn test_rosch_1975_prototype_effects() {
    category_formation_tests::run_rosch_1975_prototype_effects();
}

#[test]
fn test_neely_1977_semantic_priming_soa() {
    semantic_priming_tests::run_neely_1977_semantic_priming_validation();
}

#[test]
#[ignore]
fn test_bjork_1992_spacing_effect() {
    spacing_effect_tests::run_bjork_1992_spacing_effect();
}

#[test]
#[ignore]
fn test_craik_lockhart_1972_levels_of_processing() {
    levels_of_processing_tests::run_craik_lockhart_levels_of_processing();
}

#[test]
#[ignore]
fn test_godden_baddeley_1975_context_dependent_memory() {
    context_dependent_memory_tests::run_godden_baddeley_context_effect();
}

#[test]
fn psychological_validation_fast_suite() {
    fan_effect_replication::run_anderson_1974_person_location_paradigm();
    category_formation_tests::run_rosch_1975_prototype_effects();
    semantic_priming_tests::run_neely_1977_semantic_priming_validation();
}

#[test]
#[ignore]
fn psychological_validation_full_suite() {
    spacing_effect_tests::run_bjork_1992_spacing_effect();
    levels_of_processing_tests::run_craik_lockhart_levels_of_processing();
    context_dependent_memory_tests::run_godden_baddeley_context_effect();
}
