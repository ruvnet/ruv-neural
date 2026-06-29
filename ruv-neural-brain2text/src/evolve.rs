//! Evolutionary hyperparameter optimizer — the in-crate "Darwin mode".
//!
//! Philosophy borrowed from ruvnet's `@metaharness/darwin` and the Darwin Gödel
//! Machine: **freeze the model, evolve the harness**. We keep the acoustic model
//! and LM fixed and *evolve the [`Brain2TextConfig`]* (bandpass, resample,
//! keystroke window, feature kind, LM weight, beam size). Fitness is
//! `1 - validation_CER`. Each generation tournament-selects parents, recombines
//! and mutates them into children, keeps elites, and records the best variant in
//! a growing archive — the genetic search the external Darwin tooling would
//! orchestrate, implemented natively so it runs with zero extra dependencies.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use ruv_neural_core::error::Result;

use crate::config::{Brain2TextConfig, FeatureKind};
use crate::dataset::Recording;
use crate::{evaluate, EvalSplit};

/// Settings for the evolutionary search.
#[derive(Debug, Clone)]
pub struct EvolveConfig {
    /// Population size per generation.
    pub population: usize,
    /// Number of generations.
    pub generations: usize,
    /// Number of elite variants carried over unchanged.
    pub elitism: usize,
    /// Per-field mutation probability.
    pub mutation_rate: f64,
    /// Fraction of sentences used for training during fitness evaluation.
    pub train_frac: f64,
    /// Fraction used for validation (fitness).
    pub val_frac: f64,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for EvolveConfig {
    fn default() -> Self {
        Self {
            population: 12,
            generations: 8,
            elitism: 2,
            mutation_rate: 0.3,
            train_frac: 0.7,
            val_frac: 0.15,
            seed: 0xB2A17,
        }
    }
}

/// A scored configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variant {
    /// The evolved configuration.
    pub config: Brain2TextConfig,
    /// Fitness (`1 - validation_CER`), higher is better.
    pub fitness: f64,
}

/// Output of an evolutionary run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolveResult {
    /// Best variant found across all generations.
    pub best: Variant,
    /// Best fitness per generation (the improvement curve).
    pub history: Vec<f64>,
    /// Archive of the best variant from each generation (DGM-style record).
    pub archive: Vec<Variant>,
}

/// Run the evolutionary optimizer against a recording.
pub fn evolve(recording: &Recording, ec: &EvolveConfig) -> Result<EvolveResult> {
    let mut rng = StdRng::seed_from_u64(ec.seed);

    // Seed the population: the V1 default plus random perturbations of it.
    let mut population: Vec<Brain2TextConfig> = Vec::with_capacity(ec.population);
    population.push(Brain2TextConfig::default());
    while population.len() < ec.population.max(1) {
        population.push(mutate(&Brain2TextConfig::default(), 1.0, &mut rng));
    }

    let mut history = Vec::with_capacity(ec.generations);
    let mut archive: Vec<Variant> = Vec::new();
    let mut best: Option<Variant> = None;

    for _gen in 0..ec.generations.max(1) {
        // Evaluate the whole population.
        let mut scored: Vec<Variant> = population
            .iter()
            .map(|cfg| {
                let fitness = evaluate(
                    recording,
                    cfg,
                    EvalSplit::Validation,
                    ec.train_frac,
                    ec.val_frac,
                )
                .map(|r| r.fitness())
                .unwrap_or(0.0);
                Variant {
                    config: cfg.clone(),
                    fitness,
                }
            })
            .collect();
        scored.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let gen_best = scored[0].clone();
        history.push(gen_best.fitness);
        archive.push(gen_best.clone());
        if best.as_ref().map(|b| gen_best.fitness > b.fitness).unwrap_or(true) {
            best = Some(gen_best);
        }

        // Build the next generation: elites + offspring.
        let mut next: Vec<Brain2TextConfig> = scored
            .iter()
            .take(ec.elitism.min(scored.len()))
            .map(|v| v.config.clone())
            .collect();
        while next.len() < ec.population.max(1) {
            let p1 = tournament(&scored, &mut rng);
            let p2 = tournament(&scored, &mut rng);
            let child = crossover(&p1.config, &p2.config, &mut rng);
            next.push(mutate(&child, ec.mutation_rate, &mut rng));
        }
        population = next;
    }

    Ok(EvolveResult {
        best: best.expect("at least one generation runs"),
        history,
        archive,
    })
}

/// Tournament selection: pick the fitter of two random variants.
fn tournament<'a>(scored: &'a [Variant], rng: &mut StdRng) -> &'a Variant {
    let i = rng.gen_range(0..scored.len());
    let j = rng.gen_range(0..scored.len());
    if scored[i].fitness >= scored[j].fitness {
        &scored[i]
    } else {
        &scored[j]
    }
}

/// Uniform per-field crossover.
fn crossover(a: &Brain2TextConfig, b: &Brain2TextConfig, rng: &mut StdRng) -> Brain2TextConfig {
    macro_rules! pick {
        ($f:ident) => {
            if rng.gen::<bool>() { a.$f } else { b.$f }
        };
    }
    Brain2TextConfig {
        bandpass_low_hz: pick!(bandpass_low_hz),
        bandpass_high_hz: pick!(bandpass_high_hz),
        filter_order: pick!(filter_order),
        resample_hz: pick!(resample_hz),
        epoch_pre_s: pick!(epoch_pre_s),
        epoch_post_s: pick!(epoch_post_s),
        feature: pick!(feature),
        ngram_order: pick!(ngram_order),
        lm_weight: pick!(lm_weight),
        beam_size: pick!(beam_size),
    }
    .clamp()
}

/// Mutate a config: each field is perturbed with probability `rate`.
fn mutate(base: &Brain2TextConfig, rate: f64, rng: &mut StdRng) -> Brain2TextConfig {
    let mut c = base.clone();
    let maybe = |rng: &mut StdRng| rng.gen::<f64>() < rate;

    if maybe(rng) {
        c.bandpass_low_hz *= jitter(rng, 0.5);
    }
    if maybe(rng) {
        c.bandpass_high_hz *= jitter(rng, 0.4);
    }
    if maybe(rng) {
        c.filter_order = (c.filter_order as i64 + rng.gen_range(-1..=1)).max(2) as usize;
    }
    if maybe(rng) {
        c.resample_hz *= jitter(rng, 0.3);
    }
    if maybe(rng) {
        c.epoch_pre_s *= jitter(rng, 0.4);
    }
    if maybe(rng) {
        c.epoch_post_s *= jitter(rng, 0.4);
    }
    if maybe(rng) {
        c.feature = match rng.gen_range(0..3) {
            0 => FeatureKind::Mean,
            1 => FeatureKind::Energy,
            _ => FeatureKind::MeanEnergy,
        };
    }
    if maybe(rng) {
        c.ngram_order = (c.ngram_order as i64 + rng.gen_range(-2..=2)).max(1) as usize;
    }
    if maybe(rng) {
        c.lm_weight = (c.lm_weight + rng.gen_range(-3.0..=3.0)).max(0.0);
    }
    if maybe(rng) {
        let factor = if rng.gen::<bool>() { 2 } else { 1 };
        c.beam_size = (c.beam_size as i64 * factor + rng.gen_range(-8..=8)).max(1) as usize;
    }
    c.clamp()
}

/// Multiplicative jitter in `[1-amt, 1+amt]`.
fn jitter(rng: &mut StdRng, amt: f64) -> f64 {
    1.0 + rng.gen_range(-amt..=amt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{generate_synthetic, SyntheticParams};

    fn corpus() -> Vec<&'static str> {
        vec![
            "hola mundo",
            "buenos dias amigo",
            "como estas hoy",
            "muy bien gracias",
            "hasta luego pronto",
            "que tengas buen dia",
            "nos vemos manana",
            "buenas noches a todos",
            "feliz cumpleanos hoy",
            "muchas gracias por todo",
            "hola que tal estas",
            "todo esta bien aqui",
            "vamos a la playa",
            "el sol brilla mucho",
        ]
    }

    #[test]
    fn mutate_and_crossover_stay_valid() {
        let mut rng = StdRng::seed_from_u64(1);
        let a = Brain2TextConfig::default();
        let b = mutate(&a, 1.0, &mut rng);
        let c = crossover(&a, &b, &mut rng);
        // Clamp invariants hold.
        assert!(c.bandpass_high_hz > c.bandpass_low_hz);
        assert!(c.beam_size >= 1);
        assert!(c.ngram_order >= 1);
        assert!(c.epoch_post_s > 0.0 && c.epoch_pre_s < 0.0);
    }

    #[test]
    fn evolution_does_not_regress() {
        let sents = corpus();
        let rec = generate_synthetic(&sents, &SyntheticParams::default(), 5);
        let ec = EvolveConfig {
            population: 8,
            generations: 5,
            ..Default::default()
        };
        let result = evolve(&rec, &ec).unwrap();

        // Best fitness is the max of the per-generation bests.
        let max_hist = result
            .history
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((result.best.fitness - max_hist).abs() < 1e-9);
        assert_eq!(result.archive.len(), 5);
        assert!(result.best.fitness >= 0.0 && result.best.fitness <= 1.0);

        // The evolved best should be at least as good as the V1 default baseline.
        let baseline = evaluate(
            &rec,
            &Brain2TextConfig::default(),
            EvalSplit::Validation,
            ec.train_frac,
            ec.val_frac,
        )
        .unwrap()
        .fitness();
        assert!(
            result.best.fitness >= baseline - 1e-9,
            "evolved {} < baseline {}",
            result.best.fitness,
            baseline
        );
    }
}
