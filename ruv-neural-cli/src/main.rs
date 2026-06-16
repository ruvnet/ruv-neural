//! rUv Neural CLI — Brain topology analysis, simulation, and visualization.

mod commands;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "ruv-neural")]
#[command(about = "rUv Neural — Brain Topology Analysis System")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbosity level
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

#[derive(Subcommand)]
enum Commands {
    /// Simulate neural sensor data
    Simulate {
        /// Number of channels
        #[arg(short, long, default_value = "64")]
        channels: usize,
        /// Duration in seconds
        #[arg(short, long, default_value = "10.0")]
        duration: f64,
        /// Sample rate in Hz
        #[arg(short, long, default_value = "1000.0")]
        sample_rate: f64,
        /// Output file (JSON)
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Analyze a brain connectivity graph
    Analyze {
        /// Input graph file (JSON)
        #[arg(short, long)]
        input: String,
        /// Show ASCII visualization
        #[arg(long)]
        ascii: bool,
        /// Export metrics to CSV
        #[arg(long)]
        csv: Option<String>,
    },
    /// Compute minimum cut on brain graph
    Mincut {
        /// Input graph file (JSON)
        #[arg(short, long)]
        input: String,
        /// Multi-way cut with k partitions
        #[arg(short, long)]
        k: Option<usize>,
    },
    /// Run full pipeline: simulate -> process -> analyze -> decode
    Pipeline {
        /// Number of channels
        #[arg(short, long, default_value = "32")]
        channels: usize,
        /// Duration in seconds
        #[arg(short, long, default_value = "5.0")]
        duration: f64,
        /// Show real-time ASCII dashboard
        #[arg(long)]
        dashboard: bool,
    },
    /// Export brain graph to visualization format
    Export {
        /// Input graph file (JSON)
        #[arg(short, long)]
        input: String,
        /// Output format: d3, dot, gexf, csv, rvf
        #[arg(short, long, default_value = "d3")]
        format: String,
        /// Output file
        #[arg(short, long)]
        output: String,
    },
    /// Run a closed-loop sensory neuromodulation session (Ruflo)
    Neuromod {
        /// Target state: relaxed, focused, or gamma
        #[arg(short, long, default_value = "relaxed")]
        target: String,
        /// Protocol: audio-haptic or multimodal
        #[arg(short, long, default_value = "audio-haptic")]
        protocol: String,
        /// Maximum control steps
        #[arg(short, long, default_value = "64")]
        steps: u64,
        /// Deterministic RNG seed
        #[arg(long, default_value = "7")]
        seed: u64,
        /// Inject an arousal-spike perturbation at this step (to demo safe-stop)
        #[arg(long)]
        perturb: Option<u64>,
        /// Photosensitivity screen cleared (enables the light channel)
        #[arg(long)]
        screened: bool,
        /// Write the session report (JSON) to this path
        #[arg(short, long)]
        output: Option<String>,
        /// Write the tamper-evident audit trail (JSON) to this path
        #[arg(long)]
        audit: Option<String>,
        /// Write a portable Ruflo evidence bundle (JSON) for the web console
        #[arg(long)]
        bundle: Option<String>,
        /// Ed25519-sign the audit-chain head and evidence bundle
        #[arg(long)]
        sign: bool,
    },
    /// Verify a Ruflo evidence bundle (reference verifier; matches the web UI)
    VerifyBundle {
        /// Path to a Ruflo evidence bundle (JSON)
        #[arg(short, long)]
        input: String,
    },
    /// Train a logistic-regression classifier and save it as a signed .rvf model
    Train {
        /// Input dataset (CSV or ARFF); last numeric column is the label
        #[arg(short, long)]
        input: String,
        /// Output path for the signed .rvf model
        #[arg(short, long)]
        output: String,
        /// Number of leading columns to skip (e.g. an id column)
        #[arg(long, default_value = "0")]
        skip_cols: usize,
        /// Label value mapped to the positive class (1)
        #[arg(long, default_value = "1")]
        positive: i64,
        /// Fraction of rows held out for the test report
        #[arg(long, default_value = "0.2")]
        test_frac: f64,
        /// Shuffle rows before the holdout split (else chronological)
        #[arg(long)]
        shuffle: bool,
        /// RNG seed for shuffling
        #[arg(long, default_value = "42")]
        seed: u64,
        /// Gradient-descent epochs
        #[arg(long, default_value = "400")]
        epochs: usize,
    },
    /// Inspect and verify a signed .rvf model
    ModelInfo {
        /// Path to a .rvf model
        #[arg(short, long)]
        input: String,
    },
    /// Score feature rows with a signed .rvf model
    Predict {
        /// Path to a .rvf model
        #[arg(short, long)]
        model: String,
        /// CSV of feature rows (no label column)
        #[arg(short, long)]
        input: String,
        /// Number of leading columns to skip
        #[arg(long, default_value = "0")]
        skip_cols: usize,
        /// Emit probabilities instead of 0/1 labels
        #[arg(long)]
        proba: bool,
    },
    /// Show system info and capabilities
    Info,
    /// Generate or verify Ed25519-signed capability witness bundles
    Witness {
        /// Output file path for generated witness bundle (JSON)
        #[arg(short, long)]
        output: Option<String>,
        /// Path to a witness bundle to verify
        #[arg(long)]
        verify: Option<String>,
    },
}

fn init_tracing(verbose: u8) {
    let level = match verbose {
        0 => tracing::Level::WARN,
        1 => tracing::Level::INFO,
        2 => tracing::Level::DEBUG,
        _ => tracing::Level::TRACE,
    };
    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(false)
        .init();
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    let result = match cli.command {
        Commands::Simulate {
            channels,
            duration,
            sample_rate,
            output,
        } => commands::simulate::run(channels, duration, sample_rate, output),
        Commands::Analyze { input, ascii, csv } => commands::analyze::run(&input, ascii, csv),
        Commands::Mincut { input, k } => commands::mincut::run(&input, k),
        Commands::Pipeline {
            channels,
            duration,
            dashboard,
        } => commands::pipeline::run(channels, duration, dashboard),
        Commands::Export {
            input,
            format,
            output,
        } => commands::export::run(&input, &format, &output),
        Commands::Neuromod {
            target,
            protocol,
            steps,
            seed,
            perturb,
            screened,
            output,
            audit,
            bundle,
            sign,
        } => commands::neuromod::run(
            &target,
            &protocol,
            steps,
            seed,
            perturb,
            screened,
            output.map(std::path::PathBuf::from),
            audit.map(std::path::PathBuf::from),
            bundle.map(std::path::PathBuf::from),
            sign,
        ),
        Commands::VerifyBundle { input } => {
            commands::verify_bundle::run(std::path::PathBuf::from(input))
        }
        Commands::Train {
            input,
            output,
            skip_cols,
            positive,
            test_frac,
            shuffle,
            seed,
            epochs,
        } => commands::model::train(
            &input, &output, skip_cols, positive, test_frac, shuffle, seed, epochs,
        ),
        Commands::ModelInfo { input } => commands::model::info(&input),
        Commands::Predict {
            model,
            input,
            skip_cols,
            proba,
        } => commands::model::predict(&model, &input, skip_cols, proba),
        Commands::Info => {
            commands::info::run();
            Ok(())
        }
        Commands::Witness { output, verify } => {
            commands::witness::run(
                output.map(std::path::PathBuf::from),
                verify.map(std::path::PathBuf::from),
            )
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn verify_cli() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parse_simulate_defaults() {
        let cli = Cli::try_parse_from(["ruv-neural", "simulate"]).unwrap();
        match cli.command {
            Commands::Simulate {
                channels,
                duration,
                sample_rate,
                output,
            } => {
                assert_eq!(channels, 64);
                assert!((duration - 10.0).abs() < 1e-9);
                assert!((sample_rate - 1000.0).abs() < 1e-9);
                assert!(output.is_none());
            }
            _ => panic!("Expected Simulate command"),
        }
    }

    #[test]
    fn parse_simulate_with_args() {
        let cli = Cli::try_parse_from([
            "ruv-neural",
            "simulate",
            "-c",
            "32",
            "-d",
            "5.0",
            "-s",
            "500.0",
            "-o",
            "out.json",
        ])
        .unwrap();
        match cli.command {
            Commands::Simulate {
                channels,
                duration,
                sample_rate,
                output,
            } => {
                assert_eq!(channels, 32);
                assert!((duration - 5.0).abs() < 1e-9);
                assert!((sample_rate - 500.0).abs() < 1e-9);
                assert_eq!(output.as_deref(), Some("out.json"));
            }
            _ => panic!("Expected Simulate command"),
        }
    }

    #[test]
    fn parse_analyze() {
        let cli =
            Cli::try_parse_from(["ruv-neural", "analyze", "-i", "graph.json", "--ascii"]).unwrap();
        match cli.command {
            Commands::Analyze { input, ascii, csv } => {
                assert_eq!(input, "graph.json");
                assert!(ascii);
                assert!(csv.is_none());
            }
            _ => panic!("Expected Analyze command"),
        }
    }

    #[test]
    fn parse_mincut() {
        let cli = Cli::try_parse_from(["ruv-neural", "mincut", "-i", "graph.json", "-k", "4"])
            .unwrap();
        match cli.command {
            Commands::Mincut { input, k } => {
                assert_eq!(input, "graph.json");
                assert_eq!(k, Some(4));
            }
            _ => panic!("Expected Mincut command"),
        }
    }

    #[test]
    fn parse_pipeline() {
        let cli = Cli::try_parse_from([
            "ruv-neural",
            "pipeline",
            "-c",
            "16",
            "-d",
            "3.0",
            "--dashboard",
        ])
        .unwrap();
        match cli.command {
            Commands::Pipeline {
                channels,
                duration,
                dashboard,
            } => {
                assert_eq!(channels, 16);
                assert!((duration - 3.0).abs() < 1e-9);
                assert!(dashboard);
            }
            _ => panic!("Expected Pipeline command"),
        }
    }

    #[test]
    fn parse_export() {
        let cli = Cli::try_parse_from([
            "ruv-neural",
            "export",
            "-i",
            "graph.json",
            "-f",
            "dot",
            "-o",
            "out.dot",
        ])
        .unwrap();
        match cli.command {
            Commands::Export {
                input,
                format,
                output,
            } => {
                assert_eq!(input, "graph.json");
                assert_eq!(format, "dot");
                assert_eq!(output, "out.dot");
            }
            _ => panic!("Expected Export command"),
        }
    }

    #[test]
    fn parse_info() {
        let cli = Cli::try_parse_from(["ruv-neural", "info"]).unwrap();
        assert!(matches!(cli.command, Commands::Info));
    }

    #[test]
    fn parse_verbose() {
        let cli = Cli::try_parse_from(["ruv-neural", "-vvv", "info"]).unwrap();
        assert_eq!(cli.verbose, 3);
    }

    #[test]
    fn parse_train() {
        let cli = Cli::try_parse_from([
            "ruv-neural", "train", "-i", "d.csv", "-o", "m.rvf", "--skip-cols", "1", "--shuffle",
        ])
        .unwrap();
        match cli.command {
            Commands::Train {
                input,
                output,
                skip_cols,
                positive,
                shuffle,
                ..
            } => {
                assert_eq!(input, "d.csv");
                assert_eq!(output, "m.rvf");
                assert_eq!(skip_cols, 1);
                assert_eq!(positive, 1);
                assert!(shuffle);
            }
            _ => panic!("Expected Train command"),
        }
    }

    #[test]
    fn parse_predict_and_model_info() {
        let cli =
            Cli::try_parse_from(["ruv-neural", "predict", "-m", "m.rvf", "-i", "f.csv", "--proba"])
                .unwrap();
        match cli.command {
            Commands::Predict { model, proba, .. } => {
                assert_eq!(model, "m.rvf");
                assert!(proba);
            }
            _ => panic!("Expected Predict command"),
        }

        let cli = Cli::try_parse_from(["ruv-neural", "model-info", "-i", "m.rvf"]).unwrap();
        assert!(matches!(cli.command, Commands::ModelInfo { .. }));
    }
}
