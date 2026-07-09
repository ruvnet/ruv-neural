//! Full end-to-end pipeline: acquire -> process -> analyze -> decode.

use std::f64::consts::PI;

use ruv_neural_core::brain::Atlas;
use ruv_neural_core::graph::{BrainEdge, BrainGraph, ConnectivityMetric};
use ruv_neural_core::signal::{FrequencyBand, MultiChannelTimeSeries};
use ruv_neural_core::topology::CognitiveState;
use ruv_neural_decoder::ThresholdDecoder;
use ruv_neural_embed::spectral_embed::SpectralEmbedder;
use ruv_neural_embed::topology_embed::TopologyEmbedder;
use ruv_neural_memory::RuvectorMemoryStore;
use ruv_neural_mincut::stoer_wagner_mincut;
use ruv_neural_sensor::thinkgear::{ThinkGearConfig, ThinkGearSource};
use ruv_neural_sensor::thinkgear_binary::{
    ThinkGearBinarySource, DEFAULT_BINARY_SAMPLE_RATE_HZ,
};
use ruv_neural_sensor::SensorSource;
use ruv_neural_signal::connectivity::phase_locking_value;
use ruv_neural_signal::filter::BandpassFilter;

/// Pipeline input source.
#[derive(Debug, Clone)]
pub enum PipelineSource {
    /// Generate deterministic synthetic data.
    Simulated,
    /// Read live raw EEG from NeuroSky ThinkGear Connector.
    Thinkgear {
        /// TGC host.
        host: String,
        /// TGC TCP port.
        port: u16,
        /// TGC app name.
        app_name: String,
        /// TGC app key.
        app_key: String,
    },
    /// Read raw ThinkGear binary packets from a serial/RFCOMM port.
    MindwaveBinary {
        /// Serial/RFCOMM port, for example COM5 or /dev/rfcomm0.
        port_name: String,
        /// Serial baud rate.
        baud_rate: u32,
    },
}

/// Full pipeline command options.
#[derive(Debug, Clone)]
pub struct PipelineOptions {
    /// Input source.
    pub source: PipelineSource,
    /// Requested simulated channels, or virtual montage width for ThinkGear.
    pub channels: usize,
    /// Duration in seconds.
    pub duration: f64,
    /// Show ASCII dashboard.
    pub dashboard: bool,
    /// Index generated embeddings with the RuVector memory bridge.
    pub ruvector_index: bool,
}

/// Run the full pipeline command.
pub fn run(options: PipelineOptions) -> Result<(), Box<dyn std::error::Error>> {
    if !options.duration.is_finite() || options.duration <= 0.0 {
        return Err("duration must be finite and positive".into());
    }

    println!("=== rUv Neural - Full Pipeline ===");
    println!();

    println!("  [1/7] Acquiring sensor data...");
    let mut ts = acquire_timeseries(&options)?;
    if ts.num_channels < 2 {
        let target_channels = options.channels.clamp(2, 8);
        ts = expand_virtual_montage(&ts, target_channels)
            .map_err(|e| format!("Virtual montage failed: {e}"))?;
        println!(
            "        Expanded single-channel EEG into {target_channels} lag-derived virtual channels"
        );
    }
    let channels = ts.num_channels;
    let sample_rate = ts.sample_rate_hz;
    let num_samples = ts.num_samples;
    let duration = options.duration;
    let raw_data = ts.data.clone();
    println!("        {channels} channels, {num_samples} samples, {duration:.1}s");

    println!("  [2/7] Preprocessing (bandpass 1-100 Hz)...");
    let filter = BandpassFilter::new(4, 1.0, 100.0, sample_rate);
    let filtered: Vec<Vec<f64>> = raw_data
        .iter()
        .map(|ch| {
            use ruv_neural_signal::filter::SignalProcessor;
            filter.process(ch)
        })
        .collect();
    println!("        Bandpass filter applied to all channels");

    println!("  [3/7] Constructing brain connectivity graph (PLV)...");
    let graph = build_plv_graph(&filtered, sample_rate);
    println!(
        "        {} nodes, {} edges, density {:.4}",
        graph.num_nodes,
        graph.edges.len(),
        graph.density()
    );

    println!("  [4/7] Computing minimum cut and topology metrics...");
    let mc = stoer_wagner_mincut(&graph).map_err(|e| format!("Mincut failed: {e}"))?;
    println!(
        "        Cut value: {:.4}, balance: {:.4}",
        mc.cut_value,
        mc.balance_ratio()
    );
    println!(
        "        Partition A: {} nodes, Partition B: {} nodes",
        mc.partition_a.len(),
        mc.partition_b.len()
    );

    println!("  [5/7] Generating topology embedding...");
    let embedder = TopologyEmbedder::new();
    let embedding = embedder
        .embed_graph(&graph)
        .map_err(|e| format!("Embedding failed: {e}"))?;
    println!(
        "        Dimension: {}, norm: {:.4}",
        embedding.dimension,
        embedding.norm()
    );

    let spectral_dim = channels.min(8).max(2);
    let spectral = SpectralEmbedder::new(spectral_dim);
    let spectral_emb = spectral
        .embed_graph(&graph)
        .map_err(|e| format!("Spectral embedding failed: {e}"))?;
    println!(
        "        Spectral embedding: dim={}, norm={:.4}",
        spectral_emb.dimension,
        spectral_emb.norm()
    );

    println!("  [6/7] Decoding cognitive state...");
    let decoder = build_default_decoder();
    let metrics = ruv_neural_core::topology::TopologyMetrics {
        global_mincut: mc.cut_value,
        modularity: estimate_modularity(&graph),
        global_efficiency: estimate_efficiency(&graph),
        local_efficiency: 0.0,
        graph_entropy: estimate_entropy(&graph),
        fiedler_value: 0.0,
        num_modules: 2,
        timestamp: graph.timestamp,
    };
    let (state, confidence) = decoder.decode(&metrics);
    println!("        State:      {state:?}");
    println!("        Confidence: {confidence:.4}");

    if options.ruvector_index {
        let store = RuvectorMemoryStore::new(embedding.dimension, 1024)
            .map_err(|e| format!("RuVector memory init failed: {e}"))?;
        let id = store
            .insert(&embedding)
            .map_err(|e| format!("RuVector memory insert failed: {e}"))?;
        let hits = store
            .search_ids(&embedding, 1)
            .map_err(|e| format!("RuVector memory search failed: {e}"))?;
        println!(
            "        RuVector index: inserted {id}, nearest hits={}",
            hits.len()
        );
    }

    println!("  [7/7] Results summary");
    println!();
    println!("  Pipeline Results Summary");
    println!("  ------------------------");
    println!("  Channels:        {channels}");
    println!("  Duration:        {duration:.1} s");
    println!("  Graph density:   {:.4}", graph.density());
    println!("  Mincut value:    {:.4}", mc.cut_value);
    println!("  Balance ratio:   {:.4}", mc.balance_ratio());
    println!("  Modularity:      {:.4}", metrics.modularity);
    println!("  Graph entropy:   {:.4}", metrics.graph_entropy);
    println!("  Embedding dim:   {}", embedding.dimension);
    println!("  Cognitive state: {state:?}");
    println!("  Confidence:      {confidence:.4}");
    println!();

    if options.dashboard {
        print_dashboard(&ts, &graph, &mc, &metrics);
    }

    Ok(())
}

fn acquire_timeseries(
    options: &PipelineOptions,
) -> Result<MultiChannelTimeSeries, Box<dyn std::error::Error>> {
    match &options.source {
        PipelineSource::Simulated => {
            let sample_rate = 1000.0;
            let num_samples = (options.duration * sample_rate) as usize;
            let raw_data = generate_data(options.channels, num_samples, sample_rate);
            MultiChannelTimeSeries::new(raw_data, sample_rate, 0.0)
                .map_err(|e| format!("Time series creation failed: {e}").into())
        }
        PipelineSource::Thinkgear {
            host,
            port,
            app_name,
            app_key,
        } => {
            let config = ThinkGearConfig {
                host: host.clone(),
                port: *port,
                app_name: app_name.clone(),
                app_key: app_key.clone(),
                ..ThinkGearConfig::default()
            };
            let mut source = ThinkGearSource::connect(config)
                .map_err(|e| format!("ThinkGear connection failed: {e}"))?;
            let num_samples = (options.duration * source.sample_rate_hz()) as usize;
            source
                .read_chunk(num_samples.max(1))
                .map_err(|e| format!("ThinkGear acquisition failed: {e}").into())
        }
        PipelineSource::MindwaveBinary {
            port_name,
            baud_rate,
        } => {
            let mut source = ThinkGearBinarySource::open_serial(
                port_name,
                *baud_rate,
                DEFAULT_BINARY_SAMPLE_RATE_HZ,
            )
            .map_err(|e| format!("MindWave binary connection failed: {e}"))?;
            let num_samples = (options.duration * source.sample_rate_hz()) as usize;
            source
                .read_chunk(num_samples.max(1))
                .map_err(|e| format!("MindWave binary acquisition failed: {e}").into())
        }
    }
}

fn expand_virtual_montage(
    ts: &MultiChannelTimeSeries,
    target_channels: usize,
) -> ruv_neural_core::Result<MultiChannelTimeSeries> {
    let source = ts
        .data
        .first()
        .ok_or_else(|| ruv_neural_core::RuvNeuralError::Signal("missing EEG channel".into()))?;
    let len = source.len();
    let mut expanded = Vec::with_capacity(target_channels);
    for ch in 0..target_channels {
        let lag = ch * 2;
        let scale = 1.0 - ch as f64 * 0.05;
        let channel = (0..len)
            .map(|idx| {
                let delayed = source[idx.saturating_sub(lag)];
                let previous = source[idx.saturating_sub(lag + 1)];
                scale * delayed - (1.0 - scale) * previous
            })
            .collect();
        expanded.push(channel);
    }
    MultiChannelTimeSeries::new(expanded, ts.sample_rate_hz, ts.timestamp_start)
}

/// Generate synthetic multi-channel neural data.
fn generate_data(channels: usize, num_samples: usize, sample_rate: f64) -> Vec<Vec<f64>> {
    let mut data = Vec::with_capacity(channels);
    for ch in 0..channels {
        let mut channel_data = Vec::with_capacity(num_samples);
        let phase = (ch as f64) * PI / (channels as f64);
        let mut rng: u64 = (ch as u64)
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);

        for i in 0..num_samples {
            let t = i as f64 / sample_rate;
            let alpha = 50.0 * (2.0 * PI * 10.0 * t + phase).sin();
            let beta = 30.0 * (2.0 * PI * 20.0 * t + phase * 1.3).sin();
            let gamma = 15.0 * (2.0 * PI * 40.0 * t + phase * 0.7).sin();

            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u1 = (rng >> 11) as f64 / (1u64 << 53) as f64;
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = (rng >> 11) as f64 / (1u64 << 53) as f64;
            let noise = if u1 > 1e-15 {
                5.0 * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
            } else {
                0.0
            };

            channel_data.push(alpha + beta + gamma + noise);
        }
        data.push(channel_data);
    }
    data
}

/// Build a brain graph from PLV connectivity between all channel pairs.
fn build_plv_graph(channels: &[Vec<f64>], sample_rate: f64) -> BrainGraph {
    let n = channels.len();
    let mut edges = Vec::new();
    let plv_threshold = 0.3;

    for i in 0..n {
        for j in (i + 1)..n {
            let plv =
                phase_locking_value(&channels[i], &channels[j], sample_rate, FrequencyBand::Alpha);
            if plv > plv_threshold {
                edges.push(BrainEdge {
                    source: i,
                    target: j,
                    weight: plv,
                    metric: ConnectivityMetric::PhaseLockingValue,
                    frequency_band: FrequencyBand::Alpha,
                });
            }
        }
    }

    BrainGraph {
        num_nodes: n,
        edges,
        timestamp: 0.0,
        window_duration_s: 1.0,
        atlas: Atlas::Custom(n),
    }
}

fn estimate_modularity(graph: &BrainGraph) -> f64 {
    let n = graph.num_nodes;
    if n < 2 {
        return 0.0;
    }
    let total = graph.total_weight();
    if total < 1e-12 {
        return 0.0;
    }

    let adj = graph.adjacency_matrix();
    let degrees: Vec<f64> = (0..n).map(|i| graph.node_degree(i)).collect();
    let two_m = 2.0 * total;
    let mid = n / 2;
    let mut q = 0.0;
    for i in 0..n {
        for j in 0..n {
            let same_community = (i < mid && j < mid) || (i >= mid && j >= mid);
            if same_community {
                q += adj[i][j] - degrees[i] * degrees[j] / two_m;
            }
        }
    }
    q / two_m
}

fn estimate_efficiency(graph: &BrainGraph) -> f64 {
    let n = graph.num_nodes;
    if n < 2 {
        return 0.0;
    }
    let adj = graph.adjacency_matrix();
    let mut sum = 0.0;
    let mut count = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if adj[i][j] > 0.0 {
                sum += adj[i][j];
            }
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    sum / count as f64
}

fn estimate_entropy(graph: &BrainGraph) -> f64 {
    let total = graph.total_weight();
    if total < 1e-12 || graph.edges.is_empty() {
        return 0.0;
    }
    let mut entropy = 0.0;
    for edge in &graph.edges {
        let p = edge.weight / total;
        if p > 1e-15 {
            entropy -= p * p.ln();
        }
    }
    entropy
}

fn build_default_decoder() -> ThresholdDecoder {
    let mut decoder = ThresholdDecoder::new();

    decoder.set_threshold(
        CognitiveState::Rest,
        ruv_neural_decoder::TopologyThreshold {
            mincut_range: (0.0, 5.0),
            modularity_range: (0.2, 0.6),
            efficiency_range: (0.1, 0.4),
            entropy_range: (1.0, 3.0),
        },
    );
    decoder.set_threshold(
        CognitiveState::Focused,
        ruv_neural_decoder::TopologyThreshold {
            mincut_range: (3.0, 15.0),
            modularity_range: (0.4, 0.8),
            efficiency_range: (0.3, 0.7),
            entropy_range: (2.0, 4.0),
        },
    );
    decoder.set_threshold(
        CognitiveState::MotorPlanning,
        ruv_neural_decoder::TopologyThreshold {
            mincut_range: (2.0, 10.0),
            modularity_range: (0.3, 0.7),
            efficiency_range: (0.2, 0.6),
            entropy_range: (1.5, 3.5),
        },
    );

    decoder
}

fn print_dashboard(
    ts: &MultiChannelTimeSeries,
    graph: &BrainGraph,
    mc: &ruv_neural_core::topology::MincutResult,
    metrics: &ruv_neural_core::topology::TopologyMetrics,
) {
    println!("  rUv Neural - Live Dashboard");
    println!("  ---------------------------");

    let display_channels = ts.num_channels.min(6);
    let display_samples = ts.num_samples.min(50);
    let sparkline_chars = ['_', '.', '-', '~', '=', '+', '*', '#'];

    for ch in 0..display_channels {
        let data = &ts.data[ch];
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;
        let step = (ts.num_samples / display_samples).max(1);
        let mut sparkline = String::new();
        for i in 0..display_samples {
            let val = data[i * step];
            let normalized = if range > 1e-12 {
                ((val - min_val) / range * 7.0) as usize
            } else {
                4
            };
            sparkline.push(sparkline_chars[normalized.min(7)]);
        }
        println!("  Ch{ch:02}: {sparkline}");
    }

    println!("  Graph: {} nodes, {} edges", graph.num_nodes, graph.edges.len());
    println!(
        "  Mincut: {:.4}, balance: {:.4}",
        mc.cut_value,
        mc.balance_ratio()
    );
    println!(
        "  Modularity: {:.4}, entropy: {:.4}",
        metrics.modularity,
        metrics.graph_entropy
    );
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_runs_end_to_end() {
        let result = run(PipelineOptions {
            source: PipelineSource::Simulated,
            channels: 4,
            duration: 1.0,
            dashboard: false,
            ruvector_index: false,
        });
        assert!(result.is_ok());
    }

    #[test]
    fn pipeline_with_dashboard() {
        let result = run(PipelineOptions {
            source: PipelineSource::Simulated,
            channels: 4,
            duration: 0.5,
            dashboard: true,
            ruvector_index: false,
        });
        assert!(result.is_ok());
    }

    #[test]
    fn pipeline_with_ruvector_index() {
        let result = run(PipelineOptions {
            source: PipelineSource::Simulated,
            channels: 4,
            duration: 0.5,
            dashboard: false,
            ruvector_index: true,
        });
        assert!(result.is_ok());
    }

    #[test]
    fn virtual_montage_expands_single_channel() {
        let ts = MultiChannelTimeSeries::new(vec![vec![1.0, 2.0, 3.0, 4.0]], 512.0, 0.0)
            .unwrap();
        let expanded = expand_virtual_montage(&ts, 4).unwrap();
        assert_eq!(expanded.num_channels, 4);
        assert_eq!(expanded.num_samples, 4);
    }

    #[test]
    fn plv_graph_has_nodes() {
        let data = generate_data(4, 1000, 1000.0);
        let graph = build_plv_graph(&data, 1000.0);
        assert_eq!(graph.num_nodes, 4);
    }

    #[test]
    fn entropy_non_negative() {
        let data = generate_data(4, 1000, 1000.0);
        let graph = build_plv_graph(&data, 1000.0);
        let e = estimate_entropy(&graph);
        assert!(e >= 0.0);
    }
}
