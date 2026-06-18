//! rUv Neural Viz — Brain topology visualization data structures and ASCII rendering.
//!
//! This crate provides:
//! - **Layout algorithms**: Force-directed, anatomical (MNI), and circular layouts
//! - **Color mapping**: Cool-warm, viridis, and module-color schemes
//! - **ASCII rendering**: Terminal-friendly graph, mincut, sparkline, and dashboard views
//! - **Export**: D3.js JSON, Graphviz DOT, GEXF, and CSV timeline formats
//! - **Animation**: Frame generation from temporal brain graph sequences
//!
//! # `no_std`
//!
//! The crate is `no_std` when built `--no-default-features`. In that mode only
//! [`colormap`] is available — it needs just `alloc` (for `Vec`/`String`) and
//! does all its float math without `std`/`libm`, so it runs on bare-metal
//! targets such as the ESP32. The graph-bound modules (`animation`, `ascii`,
//! `export`, `layout`) and their heavy deps are gated behind the default `std`
//! feature.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

// no_std-friendly (alloc-only) — always available, incl. on ESP32.
pub mod colormap;
pub use colormap::ColorMap;

// std/graph-bound modules — only with the default `std` feature.
#[cfg(feature = "std")]
pub mod animation;
#[cfg(feature = "std")]
pub mod ascii;
#[cfg(feature = "std")]
pub mod export;
#[cfg(feature = "std")]
pub mod layout;

#[cfg(feature = "std")]
pub use animation::{AnimatedEdge, AnimatedNode, AnimationFrame, AnimationFrames, LayoutType};
#[cfg(feature = "std")]
pub use layout::{AnatomicalLayout, ForceDirectedLayout};
