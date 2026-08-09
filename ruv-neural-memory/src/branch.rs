//! Copy-on-write branched embedding store.
//!
//! Motivation (ADR-0024, §9 row 17 of the SOTA report): per-subject and
//! per-session forks of a shared embedding base, and a checkpoint that can be
//! rolled back after a risky closed-loop experiment, without copying the base.
//!
//! **Provenance note — this is ruv-neural's own design.** A dedicated
//! deep-research sweep looking for prior art on copy-on-write branched vector
//! stores returned **zero independently verified sources**: the only material
//! is vendor self-published documentation (the RuVector RVF format and
//! `agenticow`), which shares a single author and so is not independent
//! corroboration, and no comparable published system — academic or industrial
//! — surfaced. Merge/conflict semantics, garbage collection, and read-through
//! behaviour beyond depth 1 are undocumented there. Accordingly this module
//! implements the *conventional* and well-understood part of the pattern
//! (overlay deltas with read-through resolution, as in overlayfs/ZFS-style
//! CoW) with semantics defined and tested here, and cites no external
//! performance numbers.
//!
//! Semantics, stated explicitly because no external spec pins them down:
//!
//! - A branch stores **only its own edits**; the base is immutable and shared.
//! - Reads resolve **child-first, then along the parent chain** to the base
//!   (arbitrary depth, not just one level).
//! - Deletes are recorded as **tombstones**, so a child can hide a base entry
//!   without mutating it; a tombstone shadows all ancestors.
//! - An id inserted in a child **shadows** the same id in any ancestor.
//! - Branch creation is O(1) in the base size — it allocates an empty delta.
//! - [`BranchStore::rollback`] discards a branch's delta entirely, and
//!   [`BranchStore::checkpoint`] forks a branch so work can proceed on a child
//!   while the parent stays recoverable.
//!
//! Deliberately **not** implemented, because nothing defines them: cross-branch
//! merge/conflict resolution and garbage collection of unreachable branches
//! beyond explicit [`BranchStore::drop_branch`].

use std::collections::{BTreeMap, HashMap, HashSet};

use ruv_neural_core::embedding::NeuralEmbedding;
use ruv_neural_core::error::{Result, RuvNeuralError};

/// Identifier for a branch within a [`BranchStore`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BranchId(pub usize);

impl BranchId {
    /// The root branch, which reads the immutable base directly.
    pub const ROOT: BranchId = BranchId(0);
}

/// One branch's overlay: its own writes plus tombstones for hidden ids.
#[derive(Debug, Clone, Default)]
struct Delta {
    parent: Option<BranchId>,
    writes: BTreeMap<usize, NeuralEmbedding>,
    tombstones: HashSet<usize>,
}

/// A copy-on-write store: an immutable base plus per-branch overlays.
///
/// The base is the set of embeddings present at construction; every mutation
/// lands in a branch delta, so branching is independent of base size and a
/// branch can always be discarded without touching anything else.
#[derive(Debug, Clone)]
pub struct BranchStore {
    /// Immutable shared base, keyed by id.
    base: BTreeMap<usize, NeuralEmbedding>,
    /// Per-branch overlays. `BranchId::ROOT` always exists.
    branches: HashMap<BranchId, Delta>,
    next_branch: usize,
    next_id: usize,
    dimension: Option<usize>,
}

impl BranchStore {
    /// Create a store over an immutable base.
    ///
    /// All base embeddings must share a dimension; the store enforces the
    /// same dimension for every later write.
    pub fn new(base: Vec<NeuralEmbedding>) -> Result<Self> {
        let dimension = base.first().map(|e| e.dimension);
        if let Some(dim) = dimension {
            if let Some(bad) = base.iter().find(|e| e.dimension != dim) {
                return Err(RuvNeuralError::DimensionMismatch {
                    expected: dim,
                    got: bad.dimension,
                });
            }
        }
        let base: BTreeMap<usize, NeuralEmbedding> = base.into_iter().enumerate().collect();
        let next_id = base.len();
        let mut branches = HashMap::new();
        branches.insert(BranchId::ROOT, Delta::default());
        Ok(Self {
            base,
            branches,
            next_branch: 1,
            next_id,
            dimension,
        })
    }

    /// Number of embeddings in the immutable base.
    pub fn base_len(&self) -> usize {
        self.base.len()
    }

    /// Number of live branches, including the root.
    pub fn branch_count(&self) -> usize {
        self.branches.len()
    }

    /// Bytes of overlay state a branch owns, counted as vector elements —
    /// a branch that has written nothing owns nothing, regardless of base
    /// size. Useful for asserting the CoW property in tests and telemetry.
    pub fn delta_len(&self, branch: BranchId) -> Result<usize> {
        let d = self.delta(branch)?;
        Ok(d.writes.len() + d.tombstones.len())
    }

    /// Fork a new branch from `parent`. O(1) in the base size.
    pub fn branch(&mut self, parent: BranchId) -> Result<BranchId> {
        if !self.branches.contains_key(&parent) {
            return Err(RuvNeuralError::Memory(format!(
                "unknown branch {:?}",
                parent
            )));
        }
        let id = BranchId(self.next_branch);
        self.next_branch += 1;
        self.branches.insert(
            id,
            Delta {
                parent: Some(parent),
                ..Delta::default()
            },
        );
        Ok(id)
    }

    /// Fork `branch` and return the child — the "checkpoint before a risky
    /// action" pattern: keep working on the child, and if it goes wrong drop
    /// it and the parent is untouched.
    pub fn checkpoint(&mut self, branch: BranchId) -> Result<BranchId> {
        self.branch(branch)
    }

    /// Discard every write and tombstone a branch owns, returning it to its
    /// parent's view. The parent and base are untouched.
    pub fn rollback(&mut self, branch: BranchId) -> Result<()> {
        let d = self
            .branches
            .get_mut(&branch)
            .ok_or_else(|| RuvNeuralError::Memory(format!("unknown branch {branch:?}")))?;
        d.writes.clear();
        d.tombstones.clear();
        Ok(())
    }

    /// Remove a branch entirely. The root cannot be dropped, nor can a branch
    /// that still has children (which would orphan their read-through chain).
    pub fn drop_branch(&mut self, branch: BranchId) -> Result<()> {
        if branch == BranchId::ROOT {
            return Err(RuvNeuralError::Memory(
                "the root branch cannot be dropped".to_string(),
            ));
        }
        if !self.branches.contains_key(&branch) {
            return Err(RuvNeuralError::Memory(format!("unknown branch {branch:?}")));
        }
        if self.branches.values().any(|d| d.parent == Some(branch)) {
            return Err(RuvNeuralError::Memory(format!(
                "branch {branch:?} still has children"
            )));
        }
        self.branches.remove(&branch);
        Ok(())
    }

    /// Insert an embedding into a branch, returning its id.
    pub fn insert(&mut self, branch: BranchId, embedding: NeuralEmbedding) -> Result<usize> {
        match self.dimension {
            Some(dim) if embedding.dimension != dim => {
                return Err(RuvNeuralError::DimensionMismatch {
                    expected: dim,
                    got: embedding.dimension,
                })
            }
            Some(_) => {}
            None => self.dimension = Some(embedding.dimension),
        }
        if !self.branches.contains_key(&branch) {
            return Err(RuvNeuralError::Memory(format!("unknown branch {branch:?}")));
        }
        let id = self.next_id;
        self.next_id += 1;
        let d = self.branches.get_mut(&branch).expect("checked above");
        d.tombstones.remove(&id);
        d.writes.insert(id, embedding);
        Ok(id)
    }

    /// Overwrite an existing id within a branch. The write shadows any
    /// ancestor's value for that id; ancestors are never mutated.
    pub fn overwrite(
        &mut self,
        branch: BranchId,
        id: usize,
        embedding: NeuralEmbedding,
    ) -> Result<()> {
        if let Some(dim) = self.dimension {
            if embedding.dimension != dim {
                return Err(RuvNeuralError::DimensionMismatch {
                    expected: dim,
                    got: embedding.dimension,
                });
            }
        }
        if self.get(branch, id)?.is_none() {
            return Err(RuvNeuralError::Memory(format!(
                "id {id} is not visible from branch {branch:?}"
            )));
        }
        let d = self.branches.get_mut(&branch).expect("get() validated it");
        d.tombstones.remove(&id);
        d.writes.insert(id, embedding);
        Ok(())
    }

    /// Hide an id from a branch's view with a tombstone. Ancestors keep it.
    pub fn delete(&mut self, branch: BranchId, id: usize) -> Result<bool> {
        if self.get(branch, id)?.is_none() {
            return Ok(false);
        }
        let d = self.branches.get_mut(&branch).expect("get() validated it");
        d.writes.remove(&id);
        d.tombstones.insert(id);
        Ok(true)
    }

    /// Read-through lookup: the branch's own write, else the nearest
    /// ancestor's, else the base. A tombstone anywhere in the chain hides the
    /// id from that point down.
    pub fn get(&self, branch: BranchId, id: usize) -> Result<Option<&NeuralEmbedding>> {
        let mut cursor = Some(branch);
        while let Some(b) = cursor {
            let d = self.delta(b)?;
            if d.tombstones.contains(&id) {
                return Ok(None);
            }
            if let Some(e) = d.writes.get(&id) {
                return Ok(Some(e));
            }
            cursor = d.parent;
        }
        Ok(self.base.get(&id))
    }

    /// All ids visible from a branch, ascending. Shadowed ids appear once;
    /// tombstoned ids do not appear.
    pub fn visible_ids(&self, branch: BranchId) -> Result<Vec<usize>> {
        let mut hidden: HashSet<usize> = HashSet::new();
        let mut visible: BTreeMap<usize, ()> = BTreeMap::new();
        let mut cursor = Some(branch);
        while let Some(b) = cursor {
            let d = self.delta(b)?;
            for id in &d.tombstones {
                if !visible.contains_key(id) {
                    hidden.insert(*id);
                }
            }
            for id in d.writes.keys() {
                if !hidden.contains(id) {
                    visible.insert(*id, ());
                }
            }
            cursor = d.parent;
        }
        for id in self.base.keys() {
            if !hidden.contains(id) {
                visible.insert(*id, ());
            }
        }
        Ok(visible.into_keys().collect())
    }

    /// Number of embeddings visible from a branch.
    pub fn len(&self, branch: BranchId) -> Result<usize> {
        Ok(self.visible_ids(branch)?.len())
    }

    /// Whether a branch sees no embeddings.
    pub fn is_empty(&self, branch: BranchId) -> Result<bool> {
        Ok(self.len(branch)? == 0)
    }

    /// k nearest neighbours by Euclidean distance over a branch's visible set,
    /// ascending by distance. Ties break on ascending id, so results are
    /// deterministic.
    pub fn query_nearest(
        &self,
        branch: BranchId,
        query: &NeuralEmbedding,
        k: usize,
    ) -> Result<Vec<(usize, f64)>> {
        let mut scored: Vec<(usize, f64)> = Vec::new();
        for id in self.visible_ids(branch)? {
            if let Some(e) = self.get(branch, id)? {
                if let Ok(d) = e.euclidean_distance(query) {
                    scored.push((id, d));
                }
            }
        }
        scored.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(k);
        Ok(scored)
    }

    /// Depth of a branch below the root (root itself is 0).
    pub fn depth(&self, branch: BranchId) -> Result<usize> {
        let mut depth = 0;
        let mut cursor = self.delta(branch)?.parent;
        while let Some(b) = cursor {
            depth += 1;
            cursor = self.delta(b)?.parent;
        }
        Ok(depth)
    }

    fn delta(&self, branch: BranchId) -> Result<&Delta> {
        self.branches
            .get(&branch)
            .ok_or_else(|| RuvNeuralError::Memory(format!("unknown branch {branch:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruv_neural_core::brain::Atlas;
    use ruv_neural_core::embedding::EmbeddingMetadata;

    fn meta() -> EmbeddingMetadata {
        EmbeddingMetadata {
            subject_id: None,
            session_id: None,
            cognitive_state: None,
            source_atlas: Atlas::Custom(2),
            embedding_method: "test".to_string(),
        }
    }

    fn emb(x: f64, y: f64) -> NeuralEmbedding {
        NeuralEmbedding::new(vec![x, y], 0.0, meta()).unwrap()
    }

    fn store() -> BranchStore {
        BranchStore::new(vec![emb(0.0, 0.0), emb(1.0, 0.0), emb(2.0, 0.0)]).unwrap()
    }

    #[test]
    fn branching_is_independent_of_base_size() {
        // A branch owns nothing until it writes — the CoW property.
        let big: Vec<NeuralEmbedding> = (0..10_000).map(|i| emb(i as f64, 0.0)).collect();
        let mut s = BranchStore::new(big).unwrap();
        let b = s.branch(BranchId::ROOT).unwrap();
        assert_eq!(s.delta_len(b).unwrap(), 0);
        assert_eq!(s.len(b).unwrap(), 10_000, "child sees the whole base");
        s.insert(b, emb(-1.0, 0.0)).unwrap();
        assert_eq!(s.delta_len(b).unwrap(), 1, "only the new write is owned");
    }

    #[test]
    fn child_writes_do_not_touch_the_parent() {
        let mut s = store();
        let child = s.branch(BranchId::ROOT).unwrap();
        let id = s.insert(child, emb(9.0, 9.0)).unwrap();
        assert!(s.get(child, id).unwrap().is_some());
        assert!(s.get(BranchId::ROOT, id).unwrap().is_none());
        assert_eq!(s.len(BranchId::ROOT).unwrap(), 3);
        assert_eq!(s.len(child).unwrap(), 4);
    }

    #[test]
    fn child_overwrite_shadows_the_ancestor_value() {
        let mut s = store();
        let child = s.branch(BranchId::ROOT).unwrap();
        s.overwrite(child, 1, emb(99.0, 0.0)).unwrap();
        assert_eq!(s.get(child, 1).unwrap().unwrap().vector, vec![99.0, 0.0]);
        assert_eq!(
            s.get(BranchId::ROOT, 1).unwrap().unwrap().vector,
            vec![1.0, 0.0],
            "base is immutable"
        );
        // Overwriting an invisible id is an error, not a silent insert.
        assert!(s.overwrite(child, 999, emb(0.0, 0.0)).is_err());
    }

    #[test]
    fn tombstones_hide_base_entries_without_mutating_them() {
        let mut s = store();
        let child = s.branch(BranchId::ROOT).unwrap();
        assert!(s.delete(child, 1).unwrap());
        assert!(s.get(child, 1).unwrap().is_none());
        assert_eq!(s.visible_ids(child).unwrap(), vec![0, 2]);
        assert_eq!(s.visible_ids(BranchId::ROOT).unwrap(), vec![0, 1, 2]);
        // Deleting an already-hidden id reports no change.
        assert!(!s.delete(child, 1).unwrap());
        assert!(!s.delete(child, 999).unwrap());
    }

    #[test]
    fn read_through_resolves_along_deep_chains() {
        // The behaviour the vendor docs leave undefined: depth > 1.
        let mut s = store();
        let a = s.branch(BranchId::ROOT).unwrap();
        let b = s.branch(a).unwrap();
        let c = s.branch(b).unwrap();
        assert_eq!(s.depth(c).unwrap(), 3);

        // Written at depth 1, read at depth 3.
        s.overwrite(a, 0, emb(10.0, 0.0)).unwrap();
        assert_eq!(s.get(c, 0).unwrap().unwrap().vector, vec![10.0, 0.0]);

        // Nearest ancestor wins over a farther one.
        s.overwrite(b, 0, emb(20.0, 0.0)).unwrap();
        assert_eq!(s.get(c, 0).unwrap().unwrap().vector, vec![20.0, 0.0]);
        assert_eq!(s.get(a, 0).unwrap().unwrap().vector, vec![10.0, 0.0]);

        // A tombstone at depth 2 hides it from depth 3 but not from depth 1.
        s.delete(b, 0).unwrap();
        assert!(s.get(c, 0).unwrap().is_none());
        assert!(s.get(a, 0).unwrap().is_some());

        // ...and a re-write at depth 3 resurrects it locally.
        s.insert(c, emb(30.0, 0.0)).unwrap();
        let ids = s.visible_ids(c).unwrap();
        assert!(!ids.contains(&0), "tombstone still hides the original id");
    }

    #[test]
    fn rollback_discards_only_the_branch_delta() {
        let mut s = store();
        let child = s.branch(BranchId::ROOT).unwrap();
        s.insert(child, emb(5.0, 5.0)).unwrap();
        s.delete(child, 0).unwrap();
        s.overwrite(child, 2, emb(7.0, 7.0)).unwrap();
        assert_eq!(s.delta_len(child).unwrap(), 3);

        s.rollback(child).unwrap();
        assert_eq!(s.delta_len(child).unwrap(), 0);
        assert_eq!(s.visible_ids(child).unwrap(), vec![0, 1, 2]);
        assert_eq!(s.get(child, 2).unwrap().unwrap().vector, vec![2.0, 0.0]);
        assert_eq!(s.len(BranchId::ROOT).unwrap(), 3);
    }

    #[test]
    fn checkpoint_then_discard_leaves_the_parent_intact() {
        // The closed-loop pattern: checkpoint before a risky experiment,
        // discard the child if it goes wrong.
        let mut s = store();
        let session = s.branch(BranchId::ROOT).unwrap();
        s.insert(session, emb(4.0, 0.0)).unwrap();
        let before = s.visible_ids(session).unwrap();

        let trial = s.checkpoint(session).unwrap();
        s.delete(trial, 0).unwrap();
        s.insert(trial, emb(-99.0, -99.0)).unwrap();
        assert_ne!(s.visible_ids(trial).unwrap(), before);

        s.drop_branch(trial).unwrap();
        assert_eq!(s.visible_ids(session).unwrap(), before);
        assert!(s.get(trial, 0).is_err(), "dropped branch is gone");
    }

    #[test]
    fn drop_branch_protects_the_root_and_parents_with_children() {
        let mut s = store();
        let a = s.branch(BranchId::ROOT).unwrap();
        let b = s.branch(a).unwrap();
        assert!(s.drop_branch(BranchId::ROOT).is_err());
        assert!(s.drop_branch(a).is_err(), "would orphan its child");
        s.drop_branch(b).unwrap();
        s.drop_branch(a).unwrap();
        assert_eq!(s.branch_count(), 1);
    }

    #[test]
    fn queries_respect_the_branch_view() {
        let mut s = store();
        let child = s.branch(BranchId::ROOT).unwrap();
        s.insert(child, emb(0.1, 0.0)).unwrap();
        s.delete(child, 0).unwrap();

        let q = emb(0.0, 0.0);
        // Root: exact match on id 0 is nearest.
        let root_hits = s.query_nearest(BranchId::ROOT, &q, 2).unwrap();
        assert_eq!(root_hits[0].0, 0);
        assert!(root_hits[0].1.abs() < 1e-12);
        // Child: id 0 is tombstoned, so the newly inserted 0.1 wins.
        let child_hits = s.query_nearest(child, &q, 2).unwrap();
        assert_eq!(child_hits[0].0, 3);
        assert!((child_hits[0].1 - 0.1).abs() < 1e-12);
        assert!(child_hits.iter().all(|(id, _)| *id != 0));
        assert_eq!(s.query_nearest(child, &q, 0).unwrap().len(), 0);
    }

    #[test]
    fn enforces_dimension_consistency() {
        assert!(BranchStore::new(vec![
            emb(0.0, 0.0),
            NeuralEmbedding::new(vec![1.0, 2.0, 3.0], 0.0, meta()).unwrap()
        ])
        .is_err());
        let mut s = store();
        let child = s.branch(BranchId::ROOT).unwrap();
        assert!(s
            .insert(
                child,
                NeuralEmbedding::new(vec![1.0, 2.0, 3.0], 0.0, meta()).unwrap()
            )
            .is_err());
        assert!(s
            .overwrite(
                child,
                1,
                NeuralEmbedding::new(vec![1.0, 2.0, 3.0], 0.0, meta()).unwrap()
            )
            .is_err());
    }

    #[test]
    fn unknown_branches_are_errors_everywhere() {
        let mut s = store();
        let ghost = BranchId(42);
        assert!(s.get(ghost, 0).is_err());
        assert!(s.visible_ids(ghost).is_err());
        assert!(s.insert(ghost, emb(0.0, 0.0)).is_err());
        assert!(s.branch(ghost).is_err());
        assert!(s.rollback(ghost).is_err());
        assert!(s.drop_branch(ghost).is_err());
        assert!(s.delta_len(ghost).is_err());
    }

    #[test]
    fn sibling_branches_are_isolated() {
        let mut s = store();
        let a = s.branch(BranchId::ROOT).unwrap();
        let b = s.branch(BranchId::ROOT).unwrap();
        s.overwrite(a, 1, emb(11.0, 0.0)).unwrap();
        s.delete(b, 1).unwrap();
        assert_eq!(s.get(a, 1).unwrap().unwrap().vector, vec![11.0, 0.0]);
        assert!(s.get(b, 1).unwrap().is_none());
        assert_eq!(
            s.get(BranchId::ROOT, 1).unwrap().unwrap().vector,
            vec![1.0, 0.0]
        );
    }

    #[test]
    fn empty_base_is_usable() {
        let mut s = BranchStore::new(vec![]).unwrap();
        assert_eq!(s.base_len(), 0);
        assert!(s.is_empty(BranchId::ROOT).unwrap());
        let child = s.branch(BranchId::ROOT).unwrap();
        let id = s.insert(child, emb(1.0, 1.0)).unwrap();
        assert_eq!(s.len(child).unwrap(), 1);
        assert_eq!(s.get(child, id).unwrap().unwrap().vector, vec![1.0, 1.0]);
    }
}
