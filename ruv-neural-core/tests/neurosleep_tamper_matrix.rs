//! Independent AT-05 coverage over the released JSON representation.
//!
//! The unit tests beside the contract exercise representative Rust fields.
//! This integration test walks every serialized leaf and object so newly added
//! wire fields cannot silently escape integrity or unknown-field enforcement.

use ruv_neural_core::attestation::{verify_neurosleep_bundle, SignedNeuroSleepBundleV1};
use serde_json::Value;

#[derive(Clone, Debug)]
enum Segment {
    Key(String),
    Index(usize),
}

fn collect_paths(
    value: &Value,
    path: &mut Vec<Segment>,
    leaves: &mut Vec<Vec<Segment>>,
    objects: &mut Vec<Vec<Segment>>,
) {
    match value {
        Value::Object(map) => {
            objects.push(path.clone());
            for (key, child) in map {
                path.push(Segment::Key(key.clone()));
                collect_paths(child, path, leaves, objects);
                path.pop();
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                path.push(Segment::Index(index));
                collect_paths(child, path, leaves, objects);
                path.pop();
            }
        }
        _ => leaves.push(path.clone()),
    }
}

fn value_at_mut<'a>(mut value: &'a mut Value, path: &[Segment]) -> &'a mut Value {
    for segment in path {
        value = match segment {
            Segment::Key(key) => value
                .as_object_mut()
                .and_then(|map| map.get_mut(key))
                .expect("object path collected from this JSON value"),
            Segment::Index(index) => value
                .as_array_mut()
                .and_then(|items| items.get_mut(*index))
                .expect("array path collected from this JSON value"),
        };
    }
    value
}

fn mutate_leaf(value: &mut Value) {
    match value {
        Value::Null => *value = Value::String("tampered".into()),
        Value::Bool(boolean) => *boolean = !*boolean,
        Value::Number(number) => {
            if let Some(unsigned) = number.as_u64() {
                *value = Value::from(unsigned.saturating_add(1));
            } else if let Some(signed) = number.as_i64() {
                *value = Value::from(signed.saturating_add(1));
            } else {
                *value = Value::from(number.as_f64().expect("JSON number") + 0.001);
            }
        }
        Value::String(text) => text.push_str("-tampered"),
        Value::Array(_) | Value::Object(_) => unreachable!("only leaf paths are mutated"),
    }
}

fn label(path: &[Segment]) -> String {
    let mut out = String::new();
    for segment in path {
        out.push('/');
        match segment {
            Segment::Key(key) => out.push_str(key),
            Segment::Index(index) => out.push_str(&index.to_string()),
        }
    }
    out
}

fn fixture() -> (Value, [u8; 32]) {
    let bundle = serde_json::from_str(include_str!("fixtures/neurosleep-v1/valid_bundle.json"))
        .expect("valid golden bundle JSON");
    let trust: Value =
        serde_json::from_str(include_str!("fixtures/neurosleep-v1/trust_profile.json"))
            .expect("valid trust profile JSON");
    let bytes: Vec<u8> = trust["verifying_key_ed25519"]
        .as_array()
        .expect("fixture key array")
        .iter()
        .map(|value| value.as_u64().expect("key byte") as u8)
        .collect();
    (bundle, bytes.try_into().expect("32-byte fixture key"))
}

#[test]
fn every_serialized_leaf_is_rejected_after_one_field_tamper() {
    let (original, trusted_key) = fixture();
    let clean: SignedNeuroSleepBundleV1 =
        serde_json::from_value(original.clone()).expect("typed golden bundle");
    verify_neurosleep_bundle(&clean, &trusted_key).expect("golden bundle verifies");

    let mut leaves = Vec::new();
    let mut objects = Vec::new();
    collect_paths(&original, &mut Vec::new(), &mut leaves, &mut objects);
    assert!(
        leaves.len() > 100,
        "fixture unexpectedly lost contract fields"
    );

    for path in leaves {
        let mut tampered = original.clone();
        mutate_leaf(value_at_mut(&mut tampered, &path));
        if let Ok(bundle) = serde_json::from_value::<SignedNeuroSleepBundleV1>(tampered) {
            assert!(
                verify_neurosleep_bundle(&bundle, &trusted_key).is_err(),
                "tampered leaf was accepted: {}",
                label(&path)
            );
        }
    }
}

#[test]
fn every_serialized_object_rejects_an_unknown_field() {
    let (original, _) = fixture();
    let mut leaves = Vec::new();
    let mut objects = Vec::new();
    collect_paths(&original, &mut Vec::new(), &mut leaves, &mut objects);

    for path in objects {
        let mut tampered = original.clone();
        value_at_mut(&mut tampered, &path)
            .as_object_mut()
            .expect("object path")
            .insert("_unknown_neurosleep_field".into(), Value::Bool(true));
        assert!(
            serde_json::from_value::<SignedNeuroSleepBundleV1>(tampered).is_err(),
            "unknown field was accepted at {}",
            label(&path)
        );
    }
}
