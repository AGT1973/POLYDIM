// POLYDIM_DEST
// destination: polydim/core/tests/
// filename:    integration.rs
// author:      ai.mpat.agt@gmail.com
//
// Integration tests for polydim crate
// Run with: cargo test --test integration
// Resolves: TASK_024 (partial) — estructura para TASK_025

use polydim::{Space, ObjectND, Pbf0, Pbf0Meta, bind, sup, proj, sim, N, UMBRAL};
use std::collections::HashMap;

// ─── PBF0 interop ─────────────────────────────────────────────────────────────

#[test]
fn test_pbf0_from_python_fixture() {
    // Golden fixture: mini PBF0 stream (N=5) hand-crafted to match Python output
    let meta_json = r#"{"geo_id":"aabbccddeeff","dims":["DIM_SQL"],"weights":[0.8],"modo":"H","version":"0.3"}"#;
    let hv_raw: [f32; 5] = [1.0, 0.5, -0.5, 0.25, -0.25];
    let n: u32 = 5;
    let meta_len: u32 = meta_json.len() as u32;

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"PDIM");
    buf.extend_from_slice(&0u16.to_be_bytes());
    buf.extend_from_slice(&n.to_be_bytes());
    buf.extend_from_slice(&meta_len.to_be_bytes());
    buf.extend_from_slice(meta_json.as_bytes());
    for &f in &hv_raw {
        buf.extend_from_slice(&f.to_le_bytes());
    }

    let pbf0 = Pbf0::deserialize(&buf).expect("should parse mini-fixture");
    assert_eq!(pbf0.meta.geo_id, "aabbccddeeff");
    assert_eq!(pbf0.meta.dims, vec!["DIM_SQL"]);
    assert!((pbf0.meta.weights[0] - 0.8).abs() < 1e-5);
    assert_eq!(pbf0.hypervec.len(), 5);
    assert!((pbf0.hypervec[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_full_object_roundtrip_via_pbf0() {
    let mut sp  = Space::new("INTEG_TEST");
    let mut obj = ObjectND::new(&mut sp);
    let mut props = HashMap::new();
    props.insert("table".to_string(), "orders".to_string());
    obj.add("DIM_SQL",  props, 0.9);
    obj.add("DIM_RUST", HashMap::new(), 0.7);

    let hv = obj.hv();
    let meta = Pbf0Meta {
        geo_id:  obj.geo_id(),
        dims:    vec!["DIM_SQL".to_string(), "DIM_RUST".to_string()],
        weights: vec![0.9, 0.7],
        modo:    "H".to_string(),
        version: "0.3".to_string(),
    };

    let bytes   = Pbf0::serialize(&hv, &meta).expect("serialize");
    let decoded = Pbf0::deserialize(&bytes).expect("deserialize");

    let mae: f32 = hv.iter().zip(decoded.hypervec.iter())
        .map(|(a, b)| (a - b).abs()).sum::<f32>() / hv.len() as f32;
    assert!(mae < 1e-6, "MAE={}", mae);
    assert_eq!(decoded.meta.dims, meta.dims);
    assert_eq!(decoded.meta.modo, "H");
}

// ─── Space cross-seed isolation ───────────────────────────────────────────────

#[test]
fn test_spaces_with_different_seeds_are_isolated() {
    let mut sp_a = Space::new("AI_ALPHA");
    let mut sp_b = Space::new("AI_BETA");
    let sub_a = sp_a.sub("DIM_SQL").clone();
    let sub_b = sp_b.sub("DIM_SQL").clone();
    let s = sim(&sub_a, &sub_b);
    assert!(s < 0.6, "cross-seed similarity too high: {}", s);
}

// ─── VSA algebraic properties ─────────────────────────────────────────────────

#[test]
fn test_bind_commutativity() {
    let mut sp = Space::new("COMM_TEST");
    let sa = sp.sym("X").clone();
    let sb = sp.sym("Y").clone();
    let ab = bind(&sa, &sb);
    let ba = bind(&sb, &sa);
    let diff: f32 = ab.iter().zip(ba.iter()).map(|(x,y)| (x-y).abs()).sum();
    assert!(diff < 1e-5, "bind not commutative, diff={}", diff);
}

#[test]
fn test_proj_self_is_one() {
    let mut sp = Space::new("SELF_TEST");
    let sub = sp.sub("DIM_SQL").clone();
    let p = proj(&sub, &sub);
    assert!((p - 1.0).abs() < 1e-4, "self-projection={}", p);
}

#[test]
fn test_proj_orthogonal_near_half() {
    let mut sp = Space::new("ORTHO_TEST");
    let a = sp.sub("DIM_SQL").clone();
    let b = sp.sub("DIM_PYTHON").clone();
    let p = proj(&a, &b);
    assert!((p - 0.5).abs() < 0.05, "expected ~0.5, got {}", p);
}
