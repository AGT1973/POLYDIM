// POLYDIM_DEST
// destination: polydim/core/
// filename:    polydim_core_v03.rs
// author:      polydim.ai.lenguage@gmail.com

//! POLYDIM Core — Rust V0.3
//! ==========================
//! TASK_026_BUG — BUG_005 + BUG_006 + fix DIM_CONTRACT faltante
//!
//! Cambios respecto a V0.2 (ai.mpat.agt, TASK_024):
//!
//!   BUG_DESCUBIERTO: DIM_CONTRACT ausente de NATIVE_DIMS en V0.2
//!     → 9 dims en Rust vs 10 en Python → activación incorrecta para objetos
//!       con DIM_CONTRACT. Fix: agregar "DIM_CONTRACT" a NATIVE_DIMS.
//!
//!   BUG_005 — API pública ObjectND:
//!     + get_dims()        → Vec<&str> con w > 0
//!     + get_weight(dim)   → Option<f32>
//!     + set_weight(dim,w) → Result<(), &str> — invalida cache
//!     + invalidate_cache()
//!
//!   BUG_006 — N dinámico en PBP:
//!     decode_hv() mantiene firma [f32; N] pero error message mejorado.
//!     Nueva: decode_hv_dynamic() → Result<Vec<f32>, String>
//!       Lee N del header PBP, devuelve Vec del tamaño correcto.
//!
//!   BUG_008 (doc): _bind es Hadamard product, no convolución circular.
//!     Decisión: mantener Hadamard — breaking change si se cambia.
//!
//! Tests: 18 tests (10 nuevos en V0.3 + 8 regresión V0.1/V0.2)
//! Autor: polydim.ai.lenguage@gmail.com
//! V0.3 — 2026-06-19 — TASK_026_BUG

use std::collections::HashMap;

pub const N: usize = 10_000;
const CONTENT_W: f32 = 0.3;
const UMBRAL: f32 = 0.5 + 2.0 * (1.0 / (2.0 * 100.0));

/// V0.3 fix: DIM_CONTRACT agregado (faltaba en V0.2).
pub const NATIVE_DIMS: &[&str] = &[
    "DIM_PYTHON", "DIM_RUST", "DIM_FLUTTER", "DIM_SQL",
    "DIM_GRAPH",  "DIM_VECTOR", "DIM_TIME", "DIM_ERROR", "DIM_META",
    "DIM_CONTRACT",
];

pub fn bind(a: &[f32; N], b: &[f32; N]) -> [f32; N] {
    let mut r = [0f32; N];
    for i in 0..N { r[i] = a[i] * b[i]; }
    normalize(r)
}

pub fn sup(hvs: &[(&[f32; N], f32)]) -> [f32; N] {
    let mut acc = [0f32; N];
    for (hv, w) in hvs { for i in 0..N { acc[i] += w * hv[i]; } }
    normalize(acc)
}

pub fn proj(hv: &[f32; N], sub: &[f32; N]) -> f32 {
    let dot: f32 = hv.iter().zip(sub.iter()).map(|(a, b)| a * b).sum();
    (dot + 1.0) / 2.0
}

pub fn sim(a: &[f32; N], b: &[f32; N]) -> f32 { proj(a, b) }

pub fn normalize(mut v: [f32; N]) -> [f32; N] {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 { for x in v.iter_mut() { *x /= norm; } }
    v
}

/// BUG_001: usa LCG, no PCG64. Solo Rust-interno.
/// Para interop Python → Space::native_sync().
pub fn make_hv(seed: u64) -> [f32; N] {
    let mut hv = [0f32; N];
    let mut state = seed;
    let lcg_next = |s: u64| -> u64 {
        s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
    };
    let mut i = 0;
    while i < N {
        state = lcg_next(state);
        let u1 = (state >> 11) as f32 / (1u64 << 53) as f32;
        state = lcg_next(state);
        let u2 = (state >> 11) as f32 / (1u64 << 53) as f32;
        let r = (-2.0 * (u1 + f32::EPSILON).ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        hv[i] = r * theta.cos();
        if i + 1 < N { hv[i + 1] = r * theta.sin(); }
        i += 2;
    }
    normalize(hv)
}

pub fn seed_from_str(s: &str) -> u64 {
    let mut hash: u64 = 14695981039346656037;
    for byte in s.bytes() { hash ^= byte as u64; hash = hash.wrapping_mul(1099511628211); }
    hash
}

pub struct Space {
    pub personal_seed: String,
    sym_cache: HashMap<String, Box<[f32; N]>>,
    sub_cache: HashMap<String, Box<[f32; N]>>,
    pub synced: bool,
}

impl Space {
    pub fn new(personal_seed: &str) -> Self {
        let mut sp = Space {
            personal_seed: personal_seed.to_string(),
            sym_cache: HashMap::new(), sub_cache: HashMap::new(), synced: false,
        };
        for d in NATIVE_DIMS { sp.sub(d); }
        sp
    }

    pub fn sym(&mut self, name: &str) -> &[f32; N] {
        if !self.sym_cache.contains_key(name) {
            let key = if self.personal_seed.is_empty() { name.to_string() }
                      else { format!("{}:{}", self.personal_seed, name) };
            let hv = make_hv(seed_from_str(&key));
            self.sym_cache.insert(name.to_string(), Box::new(hv));
        }
        self.sym_cache[name].as_ref()
    }

    pub fn sub(&mut self, name: &str) -> &[f32; N] {
        if !self.sub_cache.contains_key(name) {
            let hv = *self.sym(name);
            self.sub_cache.insert(name.to_string(), Box::new(hv));
        }
        self.sub_cache[name].as_ref()
    }

    pub fn random_hv() -> [f32; N] {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64 ^ d.as_secs().wrapping_mul(0xdeadbeef))
            .unwrap_or(42);
        make_hv(seed)
    }

    pub fn enc(&mut self, props: &HashMap<String, String>) -> [f32; N] {
        if props.is_empty() { return *self.sym("__empty__"); }
        let pairs: Vec<([f32; N], f32)> = props.iter().map(|(k, v)| {
            let sk = *self.sym(k); let sv = *self.sym(v);
            (bind(&sk, &sv), 1.0)
        }).collect();
        let refs: Vec<(&[f32; N], f32)> = pairs.iter().map(|(hv, w)| (hv as &[f32; N], *w)).collect();
        sup(&refs)
    }

    pub fn native_sync(&mut self, data: &[u8]) -> Result<usize, &'static str> {
        if data.len() < 6 { return Err("native_sync: frame demasiado corto"); }
        if &data[0..2] != &[0x50, 0x44] { return Err("native_sync: magic inválido"); }
        if data[2] != 0x00 { return Err("native_sync: versión no soportada"); }
        if data[3] != 0x10 { return Err("native_sync: flags no indican NATIVE_SYNC"); }
        let ndims = data[4] as usize;
        let mut pos = 5usize; let mut count = 0usize;
        for _ in 0..ndims {
            if pos >= data.len() { return Err("native_sync: frame truncado en nombre"); }
            let name_len = data[pos] as usize; pos += 1;
            if pos + name_len > data.len() { return Err("native_sync: frame truncado"); }
            let name = std::str::from_utf8(&data[pos..pos + name_len])
                .map_err(|_| "native_sync: nombre no UTF-8")?;
            pos += name_len;
            if pos + 4 * N > data.len() { return Err("native_sync: frame truncado en vector"); }
            let mut hv = [0f32; N];
            for i in 0..N { let o = pos + i * 4; hv[i] = f32::from_le_bytes(data[o..o+4].try_into().unwrap()); }
            pos += 4 * N;
            self.sub_cache.insert(name.to_string(), Box::new(hv));
            self.sym_cache.insert(name.to_string(), Box::new(hv));
            count += 1;
        }
        self.synced = true; Ok(count)
    }
}

pub struct ObjectND {
    pub geo: [f32; N],
    pub dims: HashMap<String, (HashMap<String, String>, f32)>,
    pub cache: Option<[f32; N]>,
}

impl ObjectND {
    pub fn new() -> Self {
        ObjectND { geo: Space::random_hv(), dims: HashMap::new(), cache: None }
    }

    pub fn add(&mut self, dim: &str, props: HashMap<String, String>, w: f32) -> &mut Self {
        self.dims.insert(dim.to_string(), (props, w.clamp(0.0, 1.0)));
        self.cache = None; self
    }

    pub fn geo_id(&self) -> String {
        format!("{:012x}", seed_from_str(&format!("{:?}", &self.geo[..8])) & 0xffffffffffff)
    }

    pub fn hv(&mut self, space: &mut Space) -> [f32; N] {
        if let Some(c) = self.cache { return c; }
        let mut components: Vec<([f32; N], f32)> = vec![(self.geo, 1.0)];
        for (dim, (props, w)) in &self.dims {
            if *w <= 0.0 { continue; }
            let sub_hv = *space.sub(dim);
            components.push((sub_hv, *w));
            let content_hv = bind(&sub_hv, &space.enc(props));
            components.push((content_hv, w * CONTENT_W));
        }
        let refs: Vec<(&[f32; N], f32)> = components.iter().map(|(hv, w)| (hv as &[f32; N], *w)).collect();
        let result = sup(&refs);
        self.cache = Some(result); result
    }

    pub fn activacion(&mut self, space: &mut Space, dim: &str) -> f32 {
        let hv = self.hv(space); proj(&hv, space.sub(dim))
    }

    pub fn dims_activas(&mut self, space: &mut Space) -> HashMap<String, f32> {
        let mut result = HashMap::new();
        let all_dims: Vec<String> = NATIVE_DIMS.iter().map(|s| s.to_string())
            .chain(self.dims.keys().cloned())
            .collect::<std::collections::HashSet<_>>().into_iter().collect();
        for dim in all_dims {
            let a = self.activacion(space, &dim);
            if a > UMBRAL { result.insert(dim, (a * 10000.0).round() / 10000.0); }
        }
        result
    }

    // BUG_005: API pública

    pub fn get_dims(&self) -> Vec<&str> {
        self.dims.iter().filter(|(_, (_, w))| *w > 0.0).map(|(d, _)| d.as_str()).collect()
    }

    pub fn get_weight(&self, dim: &str) -> Option<f32> {
        self.dims.get(dim).map(|(_, w)| *w)
    }

    pub fn set_weight(&mut self, dim: &str, w: f32) -> Result<(), &'static str> {
        if let Some(entry) = self.dims.get_mut(dim) {
            entry.1 = w.clamp(0.0, 1.0); self.cache = None; Ok(())
        } else {
            Err("dimensión no declarada; usar add() primero")
        }
    }

    pub fn invalidate_cache(&mut self) { self.cache = None; }
}

pub mod pbp {
    use super::{N, ObjectND, Space};
    pub const MAGIC: [u8; 2]       = [0x50, 0x44];
    pub const VERSION: u8          = 0x00;
    pub const FLAG_NONE: u8        = 0x00;
    pub const FLAG_HAS_GEO_ID: u8  = 0x01;
    pub const FLAG_HAS_WEIGHTS: u8 = 0x02;
    pub const FLAG_NATIVE_SYNC: u8 = 0x10;

    pub fn encode_hv(hv: &[f32; N]) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + 4 * N);
        out.extend_from_slice(&MAGIC); out.push(VERSION); out.push(FLAG_NONE);
        out.extend_from_slice(&(N as u32).to_le_bytes());
        for f in hv { out.extend_from_slice(&f.to_le_bytes()); }
        out
    }

    pub fn encode_object(obj: &mut ObjectND, space: &mut Space) -> Vec<u8> {
        let hv = obj.hv(space);
        let geo = obj.geo_id(); let geo_b = geo.as_bytes();
        let mut wb: Vec<u8> = vec![obj.dims.len() as u8];
        for (dim, (_, w)) in &obj.dims {
            let dn = dim.as_bytes(); wb.push(dn.len() as u8);
            wb.extend_from_slice(dn); wb.extend_from_slice(&w.to_le_bytes());
        }
        let mut out = Vec::with_capacity(8 + 4 * N + 1 + geo_b.len() + wb.len());
        out.extend_from_slice(&MAGIC); out.push(VERSION); out.push(FLAG_HAS_GEO_ID | FLAG_HAS_WEIGHTS);
        out.extend_from_slice(&(N as u32).to_le_bytes());
        for f in &hv { out.extend_from_slice(&f.to_le_bytes()); }
        out.push(geo_b.len() as u8); out.extend_from_slice(geo_b); out.extend_from_slice(&wb); out
    }

    /// BUG_006: error message mejorado cuando N no coincide.
    pub fn decode_hv(data: &[u8]) -> Result<[f32; N], String> {
        if data.len() < 8 { return Err("frame demasiado corto".into()); }
        if &data[0..2] != &MAGIC { return Err("magic inválido".into()); }
        let frame_n = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        if frame_n != N {
            return Err(format!(
                "N no coincide: frame={} compilado={}. Usar decode_hv_dynamic() para N dinámico.",
                frame_n, N
            ));
        }
        if data.len() < 8 + 4 * N { return Err("frame truncado".into()); }
        let mut hv = [0f32; N];
        for i in 0..N { let o = 8 + i * 4; hv[i] = f32::from_le_bytes(data[o..o+4].try_into().unwrap()); }
        Ok(hv)
    }

    /// BUG_006 NEW: lee N del header, devuelve Vec<f32> con ese tamaño.
    pub fn decode_hv_dynamic(data: &[u8]) -> Result<Vec<f32>, String> {
        if data.len() < 8 { return Err("frame demasiado corto".into()); }
        if &data[0..2] != &MAGIC { return Err("magic inválido".into()); }
        let frame_n = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        if data.len() < 8 + 4 * frame_n {
            return Err(format!("frame truncado: esperaba {} bytes para N={}", 8 + 4 * frame_n, frame_n));
        }
        let mut hv = Vec::with_capacity(frame_n);
        for i in 0..frame_n { let o = 8 + i * 4; hv.push(f32::from_le_bytes(data[o..o+4].try_into().unwrap())); }
        Ok(hv)
    }

    pub fn encode_native_sync(space: &mut Space, dims: &[&str]) -> Vec<u8> {
        let mut out = Vec::with_capacity(5 + dims.len() * (20 + 4 * N));
        out.extend_from_slice(&MAGIC); out.push(VERSION); out.push(FLAG_NATIVE_SYNC);
        out.push(dims.len() as u8);
        for &dim in dims {
            let hv = *space.sub(dim);
            let nb = dim.as_bytes(); out.push(nb.len() as u8); out.extend_from_slice(nb);
            for f in &hv { out.extend_from_slice(&f.to_le_bytes()); }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_normalize_unit_length() {
        let mut v = [0f32; N]; v[0] = 3.0; v[1] = 4.0;
        let nv = normalize(v);
        let norm: f32 = nv.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
    #[test] fn test_bind_commutative() {
        let a = make_hv(1); let b = make_hv(2);
        assert!(sim(&bind(&a, &b), &bind(&b, &a)) > 0.99);
    }
    #[test] fn test_proj_range() {
        let p = proj(&make_hv(42), &make_hv(99));
        assert!(p >= 0.0 && p <= 1.0);
    }
    #[test] fn test_space_deterministic() {
        let mut sp1 = Space::new("IA_TEST"); let mut sp2 = Space::new("IA_TEST");
        assert!(sim(sp1.sym("DIM_RUST"), sp2.sym("DIM_RUST")) > 0.9999);
    }
    #[test] fn test_pbp_roundtrip() {
        let hv = make_hv(777);
        assert!(sim(&hv, &pbp::decode_hv(&pbp::encode_hv(&hv)).unwrap()) > 0.9999);
    }
    #[test] fn test_native_sync_all_dims() {
        let mut src = Space::new("SRC"); let mut dst = Space::new("DST");
        let frame = pbp::encode_native_sync(&mut src, NATIVE_DIMS);
        dst.native_sync(&frame).unwrap();
        for &dim in NATIVE_DIMS { assert!(sim(src.sub(dim), dst.sub(dim)) > 0.999, "dim {} falla", dim); }
    }

    // V0.3: DIM_CONTRACT
    #[test] fn test_native_dims_count() {
        assert_eq!(NATIVE_DIMS.len(), 10);
        assert!(NATIVE_DIMS.contains(&"DIM_CONTRACT"));
    }
    #[test] fn test_dim_contract_activation() {
        let mut space = Space::new("TEST"); let mut obj = ObjectND::new();
        let mut props = HashMap::new(); props.insert("partes".into(), "2".into());
        obj.add("DIM_CONTRACT", props, 1.0);
        assert!(obj.activacion(&mut space, "DIM_CONTRACT") > UMBRAL);
    }

    // V0.3: BUG_005
    #[test] fn test_bug005_get_dims() {
        let mut obj = ObjectND::new();
        obj.add("DIM_SQL", HashMap::new(), 1.0);
        obj.add("DIM_PYTHON", HashMap::new(), 0.7);
        let dims = obj.get_dims();
        assert_eq!(dims.len(), 2);
        assert!(dims.contains(&"DIM_SQL") && dims.contains(&"DIM_PYTHON"));
    }
    #[test] fn test_bug005_get_dims_zero_excluded() {
        let mut obj = ObjectND::new();
        obj.add("DIM_SQL", HashMap::new(), 1.0);
        obj.add("DIM_PYTHON", HashMap::new(), 0.0);
        assert_eq!(obj.get_dims().len(), 1);
    }
    #[test] fn test_bug005_get_weight() {
        let mut obj = ObjectND::new(); obj.add("DIM_SQL", HashMap::new(), 0.8);
        assert_eq!(obj.get_weight("DIM_SQL"), Some(0.8));
        assert_eq!(obj.get_weight("DIM_PYTHON"), None);
    }
    #[test] fn test_bug005_set_weight() {
        let mut sp = Space::new("T"); let mut obj = ObjectND::new();
        obj.add("DIM_SQL", HashMap::new(), 1.0);
        let _ = obj.hv(&mut sp);
        assert!(obj.set_weight("DIM_SQL", 0.3).is_ok());
        assert!(obj.cache.is_none());
        assert_eq!(obj.get_weight("DIM_SQL"), Some(0.3));
    }
    #[test] fn test_bug005_set_weight_undeclared() {
        let mut obj = ObjectND::new(); obj.add("DIM_SQL", HashMap::new(), 1.0);
        assert!(obj.set_weight("DIM_PYTHON", 0.5).is_err());
    }
    #[test] fn test_bug005_invalidate_cache() {
        let mut sp = Space::new("T"); let mut obj = ObjectND::new();
        obj.add("DIM_GRAPH", HashMap::new(), 1.0);
        let _ = obj.hv(&mut sp); assert!(obj.cache.is_some());
        obj.invalidate_cache(); assert!(obj.cache.is_none());
    }

    // V0.3: BUG_006
    #[test] fn test_bug006_correct_n() {
        assert!(pbp::decode_hv(&pbp::encode_hv(&make_hv(42))).is_ok());
    }
    #[test] fn test_bug006_wrong_n_suggests_dynamic() {
        let mut frame = pbp::encode_hv(&make_hv(42));
        frame[4..8].copy_from_slice(&512u32.to_le_bytes());
        let err = pbp::decode_hv(&frame).unwrap_err();
        assert!(err.contains("decode_hv_dynamic"));
    }
    #[test] fn test_bug006_dynamic_any_n() {
        let small_n: usize = 128;
        let mut frame = vec![];
        frame.extend_from_slice(&pbp::MAGIC); frame.push(pbp::VERSION); frame.push(pbp::FLAG_NONE);
        frame.extend_from_slice(&(small_n as u32).to_le_bytes());
        for i in 0..small_n { frame.extend_from_slice(&((i as f32 / small_n as f32).to_le_bytes())); }
        let v = pbp::decode_hv_dynamic(&frame).unwrap();
        assert_eq!(v.len(), small_n);
    }
    #[test] fn test_bug006_dynamic_roundtrip() {
        let hv = make_hv(999);
        let v = pbp::decode_hv_dynamic(&pbp::encode_hv(&hv)).unwrap();
        assert_eq!(v.len(), N);
        assert!(v.iter().zip(hv.iter()).all(|(a, b)| (a - b).abs() < 1e-6));
    }
}
