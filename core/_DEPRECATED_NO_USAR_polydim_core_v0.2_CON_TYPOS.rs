// POLYDIM_DEST
// destination: polydim/core/
// filename:    polydim_core.rs
// author:      ai.mpat.agt@gmail.com

//! POLYDIM Core — Rust Prototype V0.2
//! ===================================
//! IMplementa Space, ObjectND y operaciones VSA básicas.
//! DIM_RUST como subespacio nativo.
//! Interop con Python via PBP V0 (TASK_021).
//!
//! CAMBIOS V0.2 (auditoria 2026-06-17, BUG_001 + BUG_006):
//!   BUG_001 (P0, critico): el make_hv() de V0.1 (LCG + Box-Muller propio)
//!   no replica PCG64 de NumPy. Mismo seed numerico, HVs distintos entre
//!   Python y Rust -> interop semantica rota en todos los subespacios
//!   nativos, aunque el frame PBP sea binariamente correcto.
//!   Fix aplicado = opcion A de la auditoria: Rust YA NO genera los
//!   subespacios nativos (NATIVE_DIMS) localmente con make_hv(). Python
//!   los genera una sola vez y los distribuye via NATIVE_SYNC durante el
//!   handshake; Rust los recibe y los cachea sin regenerarlos. make_hv()
//!   se conserva solo para vectores no-nativos (custom syms, geo random)
//!   donde la paridad exacta con NumPy no es un requisito de interop.
//!
//!   BUG_006 (P1): N estaba hardcodeado como `pub const N: usize = 10_000`
//!   y los hipervectores eran `[f32; N]` (array de tamaño fijo en tiempo
//!   de compilacion). Esto hacia el runtime Rust incompatible con
//!   cualquier Space Python que use N != 10000. PBP V0 ya transmite N en
//!   el header (N_u32_LE) — no usarlo era un error de diseño. Fix: N pasa
//!   a ser un campo de instancia en Space, los hipervectores son Vec<f32>
//!   en lugar de [f32; N], y decode_hv ya no rechaza frames con N distinto
//!   al hardcodeado.
//!
//! Compatibilidad conceptual con polydim_runtime_v03.py:
//!   float32, umbral = 0.5 + 2*(1/(2*sqrt(N)))  (N ahora es parametro)
//!
//! Autor:   ai.mpat.agt@gmail.com
//! Version: V0.2 — 2026-06-17
//! Task:    TASK_023 (fix BUG_001, BUG_006 de AUDITORIA_2026-06-17.md)

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────
// Constantes globales (ya NO incluyen N — ver BUG_006)
// ─────────────────────────────────────────────────────────────

const CONTENT_W: f32 = 0.3;

/// Umbral de activacion para un N dado. Antes era const fijo para N=10000;
/// ahora se calcula por instancia de Space porque N es dinamico (BUG_006).
pub fn umbral_for(n: usize) -> f32 {
    0.5 + 2.0 * (1.0 / (2.0 * (n as f32).sqrt()))
}

pub const NATIVE_DIMS: &[&str] = &[
    "DIM_PYTHON", "DIM_RUST", "DIM_FLUTTER", "DIM_SQL",
    "DIM_GRAPH",  "DIM_VECTOR", "DIM_TIME", "DIM_ERROR", "DIM_META",
];

// ─────────────────────────────────────────────────────────────
// Operaciones VSA básicas — ahora sobre Vec<f32> (BUG_006: N dinamico)
// ─────────────────────────────────────────────────────────────

/// Binding: multiplicación elemento a elemento + normalización.
/// Equivalente a _bind(a, b) en Python.
/// Precondicion: a.len() == b.len() (mismo N). Si no coinciden, panic
/// explicito en vez de comportamiento indefinido — fuerza al caller a
/// resolver el mismatch de N antes de operar (ver PEND_010 en backlog).
pub fn bind(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "bind: dimensiones N distintas ({} vs {})", a.len(), b.len());
    let r: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
    normalize(r)
}

/// Superposición ponderada: suma con pesos + normalización.
/// Equivalente a _sup(*hvs, ws=ws) en Python.
pub fn sup(hvs: &[(&[f32], f32)]) -> Vec<f32> {
    let n = hvs.first().map(|(hv, _)| hv.len()).unwrap_or(0);
    let mut acc = vec![0f32; n];
    for (hv, w) in hvs {
        assert_eq!(hv.len(), n, "sup: todos los hipervectores deben tener el mismo N");
        for i in 0..n {
            acc[i] += w * hv[i];
        }
    }
    normalize(acc)
}

/// Proyección: similitud coseno mapeada a [0, 1].
/// Equivalente a _proj(hv, sub) en Python.
pub fn proj(hv: &[f32], sub: &[f32]) -> f32 {
    let dot: f32 = hv.iter().zip(sub.iter()).map(|(a, b)| a * b).sum();
    (dot + 1.0) / 2.0
}

/// Similitud coseno entre dos hipervectores → [0, 1].
pub fn sim(a: &[f32], b: &[f32]) -> f32 {
    proj(a, b)
}

/// Normalización L2. Si norma ≈ 0, devuelve el vector sin cambio.
pub fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() { *x /= norm; }
    }
    v
}

// ─────────────────────────────────────────────────────────────
// Generador determinístico de hipervectores (LCG + Box-Muller)
// ─────────────────────────────────────────────────────────────
// BUG_001: este generador NO replica PCG64 de NumPy. Se conserva
// unicamente para vectores donde la paridad exacta con Python no es
// requisito (custom syms locales, geo random). Los subespacios NATIVOS
// (NATIVE_DIMS) ya NO se generan con esta funcion — ver NATIVE_SYNC mas
// abajo y Space::new().

/// Genera un hipervector float32 determinístico de dimension `n` a partir
/// de un seed. Usa LCG + Box-Muller (aproximacion de una normal estandar,
/// NO equivalente a np.random.default_rng(seed).standard_normal(n) — ver
/// nota BUG_001 arriba). Adecuado para vectores sin requisito de paridad
/// exacta con Python.
pub fn make_hv(seed: u64, n: usize) -> Vec<f32> {
    let mut hv = vec![0f32; n];
    let mut state = seed;

    let lcg_next = |s: u64| -> u64 {
        s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
    };

    let mut i = 0;
    while i < n {
        state = lcg_next(state);
        let u1 = (state >> 11) as f32 / (1u64 << 53) as f32;
        state = lcg_next(state);
        let u2 = (state >> 11) as f32 / (1u64 << 53) as f32;

        // Box-Muller
        let r = (-2.0 * (u1 + f32::EPSILON).ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        hv[i] = r * theta.cos();
        if i + 1 < n {
            hv[i + 1] = r * theta.sin();
        }
        i += 2;
    }
    normalize(hv)
}

/// Deriva un seed u64 desde un string (equivalente a md5 → int % 2^32 en Python).
pub fn seed_from_str(s: &str) -> u64 {
    // FNV-1a 64-bit — determinístico, sin dependencias externas
    let mut hash: u64 = 14695981039346656037;
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

// ─────────────────────────────────────────────────────────────
// Space
// ─────────────────────────────────────────────────────────────

/// Espacio POLYDIM: cachea hipervectores de subespacio.
/// N ahora es campo de instancia (BUG_006), no const global.
///
/// IMPORTANTE (BUG_001): los subespacios NATIVOS (NATIVE_DIMS) no se
/// generan localmente. Space::new() los deja vacios; deben llegar via
/// sync_native() con los HVs exactos que Python genero con PCG64 y
/// transmitio por PBP/NATIVE_SYNC durante el handshake. Si se intenta
/// leer un subespacio nativo antes de sincronizar, sub() entra en panic
/// con un mensaje explicito en vez de generar localmente un HV
/// incompatible en silencio ((que era el bug original).
pub struct Space {
    pub n: usize,
    pub personal_seed: String,
    native_synced: bool,
    sym_cache:  HashMap<String, Vec<f32>>,
    sub_cache:  HashMap<String, Vec<f32>>,
}

impl Space {
    /// Crea un Space vacio de subespacios nativos. Llamar a sync_native()
    /// con los HVs recibidos de Python antes de usar cualquier NATIVE_DIMS.
    pub fn new(personal_seed: &str, n: usize) -> Self {
        Space {
            n,
            personal_seed: personal_seed.to_string(),
            native_synced: false,
            sym_cache: HashMap::new(),
            sub_cache: HashMap::new(),
        }
    }

    /// NATIVE_SYNC (fix BUG_001): recibe los subespacios nativos generados
    /// por Python (PCG64) y los instala directamente en sub_cache, sin
    /// regenerarlos localmente. `native_hvs` debe tener una entrada por
    /// cada nombre en NATIVE_DIMS, en el mismo orden, con vectores de
    /// longitud `self.n`.
    pub fn sync_native(&mut self, native_hvs: &[Vec<f32>]) {
        assert_eq!(native_hvs.len(), NATIVE_DIMS.len(),
            "sync_native: se esperaban {} subespacios nativos, llegaron {}",
            NATIVE_DIMS.len(), native_hvs.len());
        for (name, hv) in NATIVE_DIMS.iter().zip(native_hvs.iter()) {
            assert_eq!(hv.len(), self.n,
                "sync_native: HV de '{}' tiene N={}, Space espera N={}", name, hv.len(), self.n);
            self.sub_cache.insert(name.to_string(), hv.clone());
            self.sym_cache.insert(name.to_string(), hv.clone());
        }
        self.native_synced = true;
    }

    pub fn is_native_synced(&self) -> bool { self.native_synced }

    /// Genera hipervector para un nombre simbólico custom (con personal_seed).
    /// No usar para NATIVE_DIMS — esos deben llegar via sync_native().
    pub fn sym(&mut self, name: &str) -> &[f32] {
        if !self.sym_cache.contains_key(name) {
            assert!(!NATIVE_DIMS.contains(&name),
                "sym('{}'): los subespacios nativos debuen llegar via sync_native(), no generarse localmente (BUG_001)", name);
            let key = if self.personal_seed.is_empty() {
                name.to_string()
            } else {
                format!("{}:{}", self.personal_seed, name)
            };
            let seed = seed_from_str(&key);
            let hv = make_hv(seed, self.n);
            self.sym_cache.insert(name.to_string(), hv);
        }
        &self.sym_cache[name]
    }

    /// Subespacio nativo o custom (igual que sym, pero cache separado).
    /// Para NATIVE_DIMS, panic si todavia no se llamo sync_native().
    pub fn sub(&mut self, name: &str) -> &[f32] {
        if !self.sub_cache.contains_key(name) {
            if NATIVE_DIMS.contains(&name) {
                panic!("sub('{}'): subespacio nativo sin sincronizar. Llamar sync_native() primero (BUG_001 fix).", name);
            }
            let hv = self.sym(name).to_vec();
            self.sub_cache.insert(name.to_string(), hv);
        }
        &self.sub_cache[name]
    }

    /// Vector aleatorio (para geo_id de ObjectND). No requiere paridad con
    /// Python: cada IA genera su propia identidad geometrica localmente.
    pub fn random_hv(n: usize) -> Vec<f32> {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64 ^ d.as_secs().wrapping_mul(0xdeadbeef))
            .unwrap_or(42);
        make_hv(seed, n)
    }

    /// Codifica un mapa de propiedades como hipervector.
    /// Equivalente a Space._enc(p) en Python.
    pub fn enc(&mut self, props: &HashMap<String, String>) -> Vec<f32> {
        if props.is_empty() {
            return self.sym("__empty__").to_vec();
        }
        let pairs: Vec<(Vec<f32>, f32)> = props.iter().map(|(k, v)| {
            let sk = self.sym(k).to_vec();
            let sv = self.sym(v).to_vec();
            (bind(&sk, &sv), 1.0)
        }).collect();
        let refs: Vec<(&[f32], f32)> = pairs.iter().map(|(hv, w)| (hv.as_slice(), *w)).collect();
        sup(&refs)
    }
}

// ─────────────────────────────────────────────────────────────
// ObjectND
// ─────────────────────────────────────────────────────────────

/// Objeto n-dimensional POLYDIM en Rust.
/// Equivalente a ObjectND en Python.
pub struct ObjectND {
    pub geo: Vec<f32>,
    pub dims: HashMap<String, (HashMap<String, String>, f32)>, // dim → (props, weight)
    cache: Option<Vec<f32>>,\~
}

impl ObjectND {
    pub fn new(n: usize) -> Self {
        ObjectND {
            geo: Space::random_hv(n),
            dims: HashMap::new(),
            cache: None,
        }
    }

    /// Construye un ObjectND a partir de un geo vector ya existente
    /// (ej. recibido de Python via PBP) en lugar de generar uno random.
    /// Equivalente al fix de BUG_002 en el runtime Python (set_geo).
    pub fn from_geo(geo: Vec<f32>) -> Self {
        ObjectND { geo, dims: HashMap::new(), cache: None }
    }

    /// Agrega una dimensión con propiedades y peso.
    pub fn add(&mut self, dim: &str, props: HashMap<String, String>, w: f32) -> &mut Self {
        let w = w.clamp(0.0, 1.0);
        self.dims.insert(dim.to_string(), (props, w));
        self.cache = None;
        self
    }

    /// geo_id: primeros 12 hex del hash FNV del hipervector geométrico.
    pub fn geo_id(&self) -> String {
        let preview_len = self.geo.len().min(8);
        let seed = seed_from_str(&format!("{:?}", &self.geo[..preview_len]));
        format!({:012x}", seed & 0xffffffffffff)
    }

    /// Hipervector compuesto (MEMBERSHIP + CONTENT layers).
    /// Equivalente a ObjectND._hv() en Python.
    pub fn hv(&mut self, space: &mut Space) -> Vec<f32> {
        if let Some(c) = &self.cache {
            return c.clone();
        }

        let mut components: Vec<(Vec<f32>, f32)> = vec![(self.geo.clone(), 1.0)];

        for (dim, (props, w)) in &self.dims {
            if *w <= 0.0 { continue; }
            let sub_hv = space.sub(dim).to_vec();
            // Capa MEMBERSHIP
            components.push((sub_hv.clone(), *w));
            // Capa CONTENT
            let enc_hv = space.enc(props);
            let content_hv = bind(&sub_hv, &enc_hv);
            components.push((content_hv, w * CONTENT_W));
        }

        let refs: Vec<(&[f32], f32)> = components.iter().map(|(hv, w)| (hv.as_slice(), *w)).collect();
        let result = sup(&refs);
        self.cache = Some(result.clone());
        result
    }

    /// Activación de una dimensión → [0, 1].
    pub fn activacion(&mut self, space: &mut Space, dim: &str) -> f32 {
        let hv = self.hv(space);
        let sub = space.sub(dim).to_vec();
        proj(&hv, &sub)
    }

    /// Dimensiones activas por encima del umbral.
    pub fn dims_activas(&mut self, space: &mut Space) -> HashMap<String, f32> {
        let mut result = HashMap::new();
        let umbral = umbral_for(space.n);
        let all_dims: Vec<String> = NATIVE_DIMS.iter().map(|s| s.to_string())
            .chain(self.dims.keys().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter().collect();

        for dim in all_dims {
            // Si el subespacio nativo no esta sincronizado todavia, lo
            // saltea en vez de hacer panic — dims_activas() debe poder
            // usarse antes del NATIVE_SYNC para inspeccionar solo dims custom.
            if NATIVE_DIMS.contains(&dim.as_str()) && !space.is_native_synced() {
                continue;
            }
            let a = self.activacion(space, &dim);
            if a > umbral {
                result.insert(dim, (a * 10000.0).round() / 10000.0);
            }
        }
        result
    }
}

// ─────────────────────────────────────────────────────────────
// PBP V0 — Interop con Python (TASK_021)
// ─────────────────────────────────────────────────────────────
// BUG_006 fix: decode_hv ya no rechaza frames con N != 10000. El N del
// frame (header N_u32_LE) es la fuente de verdad; el Vec<f32> resultante
// tiene esa longitud, sea la que sea.

pub mod pbp {
    use super::{ObjectND, Space};

    const MAGIC: [u8; 2] = [0x50, 0x44]; // "PD"
    const VERSION: u8 = 0x00;

    /// Serializa hipervector a bytes PBP V0 (solo hipervector, flags=0x00).
    pub fn encode_hv(hv: &[f32]) -> Vec<u8> {
        let n = hv.len();
        let mut out = Vec::with_capacity(8 + 4 * n);
        out.extend_from_slice(&MAGIC);
        out.push(VERSION);
        out.push(0x00); // flags
        out.extend_from_slice(&(n as u32).to_le_bytes());
        for f in hv {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out
    }

    /// Serializa ObjectND a PBP V0 con geo_id y pesos (flags=0x03).
    pub fn encode_object(obj: &mut ObjectND, space: &mut Space) -> Vec<u8> {
        let hv = obj.hv(space);
        let n = hv.len();
        let geo = obj.geo_id();
        let geo_b = geo.as_bytes();

        let mut weight_bytes: Vec<u8> = vec![obj.dims.len() as u8];
        for (dim, (_, w)) in &obj.dims {
            let dn = dim.as_bytes();
            weight_bytes.push(dn.len() as u8);
            weight_bytes.extend_from_slice(dn);
            weight_bytes.extend_from_slice(&w.to_le_bytes());
        }

        let flags: u8 = 0x03; // HAS_GEO_ID | HAS_WEIGHTS
        let mut out = Vec::with_capacity(8 + 4 * n + 1 + geo_b.len() + weight_bytes.len());
        out.extend_from_slice(&MAGIC);
        out.push(VERSION);
        out.push(flags);
        out.extend_from_slice(&(n as u32).to_le_bytes());
        for f in &hv {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out.push(geo_b.len() as u8);
        out.extend_from_slice(geo_b);
        out.extend_from_slice(&weight_bytes);
        out
    }

    /// Deserializa hipervector desde bytes PBP V0.
    /// BUG_006 fix: ya no valida n == N_CONST. Acepta cualquier N del header.
    pub fn decode_hv(data: &[u8]) -> Result<Vec<f32>, &'static str> {
        if data.len() < 8 { return Err("frame demasiado corto"); }
        if &data[0..2] != &MAGIC { return Err("magic invalido"); }
        let n = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        if data.len() < 8 + 4 * n { return Err("frame truncado"); }
        let hv: Vec<f32> = data[8..8 + 4 * n]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Ok(hv)
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_N: usize = 10_000;

    /// Helper de test: simula NATIVE_SYNC con vectores deterministicos
    /// locales (no PCG64 real — eso requeriria FFI a NumPy, fuera de
    /// alcance de este test unitario). Sirve para probar la mecanica de
    /// sync_native(), no la paridad numerica exacta con Python (esa
    /// verificacion es manual, ver checklist en AUDITORIA_2026-06-17.md).
    fn fake_native_sync(sp: &mut Space) {
        let hvs: Vec<Vec<f32>> = NATIVE_DIMS.iter()
            .enumerate()
            .map(|(i, _)| make_hv(1000 + i as u64, sp.n))
            .collect();
        sp.sync_native(&hvs);
    }

    #[test]
    fn test_normalize_unit_length() {
        let mut v = vec![0f32; TEST_N];
        v[0] = 3.0; v[1] = 4.0;
        let nv = normalize(v);
        let norm: f32 = nv.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norma={}", norm);
    }

    #[test]
    fn test_bind_commutative() {
        let a = make_hv(1, TEST_N);
        let b = make_hv(2, TEST_N);
        let ab = bind(&a, &b);
        let ba = bind(&b, &a);
        let s = sim(&ab, &ba);
        assert!(s > 0.99, "bind no conmutativo: sim={}", s);
    }

    #[test]
    fn test_proj_range() {
        let a = make_hv(42, TEST_N);
        let b = make_hv(99, TEST_N);
        let p = proj(&a, &b);
        assert!(p >= 0.0 && p <= 1.0, "proj fuera de [0,1]: {}", p);
    }

    #[test]
    fn test_space_deterministic_custom_sym() {
        // BUG_001 fix: este test ya no usa DIM_RUST (nativo, requiere sync).
        // Verifica determinismo de sym() para dims CUSTOM, que es lo que
        // make_hv local sigue garantizando.
        let mut sp1 = Space::new("IA_TEST", TEST_N);
        let mut sp2 = Space::new("IA_TEST", TEST_N);
        let h1 = sp1.sym("MI_DIM_CUSTOM").to_vec();
        let h2 = sp2.sym("MI_DIM_CUSTOM").to_vec();
        let s = sim(&h1, &h2);
        assert!(s > 0.9999, "Space no determinístico: sim={}", s);
    }

    #[test]
    fn test_space_different_seeds_custom_sym() {
        let mut sp1 = Space::new("IA_A", TEST_N);
        let mut sp2 = Space::new("IA_B", TEST_N);
        let h1 = sp1.sym("MI_DIM_CUSTOM").to_vec();
        let h2 = sp2.sym("MI_DIM_CUSTOM").to_vec();
        let s = sim(&h1, &h2);
        assert!(s < 0.6, "Spaces distintos demasiado similares: sim={}", s);
    }

    #[test]
    #[should_panic(expected = "subespacio nativo sin sincronizar")]
    fn test_native_dim_without_sync_panics() {
        // BUG_001 fix: leer un subespacio nativo antes de sync_native()
        // debe fallar explicitamente, no generar un HV incompatible en silencio.
        let mut sp = Space::new("TEST", TEST_N);
        sp.sub("DIM_SQL");
    }

    #[test]
    fn test_native_sync_installs_vectors() {
        let mut sp = Space::new("TEST", TEST_N);
        assert!(!sp.is_native_synced());
        fake_native_sync(&mut sp);
        assert!(sp.is_native_synced());
        let hv = sp.sub("DIM_SQL").to_vec();
        assert_eq!(hv.len(), TEST_N);
    }

    #[test]
    fn test_object_nd_activation_after_sync() {
        let mut space = Space::new("TEST", TEST_N);
        fake_native_sync(&mut space);
        let mut obj = ObjectND::new(TEST_N);
        let mut props = std::collections::HashMap::new();
        props.insert("tabla".to_string(), "usuarios".to_string());
        obj.add("DIM_SQL", props, 1.0);

        let act = obj.activacion(&mut space, "DIM_SQL");
        assert!(act > umbral_for(TEST_N), "DIM_SQL debería estar activa: act={}", act);
    }

    #[test]
    fn test_pbp_roundtrip() {
        let hv = make_hv(777, TEST_N);
        let bytes = pbp::encode_hv(&hv);
        let hv2 = pbp::decode_hv(&bytes).unwrap();
        let s = sim(&hv, &hv2);
        assert!(s > 0.9999,"PBP roundtrip perdi� precisiòn: sim={}", s);
    }

    #[test]
    fn test_pbp_magic_check() {
        let bad = vec![0u8; 50];
        assert!(pbp::decode_hv(&bad).is_err());
    }

    #[test]
    fn test_pbp_accepts_different_n() {
        // BUG_006 fix: antes decode_hv rechazaba cualquier N != 10000.
        // Ahora N=500 (u otro distinto) debe funcionar sin error.
        let hv = make_hv(123, 500);
        let bytes = pbp::encode_hv(&hv);
        let decoded = pbp::decode_hv(&bytes).unwrap();
        assert_eq!(decoded.len(), 500, "N=500 deberia preservarse, no rechazarse");
        let s = sim(&hv, &decoded);
        assert!(s > 0.9999, "roundtrip con N custom perdió precisión: sim={}", s);
    }

    #[test]
    fn test_custom_dims_orthogonality() {
        // BUG_001: ya no se puede testear ortogonalidad de NATIVE_DIMS
        // localmente (requieren sync con Python). Se testea con dims
        // custom generadas localmente, que es lo que make_hv garantiza.
        let mut sp = Space::new("", TEST_N);
        let custom = ["CUSTOM_A", "CUSTOM_B", "CUSTOM_C"];
        for i in 0..custom.len() {
            for j in (i+1)..custom.len() {
                let hi = sp.sub(custom[i]).to_vec();
                let hj = sp.sub(custom[j]).to_vec();
                let s = sim(&hi, &hj);
                assert!(s > 0.40 && s < 0.60,
                    "sims[{}][{}]={:.4} fuera de rango esperado",
                    custom[i], custom[j], s);
            }
        }
    }
}
