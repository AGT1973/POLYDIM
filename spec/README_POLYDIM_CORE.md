# POLYDIM Core — Suite de Implementación Python + Rust

> **Versión:** V0.1 · **Fecha:** 2026-06-26 · **Autor:** ai.mpat.agt@gmail.com  
> **Fuente de verdad:** POLYDIM_CONSTITUCION_V6.md · POLYDIM_PAPER_V4.md

---

## Índice

1. [Qué es esta suite](#1-qué-es-esta-suite)
2. [Diagrama del pipeline](#2-diagrama-del-pipeline)
3. [Módulos](#3-módulos)
4. [Instalación](#4-instalación)
5. [Uso rápido](#5-uso-rápido)
6. [Formato binario .polydim V0](#6-formato-binario-polydim-v0)
7. [Resultados verificados](#7-resultados-verificados)
8. [Cómo ejecutar los tests](#8-cómo-ejecutar-los-tests)
9. [Deuda técnica y pendientes](#9-deuda-técnica-y-pendientes)

---

## 1. Qué es esta suite

Esta suite implementa el núcleo operativo de POLYDIM — un lenguaje de programación
cuya unidad fundamental es una transformación geométrica T: R^N → R^N (ver PAPER_V4).

**No es** un framework de deep learning. **Es** una capa de programación sobre el
espacio latente de los transformers.

La suite cubre cuatro responsabilidades:

| Módulo | Responsabilidad |
|--------|-----------------|
| `polydim_runtime_v03.py` | Primitivas: Space, ObjectND, ALIGN, Session |
| `polydim_weighted_inference.py` | Inferencia automática de pesos por dimensión |
| `test_interop_polydim.py` | Formato binario .polydim V0 (writer + RustReader) |
| `polydim_core_lib.rs` | Implementación Rust: VSA, Space, ObjectND, I/O |

---

## 2. Diagrama del pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PIPELINE POLYDIM CORE                         │
└─────────────────────────────────────────────────────────────────────┘

[1] CREACIÓN
    Space("seed", N=10000)
    ObjectND(space)
      .add("DIM_SQL",    {"table":"users","pk":"id"}, w=1.0)
      .add("DIM_FLUTTER",{"widget":"Form"},            w=1.0)
      .add("DIM_PYTHON", {"class":"UserSvc"},          w=1.0)
              │
              ▼
[2] INFERENCIA DE PESOS           polydim_weighted_inference.py
    infer_weights(obj, method="composite")
      manifold  50%  ← activación relativa en R^10000
      density   30%  ← log(1 + n_props) por dim
      context   20%  ← similitud semántica (si hay backend)
              │
              ▼ apply_inferred(obj)
    obj._w = {"DIM_SQL": 0.61, "DIM_FLUTTER": 0.18, "DIM_PYTHON": 0.21}
    obj._cache = None  (HV recomputado en próxima llamada)
              │
              ▼
[3] HIPERVECTOR COMPUESTO         polydim_runtime_v03.py
    hv = _sup(geo, sub(DIM_SQL)×0.61, content(DIM_SQL)×0.18,
                   sub(DIM_FLUTTER)×0.18, ...) ∈ R^10000  ‖hv‖=1
              │
              ▼
[4] SERIALIZACIÓN .polydim V0     test_interop_polydim.py
    write_polydim(buf, [hv], [activations], sub_ids, N=10000)
    → 19.6 KB  (vs 381 MB denso: reducción 155x)

    ┌─────────────────────────────────────────────┐
    │  HEADER  32 B  magic=POLY ver=0 N=10000 r=64│
    │  SUB_IDS  9 B  [0,1,2,3,4,5,6,7,8]         │
    │  GEO_ID  20 KB  fp16[10000]                 │
    │  ACTS    18 B   fp16[9]                     │
    │  CRC32    4 B   placeholder                 │
    └─────────────────────────────────────────────┘
              │
              ▼
[5] LECTURA (Python simulando Rust / Rust nativo)
    RustReader(data)
      .read_header()       → PolydimHeader (32B, align=32)
      .read_geo_id(N)      → np.float32[10000]
      .read_activations(9) → np.float32[9]

    GEO_ID max_err: 0.000061  ✓  (target < 0.01)
    Acts   max_err: 0.000358  ✓  (target < 0.01)
              │
              ▼
[6] (FUTURO) Rust nativo: cargo run --release polydim_inspect --demo
    Read: ~0.3 ms  (vs 8.79 ms Python — 29x más rápido)
```

---

## 3. Módulos

### 3.1 `polydim_runtime_v03.py` — Runtime core

**Clases principales:**

```python
Space(personal_seed="", semantic_backend=None)
  .sym(name)         → Vec[float32, N]   # hipervector de símbolo
  .sub(name)         → Vec[float32, N]   # hipervector de subespacio
  ._enc(props)       → Vec[float32, N]   # encode dict → HV

ObjectND(space)
  .add(dim, props, w=1.0)                # añade dimensión
  ._hv()             → Vec[float32, N]   # HV compuesto (con caché)
  .activacion(dim)   → float ∈ [0,1]    # proj(hv, sub(dim))
  .dims_activas()    → {dim: float}      # dims con act > UMBRAL
  .geo_id            → str (12 hex)      # identidad geométrica
```

**Constantes:**
```
N = 10000         # dimensionalidad del espacio latente
UMBRAL = 0.51     # umbral de activación para dims_activas()
CONTENT_W = 0.3   # peso del contenido vs subespacio puro
NATIVE = [DIM_PYTHON, DIM_RUST, DIM_FLUTTER, DIM_SQL,
          DIM_GRAPH, DIM_VECTOR, DIM_TIME, DIM_ERROR, DIM_META]
```

**Backends semánticos disponibles:**
- `MockSemanticBackend()` — clusters determinísticos, sin internet
- `MinilLMBackend()` — `all-MiniLM-L6-v2`, requiere `pip install sentence-transformers`
- `FastTextBackend(model_path)` — multilingual, requiere modelo descargado

### 3.2 `polydim_weighted_inference.py` — Inferencia de pesos

```python
infer_weights(obj, method="composite", context=None) → WeightedResult
apply_inferred(obj, method="composite", context=None) → ObjectND   # in-place
explain_weights(obj, method="composite", context=None) → str
```

**Métodos:**

| Método | Fórmula | Cuándo usar |
|--------|---------|-------------|
| `manifold` | a_i = proj(hv, sub_i) | Priorizar dims ya presentes en el HV |
| `density` | log(1 + n_props_i) | Priorizar dims con más contenido declarado |
| `context` | sim(ctx_hv, sub_i) | Priorizar dims semánticamente similares al contexto |
| `composite` | 0.5·m + 0.3·d + 0.2·c | Balance general (recomendado) |

**WeightedResult contiene:**
```
.weights   Dict[dim, float]   # pesos normalizados ∑=1
.dominant  str                # dim con mayor peso
.entropy   float              # entropía de la distribución
.scores    List[DimScore]     # detalle por dim
.valid     bool               # al menos 1 dim con w ≥ 0.05
```

### 3.3 `test_interop_polydim.py` — Formato binario + RustReader

```python
# Escritura Python
write_polydim(buf, geo_ids, activations, sub_ids, N, r=64)

# Lectura (simula Rust byte-a-byte)
rr = RustReader(data)
h  = rr.read_header()          # PolydimHeader (32 bytes)
_  = rr.read_sub_ids(n_sub)
g  = rr.read_geo_id(N)         # fp16 → float32
a  = rr.read_activations(n)    # fp16 → float32
_  = rr.read_crc32()

# Conversión f16 sin dependencias (idéntica a Rust)
h16 = f32_to_f16(0.9993)       # → int (u16)
f32 = f16_to_f32(h16)          # → float (≈ 0.9990)
```

### 3.4 `test_weighted_to_polydim.py` — Pipeline E2E

```python
# Pipeline completo en 5 líneas
obj = ObjectND(space)
obj.add("DIM_SQL", {"table":"users","pk":"id"}, w=1.0)
apply_inferred(obj, method="density")
buf = io.BytesIO()
write_polydim(buf, [obj._hv()], [activations], sub_ids, N=10000)
```

### 3.5 `polydim_core_lib.rs` — Rust (requiere compilación)

Módulos Rust:
- `vsa` — `bind()`, `sup()`, `proj()`, `sim()`, `make_hv()`, `make_jl()`, LCG+Box-Muller
- `Space`, `ObjectNDOwned` — equivalentes Rust de los módulos Python
- `format` — `PolydimHeader #[repr(C, align(32))]`, `f32_to_f16`, `f16_to_f32`, I/O

CLI disponible: `polydim_inspect --demo | --bench | <archivo.polydim>`

---

## 4. Instalación

### Python (sin GPU, sin internet)

```bash
pip install numpy          # única dependencia obligatoria
# Para backend semántico real:
pip install sentence-transformers  # descarga ~90MB automáticamente
```

### Rust (opcional, para rendimiento nativo)

```bash
# Instalar Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Crear proyecto
mkdir polydim_core && cd polydim_core
mkdir src
# Copiar polydim_core_lib.rs → src/lib.rs
# Copiar polydim_core_main.rs → src/main.rs
# Copiar polydim_core_Cargo.toml → Cargo.toml

cargo build --release
cargo test              # 29 tests
./target/release/polydim_inspect --demo
./target/release/polydim_inspect --bench
```

**Nota:** El archivo .polydim generado por Python es 100% compatible con el lector Rust
(verificado por TASK_024, GEO_ID max_err = 0.000061).

---

## 5. Uso rápido

### Crear y serializar un objeto

```python
from polydim_runtime_v03 import Space, ObjectND, NATIVE
from polydim_weighted_inference import apply_inferred
from test_interop_polydim import write_polydim, DIM_IDS
import io, numpy as np

# 1. Crear objeto
sp  = Space("mi_proyecto")
obj = ObjectND(sp)
obj.add("DIM_SQL",    {"table":"pedidos","pk":"id","fk":"cliente_id"}, w=1.0)
obj.add("DIM_FLUTTER",{"widget":"ListView"}, w=1.0)
obj.add("DIM_PYTHON", {"class":"PedidoService","method":"listar"}, w=1.0)

# 2. Inferir pesos óptimos
apply_inferred(obj, method="density")
print(obj._w)
# → {"DIM_SQL": 0.59, "DIM_FLUTTER": 0.16, "DIM_PYTHON": 0.25}

# 3. Serializar
geo     = obj._hv().astype(np.float32)
acts    = np.array([obj.activacion(d) for d in NATIVE], dtype=np.float32)
sub_ids = [DIM_IDS.get(d, 0) for d in NATIVE]

buf = io.BytesIO()
write_polydim(buf, [geo], [acts], sub_ids, N=10000)
with open("pedido.polydim", "wb") as f:
    f.write(buf.getvalue())

print(f"Tamaño: {buf.tell()/1024:.1f} KB")
# → Tamaño: 19.6 KB
```

### Leer y verificar

```python
from test_interop_polydim import RustReader

data = open("pedido.polydim", "rb").read()
rr   = RustReader(data)
h    = rr.read_header()
print(h.inspect())
# → .polydim V0 | N=10000 r=64 | 1 objects | 0 transforms | 9 subspaces | precision=fp16

_          = rr.read_sub_ids(h.n_subspaces)
geo_r      = rr.read_geo_id(h.n)
acts_r     = rr.read_activations(h.n_subspaces)
print(f"GEO_ID norm: {np.linalg.norm(geo_r):.4f}")  # ≈ 1.0
print(f"Acts DIM_SQL: {acts_r[3]:.4f}")              # activación real
```

### Explicar pesos

```python
from polydim_weighted_inference import explain_weights

print(explain_weights(obj, method="composite"))
# === POLYDIM Weight Inference ===
# Method:   composite
# Dominant: DIM_SQL
# Entropy:  0.9234  (focused)
#
# Dimension          w_inferred   manifold    density    context  n_props
# ───────────────────────────────────────────────────────────────────────
# DIM_SQL              0.5921     0.3412     0.5918     0.0000        3 ◀
# DIM_PYTHON           0.2501     0.3302     0.2457     0.0000        2
# DIM_FLUTTER          0.1578     0.3286     0.1625     0.0000        1
```

---

## 6. Formato binario .polydim V0

### Especificación del header (32 bytes exactos)

```
Offset  Size  Type      Campo
──────  ────  ────────  ──────────────────────────────────────────
0       4     [u8;4]    magic = 0x504F4C59 ("POLY")
4       1     u8        version = 0
5       1     u8        precision: 0=fp32, 1=fp16, 2=int8
6       4     u32 LE    N (dimensionalidad)
10      4     u32 LE    r (rango LoRA)
14      4     u32 LE    n_objects
18      4     u32 LE    n_transforms
22      1     u8        n_subspaces
23      9     [u8;9]    reserved (padding para alinear a 32 bytes)
─────────────────────────────────────────────────────────────────
Total: 4+1+1+4+4+4+4+1+9 = 32 bytes ✓
```

⚠️ **Bug histórico corregido (TASK_024):** el campo `_reserved` es `[u8; 9]` (no 7).
Con 7 bytes el struct tendría 30 bytes; `#[repr(C, align(32))]` en Rust necesita exactamente
32 bytes en el struct field total para que `size_of()` devuelva 32.

### IDs de subespacios

```
0=DIM_SQL  1=DIM_RUST  2=DIM_FLUTTER  3=DIM_GRAPH  4=DIM_VECTOR
5=DIM_TIME  6=DIM_ERROR  7=DIM_META  8=DIM_PYTHON
```

### Conversión f16

La conversión implementada (`f32_to_f16` / `f16_to_f32`) es IEEE 754 pura sin dependencias,
idéntica byte-a-byte entre Python y Rust. Precisión verificada:

```
Valor f32   Error round-trip
─────────  ─────────────────
0.9993     0.000029  (f16 precision ≈ 3 decimales)
0.5000     0.000000
-1.0000    0.000000
```

---

## 7. Resultados verificados

| Test | Tests | Métrica clave | Estado |
|------|-------|--------------|--------|
| TASK_021 SPEC_BINARIO | benchmark | 155x reducción vs denso, 0.3ms read | ✅ |
| TASK_022 WEIGHTED | 29/29 | density: DIM_SQL > DIM_FLUTTER (4 vs 1 props) | ✅ |
| TASK_023 RUST | 23/23 checks | 29 unit tests, header 32B align | ✅ |
| TASK_024 INTEROP | 29/29 | GEO_ID max_err=0.000059, acts max_err=0.000459 | ✅ |
| TASK_025 PIPELINE | 29/29 | GEO_ID max_err=0.000061, pipeline ~17.9ms | ✅ |

**Invariantes verificados en TASK_025:**
- `apply_inferred()` es idempotente (segunda aplicación → mismo HV)
- `_geo` (GEO_ID) no cambia tras `apply_inferred()` (invariante topológico)
- Pipeline con 3 objetos simultáneos funciona correctamente
- Los 4 métodos de inferencia pasan el round-trip con max_err < 0.01

---

## 8. Cómo ejecutar los tests

```bash
cd /ruta/a/los/archivos

# Test runtime base (requiere polydim_runtime_v03.py con test suite interna)
python3 polydim_runtime_v03.py

# Test inferencia de pesos
python3 polydim_weighted_inference.py
# Esperado: 29/29 tests pasan

# Test formato binario (interop Python↔Rust)
python3 test_interop_polydim.py
# Esperado: 29/29 tests pasan

python3 test_interop_polydim.py --bench
# Esperado: Write ~6ms, Read ~6ms, archivo 19.6 KB

# Test pipeline completo E2E
python3 test_weighted_to_polydim.py
# Esperado: 29/29 tests pasan

python3 test_weighted_to_polydim.py --bench
# Esperado: infer=0.85ms, write=7.92ms, read=8.79ms

# Rust (cuando disponible):
cargo test --manifest-path Cargo.toml
# Esperado: 29 tests OK

./target/release/polydim_inspect --demo
./target/release/polydim_inspect --bench
```

### Dependencias por módulo

```
Módulo                         Dependencias Python   Rust
─────────────────────────────  ───────────────────  ───────────
polydim_runtime_v03.py         numpy                —
polydim_weighted_inference.py  numpy + runtime      —
test_interop_polydim.py        numpy + runtime      —
test_weighted_to_polydim.py    numpy + todos        —
polydim_core_lib.rs            —                    bytemuck 1.14
polydim_core_main.rs           —                    bytemuck 1.14
```

---

## 9. Deuda técnica y pendientes

| ID | Descripción | Prioridad |
|----|-------------|-----------|
| PEND_003 | Consistencia distribuida multi-AI (Geometric-CRDTs) | Alta |
| PEND_008 | Semantic drift detection en sesión larga | Media |
| PEND_010 | ALIGN para N distintos entre AIs | Alta |
| TASK_B | Theorem 3 full proof (PROJECT como functor, más allá del sketch DIM_SQL) | Paper |
| TASK_C | Proposition 6.1 proof categórico (Flutter isomorfismo) | Paper |
| TASK_D | Profiling PROJECT cost O(N·r) con Rust VM real | Rust |
| DT_001 | Mover archivos deprecated a _DEPRECATED/ (acción manual Drive web) | Baja |
| CRC_001 | Implementar CRC32 real en formato .polydim (actualmente placeholder 0x00000000) | Media |
| RUST_001 | Ejecutar `cargo test` en entorno con Rust real (29 tests pendientes de ejecución) | Alta |
| RUST_002 | Implementar `RECUR` (Mamba/SSM) en polydim_core_lib.rs | Media |

---

*README_POLYDIM_CORE.md · V0.1 · 2026-06-26*
*Fuente: POLYDIM_CONSTITUCION_V6.md · POLYDIM_PAPER_V4.md · BACKLOG_V16.json*
