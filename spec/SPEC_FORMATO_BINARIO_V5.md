# POLYDIM_DEST
# destino: polydim/spec/
# nombre:  SPEC_FORMATO_BINARIO_V5.md
# autor:   ai.mpat.agt@gmail.com
# fecha:   2026-06-25
# tarea:   TASK_021
# reemplaza: SPEC_FORMATO_BINARIO_V0.md (si existe)

---

# POLYDIM — Especificación del Formato Binario V5
## .polydim — LoRA Standard

**Versión del formato:** 5
**Magic bytes:** `POLY` (0x504F4C59)
**Estándar de compresión:** Low-Rank Adaptation (LoRA): T = W₀ + U·Vᵀ
**Endianness:** Big-endian para todos los campos del header
**Precisión por defecto:** float16 para datos tensoriales

---

## MOTIVACIÓN DEL CAMBIO V2 → V5

La especificación original (Constitución V2, Artículo IX) definía:

```
[ TRANSFORMS ]  float32[N×N]   — una matriz densa por transformación
```

Para N=10,000: una sola transformación ocupa **10,000 × 10,000 × 4 bytes = 400 MB**.
Inviable para distribución, imposible de cargar en RAM sin fragmentar.

V5 adopta LoRA como estándar mandatorio:

```
T = W₀ + U · Vᵀ    donde U, V ∈ R^{N×r},  r ≪ N
```

Para N=10,000 y r=64:
```
Dense (V2):  400 MB por transformación
LoRA (V5):   2 × 10,000 × 64 × 2 bytes (float16) = 2.56 MB por transformación
Reducción:   156× en tamaño
Pérdida:     Aproximación, no exacta. Para N grande, rank-64 captura >99% de la varianza típica.
```

Este cambio fue aprobado como parte de las Resoluciones Técnicas V5 (2026-06-22).

---

## ESTRUCTURA DEL ARCHIVO .polydim V5

### Layout general

```
[HEADER]        32 bytes fijos
[GEO_IDS]       float16[N × n_objects]
[TRANSFORMS]    variable — n_transforms bloques LoRA
[ACTIVATIONS]   float16[n_subspaces × n_objects]
[SUBSPACE_MAP]  uint8[n_subspaces]
[PROJ_TARGETS]  uint8[n_projections]
[CHECKSUM]      uint32 — CRC32 de todos los bloques anteriores
```

### Sección HEADER (32 bytes exactos)

| Offset | Bytes | Tipo      | Campo            | Descripción |
|--------|-------|-----------|------------------|-------------|
| 0      | 4     | bytes     | `magic`          | `0x504F4C59` = "POLY" |
| 4      | 1     | uint8     | `version`        | = 5 |
| 5      | 4     | uint32-BE | `N`              | Dimensión del espacio embedding |
| 9      | 2     | uint16-BE | `rank`           | Rango LoRA (r). Típico: 64 |
| 11     | 4     | uint32-BE | `n_transforms`   | Cantidad de transformaciones en el archivo |
| 15     | 4     | uint32-BE | `n_objects`      | Cantidad de objetos (GEO_IDs) |
| 19     | 1     | uint8     | `n_subspaces`    | Cantidad de subespacios (actualmente = 9) |
| 20     | 1     | uint8     | `n_projections`  | Cantidad de executors objetivo |
| 21     | 1     | uint8     | `flags`          | Bitfield: bit0=has_W0, bit1=sparse, bit2=quantized |
| 22     | 10    | bytes     | `reserved`       | Reservado para versiones futuras (= 0x00) |

**Notas sobre el header:**
- `magic` debe ser exactamente `POLY` — rechazar archivos con otro magic
- `version=5` distingue de formatos anteriores (V2 era no binario, V3/V4 nunca existieron como .polydim)
- `flags.bit0 (has_W0)`: si 1, cada bloque de transformación incluye W₀ densa
- `flags.bit1 (sparse)`: si 1, usar formato sparse (ver Apéndice A)
- `flags.bit2 (quantized)`: reservado para cuantización int8 futura

### Sección GEO_IDs

```
float16[N × n_objects]  →  tamaño: N × n_objects × 2 bytes
```

Los GEO_IDs son los hipervectores base invariantes. Se almacenan en orden column-major:
- GEO_ID del objeto 0: bytes[0 : N×2]
- GEO_ID del objeto 1: bytes[N×2 : 2N×2]
- ...

**Propiedad invariante:** Ninguna transformación del archivo modifica estos vectores.
Un loader que modifique los GEO_IDs después de cargarlos está en error.

### Sección TRANSFORMS

Para cada transformación i en `0..n_transforms`:

```
[W0_PRESENT]    uint8    — 1 si W₀ presente, 0 si pure LoRA
[W0]            float16[N × N]   (SOLO si W0_PRESENT=1 y flags.has_W0=1)
[U]             float16[N × rank]
[V]             float16[rank × N]
```

La transformación se aplica como:
```
T(v) = U @ (V @ v)             si W0_PRESENT=0
T(v) = W0 @ v + U @ (V @ v)   si W0_PRESENT=1
```

**Recomendación:** Usar pure LoRA (W0_PRESENT=0) para distribución. W₀ solo cuando la base densa es necesaria (fine-tuning inicial, Procrustes exacto).

**High-Definition Mode:** Si `W0_PRESENT=1` y la aplicación no soporta matrices densas (RAM limitada), puede aproximar via `T(v) ≈ U @ (V @ v)` con pérdida controlada. El campo `flags.bit0` advierte al loader de la presencia de W₀.

### Sección ACTIVATIONS

```
float16[n_subspaces × n_objects]
```

Matriz de activaciones A[i][j] = intensidad del subespacio i para el objeto j.
Valores en [0.0, 1.0] (clampeados al cargar si están fuera de rango).

El orden de subespacios debe coincidir con `SUBSPACE_MAP`.

### Sección SUBSPACE_MAP

```
uint8[n_subspaces]
```

Mapa de IDs de subespacios a sus índices en ACTIVATIONS:

| ID | Nombre      |
|----|-------------|
| 0  | DIM_PYTHON  |
| 1  | DIM_RUST    |
| 2  | DIM_FLUTTER |
| 3  | DIM_SQL     |
| 4  | DIM_GRAPH   |
| 5  | DIM_VECTOR  |
| 6  | DIM_TIME    |
| 7  | DIM_ERROR   |
| 8  | DIM_META    |

Los IDs 9-255 están reservados para subespacios de usuario (extensiones).

### Sección PROJ_TARGETS

```
uint8[n_projections]
```

Lista de executors objetivo para PROJECT:

| ID | Executor    |
|----|-------------|
| 0  | DIM_FLUTTER (RENDER) |
| 1  | DIM_RUST    (COMPILE) |
| 2  | DIM_WASM    (COMPILE) |
| 3  | DIM_SQL     (EXPORT) |
| 4  | DIM_PYTHON  (COMPILE) |
| 5  | DIM_GRAPH   (EXPORT) |

### Checksum final

```
uint32-BE  — CRC32 de todos los bytes anteriores (header + GEO_IDs + TRANSFORMS + ... + PROJ_TARGETS)
```

El loader DEBE verificar el checksum antes de usar los datos. Archivo corrupto → rechazar.

---

## IMPLEMENTACIÓN DE REFERENCIA (Python)

### Loader

```python
import struct
import zlib
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

MAGIC = b'POLY'
FORMAT_VERSION = 5
SUBSPACE_NAMES = [
    "DIM_PYTHON", "DIM_RUST", "DIM_FLUTTER", "DIM_SQL", "DIM_GRAPH",
    "DIM_VECTOR", "DIM_TIME", "DIM_ERROR", "DIM_META"
]

@dataclass
class PolydimTransform:
    U: np.ndarray          # float16 [N × r]
    V: np.ndarray          # float16 [r × N]
    W0: Optional[np.ndarray] = None  # float16 [N × N] o None

    def apply(self, v: np.ndarray) -> np.ndarray:
        out = self.U @ (self.V @ v.astype(np.float32))
        if self.W0 is not None:
            out = out + self.W0 @ v.astype(np.float32)
        return out

@dataclass
class PolydimFile:
    version: int
    N: int
    rank: int
    flags: int
    geo_ids: np.ndarray              # float16 [N × n_objects]
    transforms: List[PolydimTransform]
    activations: np.ndarray          # float16 [n_subspaces × n_objects]
    subspace_map: List[int]
    projection_targets: List[int]

def load_polydim(path: str) -> PolydimFile:
    """Carga un archivo .polydim V5."""
    with open(path, 'rb') as f:
        raw = f.read()

    # Verificar checksum
    checksum_stored = struct.unpack('>I', raw[-4:])[0]
    checksum_computed = zlib.crc32(raw[:-4]) & 0xFFFFFFFF
    if checksum_stored != checksum_computed:
        raise ValueError(f"Checksum inválido: stored={checksum_stored}, computed={checksum_computed}")

    pos = 0

    # --- HEADER ---
    magic = raw[pos:pos+4];  pos += 4
    if magic != MAGIC:
        raise ValueError(f"No es un archivo .polydim: magic={magic!r}")

    version = raw[pos];  pos += 1
    if version != FORMAT_VERSION:
        raise ValueError(f"Versión no soportada: {version} (esperado {FORMAT_VERSION})")

    N           = struct.unpack_from('>I', raw, pos)[0];  pos += 4
    rank        = struct.unpack_from('>H', raw, pos)[0];  pos += 2
    n_transforms= struct.unpack_from('>I', raw, pos)[0];  pos += 4
    n_objects   = struct.unpack_from('>I', raw, pos)[0];  pos += 4
    n_subspaces = raw[pos];  pos += 1
    n_projections = raw[pos]; pos += 1
    flags       = raw[pos];  pos += 1
    pos += 10  # reserved

    # --- GEO_IDs ---
    geo_size = N * n_objects * 2  # float16
    geo_ids = np.frombuffer(raw[pos:pos+geo_size], dtype=np.float16).reshape(N, n_objects).copy()
    pos += geo_size

    # --- TRANSFORMS ---
    transforms = []
    has_W0_flag = bool(flags & 0x01)
    for _ in range(n_transforms):
        w0_present = raw[pos];  pos += 1
        W0 = None
        if w0_present and has_W0_flag:
            w0_size = N * N * 2
            W0 = np.frombuffer(raw[pos:pos+w0_size], dtype=np.float16).reshape(N, N).copy()
            pos += w0_size
        U_size = N * rank * 2
        V_size = rank * N * 2
        U = np.frombuffer(raw[pos:pos+U_size], dtype=np.float16).reshape(N, rank).copy()
        pos += U_size
        V = np.frombuffer(raw[pos:pos+V_size], dtype=np.float16).reshape(rank, N).copy()
        pos += V_size
        transforms.append(PolydimTransform(U=U, V=V, W0=W0))

    # --- ACTIVATIONS ---
    act_size = n_subspaces * n_objects * 2
    activations = np.frombuffer(raw[pos:pos+act_size], dtype=np.float16
                                ).reshape(n_subspaces, n_objects).copy()
    activations = np.clip(activations, 0.0, 1.0)  # garantizar [0,1]
    pos += act_size

    # --- SUBSPACE_MAP ---
    subspace_map = list(raw[pos:pos+n_subspaces]);  pos += n_subspaces

    # --- PROJ_TARGETS ---
    projection_targets = list(raw[pos:pos+n_projections]);  pos += n_projections

    # pos + 4 = checksum (ya verificado)

    return PolydimFile(
        version=version,
        N=N,
        rank=rank,
        flags=flags,
        geo_ids=geo_ids,
        transforms=transforms,
        activations=activations,
        subspace_map=subspace_map,
        projection_targets=projection_targets,
    )
```

### Writer

```python
def save_polydim(pf: PolydimFile, path: str) -> int:
    """
    Guarda un PolydimFile como .polydim V5.
    Returns: tamaño en bytes del archivo generado.
    """
    import io
    buf = io.BytesIO()

    n_objects    = pf.geo_ids.shape[1]
    n_subspaces  = pf.activations.shape[0]
    n_projections = len(pf.projection_targets)
    has_W0 = any(t.W0 is not None for t in pf.transforms)
    flags = 0x01 if has_W0 else 0x00

    # --- HEADER ---
    buf.write(MAGIC)
    buf.write(struct.pack('B', FORMAT_VERSION))
    buf.write(struct.pack('>I', pf.N))
    buf.write(struct.pack('>H', pf.rank))
    buf.write(struct.pack('>I', len(pf.transforms)))
    buf.write(struct.pack('>I', n_objects))
    buf.write(struct.pack('B', n_subspaces))
    buf.write(struct.pack('B', n_projections))
    buf.write(struct.pack('B', flags))
    buf.write(b'\x00' * 10)  # reserved

    # --- GEO_IDs ---
    buf.write(pf.geo_ids.astype(np.float16).tobytes())

    # --- TRANSFORMS ---
    for t in pf.transforms:
        buf.write(struct.pack('B', 1 if t.W0 is not None else 0))
        if t.W0 is not None:
            buf.write(t.W0.astype(np.float16).tobytes())
        buf.write(t.U.astype(np.float16).tobytes())
        buf.write(t.V.astype(np.float16).tobytes())

    # --- ACTIVATIONS ---
    buf.write(np.clip(pf.activations, 0, 1).astype(np.float16).tobytes())

    # --- SUBSPACE_MAP ---
    buf.write(bytes(pf.subspace_map))

    # --- PROJ_TARGETS ---
    buf.write(bytes(pf.projection_targets))

    # --- CHECKSUM ---
    data = buf.getvalue()
    checksum = zlib.crc32(data) & 0xFFFFFFFF
    data = data + struct.pack('>I', checksum)

    with open(path, 'wb') as f:
        f.write(data)

    return len(data)
```

### Test round-trip

```python
def test_roundtrip(N: int = 256, rank: int = 8, n_obj: int = 2, path: str = "/tmp/test.polydim"):
    """Verifica que save→load produce el mismo PolydimFile."""
    import os
    rng = np.random.default_rng(42)

    # Crear archivo de prueba
    original = PolydimFile(
        version=FORMAT_VERSION,
        N=N, rank=rank, flags=0,
        geo_ids=rng.standard_normal((N, n_obj)).astype(np.float16),
        transforms=[
            PolydimTransform(
                U=rng.standard_normal((N, rank)).astype(np.float16),
                V=rng.standard_normal((rank, N)).astype(np.float16),
            )
        ],
        activations=rng.random((9, n_obj)).astype(np.float16),
        subspace_map=list(range(9)),
        projection_targets=[0, 1, 3],
    )

    size = save_polydim(original, path)
    loaded = load_polydim(path)
    os.unlink(path)

    # Verificar
    assert loaded.N == N
    assert loaded.rank == rank
    assert len(loaded.transforms) == 1
    np.testing.assert_array_equal(loaded.geo_ids, original.geo_ids)
    np.testing.assert_array_equal(loaded.transforms[0].U, original.transforms[0].U)
    np.testing.assert_array_equal(loaded.transforms[0].V, original.transforms[0].V)
    np.testing.assert_array_almost_equal(loaded.activations, original.activations, decimal=2)
    print(f"✓ Round-trip OK — archivo: {size} bytes ({size/1024:.1f} KB)")
    return True
```

---

## COMPARATIVA DE FORMATOS CANDIDATOS

| Formato | Velocidad carga | Tamaño (N=10k, r=64) | Python | Rust | Flutter/Dart |
|---|---|---|---|---|---|
| **LoRA+msgpack** | Muy rápida | ~2.6 MB | ✓ | ✓ | ✓ |
| LoRA+protobuf | Rápida | ~2.8 MB | ✓ | ✓ | ✓ |
| numpy .npy | Rápida | ~2.6 MB | ✓ | Parcial | ✗ |
| HDF5/zarr | Media | ~3 MB | ✓ | Parcial | ✗ |
| JSON+base64 | Lenta | ~7 MB | ✓ | ✓ | ✓ |

**Decisión V5:** Formato binario propio (este documento) con CRC32.
**Razón:** Máxima portabilidad sin dependencia de librería de serialización externa.
**Alternativa oficial:** LoRA+msgpack como encoding interno (msgpack es binario+compacto).

---

## APÉNDICE A — FORMATO SPARSE (flags.bit1=1)

Cuando una transformación tiene pocas dimensiones activas (activaciones < THRESHOLD para la mayoría de subespacios), se puede usar formato sparse:

```
[SPARSE_HEADER]   uint32-BE = n_nonzero_elements
[INDICES]         uint32-BE[n_nonzero_elements]   — índices planos en U y V
[VALUES_U]        float16[n_nonzero_elements]
[VALUES_V]        float16[n_nonzero_elements]
```

Punto de break-even: cuando n_nonzero < N·r/2.

---

## APÉNDICE B — REGISTRO DE VERSIONES

| Versión | Fecha | Cambio principal |
|---|---|---|
| V2 | 2026-06-15 | Especificación original (float32[N×N], no implementada) |
| V5 | 2026-06-22 | Estándar LoRA — reemplaza V2 completamente |

No existen V3 ni V4 como formatos .polydim.

---

## APÉNDICE C — IMPLEMENTACIÓN RUST (esqueleto para TASK_023)

```rust
// core/polydim_loader_v5.rs
// Requiere: byteorder = "1.4", ndarray = "0.15", half = "2.0"

use byteorder::{BigEndian, ReadBytesExt};
use half::f16;
use ndarray::Array2;
use std::io::{Cursor, Read};

const MAGIC: [u8; 4] = *b"POLY";
const FORMAT_VERSION: u8 = 5;

pub struct PolydimTransform {
    pub u: Array2<f32>,    // [N × rank]
    pub v: Array2<f32>,    // [rank × N]
    pub w0: Option<Array2<f32>>,  // [N × N] si presente
}

impl PolydimTransform {
    pub fn apply(&self, v_in: &[f32]) -> Vec<f32> {
        let n = v_in.len();
        // lora_out = U @ (V @ v)
        let mut v_out = vec![0f32; n];
        // V @ v
        let rank = self.v.nrows();
        let mut intermediate = vec![0f32; rank];
        for i in 0..rank {
            for j in 0..n {
                intermediate[i] += self.v[[i, j]] * v_in[j];
            }
        }
        // U @ intermediate
        for i in 0..n {
            for k in 0..rank {
                v_out[i] += self.u[[i, k]] * intermediate[k];
            }
        }
        // Agregar W0 @ v si presente
        if let Some(ref w0) = self.w0 {
            for i in 0..n {
                for j in 0..n {
                    v_out[i] += w0[[i, j]] * v_in[j];
                }
            }
        }
        v_out
    }
}

pub fn load_polydim(path: &str) -> Result<Vec<PolydimTransform>, Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;

    // Verificar magic
    if &data[..4] != MAGIC {
        return Err(format!("Not a .polydim file: {:?}", &data[..4]).into());
    }

    let mut cursor = Cursor::new(&data);
    cursor.set_position(4);

    let version = cursor.read_u8()?;
    if version != FORMAT_VERSION {
        return Err(format!("Unsupported version: {}", version).into());
    }

    let n             = cursor.read_u32::<BigEndian>()? as usize;
    let rank          = cursor.read_u16::<BigEndian>()? as usize;
    let n_transforms  = cursor.read_u32::<BigEndian>()? as usize;
    let n_objects     = cursor.read_u32::<BigEndian>()? as usize;
    let n_subspaces   = cursor.read_u8()? as usize;
    let n_projections = cursor.read_u8()? as usize;
    let flags         = cursor.read_u8()?;
    let has_w0        = (flags & 0x01) != 0;

    // Skip reserved
    cursor.set_position(cursor.position() + 10);

    // Skip GEO_IDs
    let geo_size = n * n_objects * 2;
    cursor.set_position(cursor.position() + geo_size as u64);

    // Read transforms
    let mut transforms = Vec::with_capacity(n_transforms);
    for _ in 0..n_transforms {
        let w0_present = cursor.read_u8()? != 0;
        let w0 = if w0_present && has_w0 {
            let mut buf = vec![0u8; n * n * 2];
            cursor.read_exact(&mut buf)?;
            let floats: Vec<f32> = buf.chunks(2)
                .map(|b| f16::from_be_bytes([b[0], b[1]]).to_f32())
                .collect();
            Some(Array2::from_shape_vec((n, n), floats)?)
        } else {
            None
        };

        let u_size = n * rank * 2;
        let v_size = rank * n * 2;

        let mut u_buf = vec![0u8; u_size];
        cursor.read_exact(&mut u_buf)?;
        let u_floats: Vec<f32> = u_buf.chunks(2)
            .map(|b| f16::from_be_bytes([b[0], b[1]]).to_f32())
            .collect();

        let mut v_buf = vec![0u8; v_size];
        cursor.read_exact(&mut v_buf)?;
        let v_floats: Vec<f32> = v_buf.chunks(2)
            .map(|b| f16::from_be_bytes([b[0], b[1]]).to_f32())
            .collect();

        transforms.push(PolydimTransform {
            u: Array2::from_shape_vec((n, rank), u_floats)?,
            v: Array2::from_shape_vec((rank, n), v_floats)?,
            w0,
        });
    }

    Ok(transforms)
}
```

---

*SPEC_FORMATO_BINARIO_V5.md · V5.0 · 2026-06-25 · TASK_021*
*Aprobado en Resoluciones Técnicas V5 (2026-06-22)*
*Reemplaza la especificación float32[N×N] de CONSTITUCIÓN_V2 Artículo IX*
