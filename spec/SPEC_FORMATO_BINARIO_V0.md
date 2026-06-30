# POLYDIM_DEST
# destination: polydim_v1/spec/
# filename: SPEC_FORMATO_BINARIO_V0.md
# autor: claude-sonnet-4-6
# fecha: 2026-06-27
# tarea: TASK_021
# fuentes normativas:
#   - POLYDIM_PAPER_V7.md Sec 5 (fileId 1nKKcnluDM3PuUgJI_KcF5Hf2oW1JNmo8)
#   - POLYDIM_CONSTITUCION_V6.md Art. XI Regla R12
#   - POLYDIM_THEOREM3_PROOF_V1.md Sec II (fileId 1W41O6eNTKIjLHPoswBRVfgmNQDQZIC32)
# nota: POLYDIM_PAPER_APPENDIX_E_V1.md no existe físicamente en docs_v6/ —
#       su contenido fue incorporado directamente en PAPER_V7 Sec 5. Esta spec
#       consolida ambas fuentes en el artefacto formal de spec/ que TASK_021 pedía.

---

# SPEC — Formato Binario .polydim V0
## Especificación normativa del formato de serialización de transformaciones POLYDIM

**Estado:** VIGENTE · V0 · 2026-06-27
**Tier (Constitución V6 Art. XI):** ⚙️ MECANISMO — especificación técnica concreta, en implementación.

---

## 0. Propósito y alcance

Un archivo `.polydim` es **estado tensorial serializado**, no código fuente. Cuando una IA lo lee, aplica la transformación directamente a su propio estado vectorial — sin parsing, sin re-proyección entre lenguajes.

Esta spec define el layout binario V5 del formato `.polydim` tal como aparece en PAPER_V7 Sec 5.2, con las adiciones normativas de Art. XI de la Constitución V6 (regla R12: bajo rango por defecto, nunca denso N×N).

Alcance: serialización, deserialización, y contratos de alineación de memoria para ejecución SIMD en la VM Rust (TRACK 3).

---

## 1. Motivación: el problema de tamaño

Para N=10,000 (dimensionalidad típica de un modelo transformer), una matriz densa float32[N×N] ocupa:

```
N × N × 4 bytes = 10,000 × 10,000 × 4 = 400 MB por transformación
```

Esto hace inviable el formato denso como estándar de distribución. La solución normativa (R12) es la representación de bajo rango (LoRA):

```
T_geo = W_0 + U * V^T
donde U, V ∈ R^(N×r), r << N
```

Con N=10,000 y r=64 (rango por defecto):

```
W_0: float16[N] = 20,000 bytes (vector de sesgo, no matriz)
U:   float16[N×r] = 10,000 × 64 × 2 = 1,280,000 bytes
V:   float16[N×r] = 10,000 × 64 × 2 = 1,280,000 bytes
Total por transformación: ~2.58 MB
Reducción vs. denso: ~99.4%
```

---

## 2. Layout binario — .polydim V5

El archivo es una secuencia binaria contigua con las siguientes secciones. Todos los campos multibyte son little-endian. Alineación de cada sección a 64 bytes para SIMD.

### 2.1 HEADER (64 bytes, fijo)

```
Offset  Size  Type      Field
0       8     uint8[8]  magic: 0x50 0x4F 0x4C 0x59 0x44 0x49 0x4D 0x00
                        (ASCII "POLYDIM\0")
8       2     uint16    version: 5 (este formato)
10      4     uint32    N: dimensionalidad del espacio (e.g. 10000)
14      2     uint16    precision: 0=float16, 1=float32, 2=bfloat16
16      4     uint32    n_transforms: número de transformaciones en el archivo
20      2     uint16    rank_default (r): rango LoRA por defecto
22      4     uint32    n_objects: número de objetos con GEO_ID
26      4     uint32    n_dims: número de subspacios nativos declarados
30      2     uint16    flags: bit 0 = tiene ACTIVATIONS, bit 1 = tiene PROJECTIONS
32      32    uint8[32] reserved: ceros, para extensiones futuras
```

Total: 64 bytes ✓

### 2.2 GEO_IDs (alineado a 64 bytes tras HEADER)

```
Para cada objeto i en [0, n_objects):
  base_hv[i]: float16[N]   — hipervector base del GEO_ID
               = N × 2 bytes

Total sección: n_objects × N × 2 bytes
Alineación: padding hasta múltiplo de 64 bytes al final de la sección
```

Invariante (Constitución V6 Art. VI.3 / R10): `base_hv[i]` no cambia bajo ninguna transformación admisible. La VM DEBE verificar esto en carga.

### 2.3 TRANSFORMS (alineado a 64 bytes)

```
Para cada transformación j en [0, n_transforms):

  rank_j: uint16     — rango real de esta transformación (puede diferir de rank_default)
  _pad:   uint8[6]   — padding a 8 bytes

  W_0_j:  float16[N]       — vector de sesgo (N × 2 bytes)
  U_j:    float16[N×rank_j] — factor izquierdo LoRA (N × rank_j × 2 bytes)
  V_j:    float16[N×rank_j] — factor derecho LoRA (N × rank_j × 2 bytes)

  _align: padding hasta siguiente múltiplo de 64 bytes
```

Operación de la transformación en la VM:

```
T_j(v) = W_0_j + U_j @ (V_j^T @ v)
        = desplazamiento + proyección de bajo rango
```

Costo de aplicación: O(N·r) multiplicaciones — no O(N²).

### 2.4 ACTIVATIONS (opcional, presente si flags bit 0 = 1)

```
Para cada objeto i en [0, n_objects):
  weights[i]: float16[n_dims]  — pesos de activación a_d ∈ [0,1]
              para cada subespacio nativo d ∈ D

Total: n_objects × n_dims × 2 bytes
Alineación: padding a 64 bytes
```

### 2.5 PROJECTIONS (opcional, presente si flags bit 1 = 1)

```
Para cada transformación j:
  n_targets_j: uint16          — número de contratos de proyección
  targets_j:   uint8[n_targets_j]  — IDs de executor:
                                    0 = COMPILE (DIM_RUST)
                                    1 = RENDER (DIM_FLUTTER)
                                    2 = EXPORT_SQL (DIM_SQL)
                                    3 = EXPORT_GRAPH (DIM_GRAPH)
                                    [4-255 reservados]
  _align: padding a 64 bytes
```

---

## 3. Enumeración de operaciones

```
Op ID  Mnemónico     Descripción
0x01   COMPOSE       T2 ∘ T1: concatenar dos transforms (aplicar T1 primero)
0x02   MIX           a*T1 + b*T2: superposición ponderada
0x03   FIXPOINT      convergencia iterativa hasta ||s_{k+1} - s_k|| < ε
0x04   PROJECT       proyección a subespacio executor
0x05   ATTEND        cross-attention (capa de implementación)
0x06   RECUR         recurrencia SSM/Mamba (capa de implementación)
```

En el archivo .polydim V0, las operaciones no se almacenan como bytecode sino como transformaciones ya compuestas (resultado de COMPOSE aplicado al grafo). La VM lee T directamente y la aplica.

Nota constitucional (R9): las operaciones 0x05 y 0x06 (ATTEND, RECUR) son capa de implementación — no parte del núcleo algebraico. Una VM mínima puede omitirlas e implementar solo 0x01–0x04.

---

## 4. Verificación en carga (VM contract)

La VM DEBE verificar al cargar un .polydim:

1. **Magic bytes**: primeros 8 bytes = `POLYDIM\0`.
2. **Versión soportada**: `version` ≤ versión máxima que la VM soporta.
3. **Invariancia GEO_ID**: para cada `base_hv[i]`, verificar que ninguna transformación en TRANSFORMS lo modifica más de ε = 1e-6 (regresión BUG_002).
4. **Alineación SIMD**: cada sección empieza en offset múltiplo de 64.
5. **Rango válido**: para cada transformación, `rank_j` ≤ N.

Si cualquier verificación falla, la VM DEBE rechazar el archivo y reportar el error — nunca ejecutar parcialmente.

---

## 5. Ejemplo de tamaños para N=10,000, r=64, 3 transforms, 2 objetos

```
Sección         Tamaño
HEADER          64 B
GEO_IDs         2 obj × 10,000 × 2B = 40,000 B + padding = 40,000 B
TRANSFORMS      3 × (8 + 20,000 + 1,280,000 + 1,280,000 + padding)
                = 3 × ~2,580,008 B = ~7,740,024 B + padding
ACTIVATIONS     2 × 5 dims × 2B = 20 B + padding = 64 B (mínimo)
PROJECTIONS     3 × (2 + 3 targets + padding) = ~192 B

TOTAL ~7,780 KB ≈ 7.6 MB
(vs. 3 × 400 MB = 1,200 MB para formato denso — factor de compresión ~154×)
```

---

## 6. Referencias cruzadas normativas

| Concepto | Fuente canónica |
|---|---|
| R12: bajo rango por defecto | CONSTITUCION_V6.md Art. XI (fileId 1LneDD0D8fHREiZN-_Xw-Ax9AzxfF6tKv) |
| Layout binario (origin) | PAPER_V7.md Sec 5.2 (fileId 1nKKcnluDM3PuUgJI_KcF5Hf2oW1JNmo8) |
| Executor contracts | THEOREM3_PROOF_V1.md Sec II (fileId 1W41O6eNTKIjLHPoswBRVfgmNQDQZIC32) |
| GEO_ID invariance | PAPER_V7.md Sec 4.1 + BUG_002 test en polydim_tests.py |
| VM Rust target | TASK_023 (TASK_D) — siguiente tarea del backlog |

---

## 7. Estado de implementación

| Componente | Estado |
|---|---|
| Spec binaria (este documento) | ✅ VIGENTE |
| Parser Python (bootstrap) | ⚙️ Parcial — polydim_runtime_v03.py usa proxies de string |
| VM Rust con SIMD | ❌ Pendiente — TASK_023 |
| Validador de alineación | ❌ Pendiente — TASK_023 |
| Tests .polydim round-trip | ❌ Pendiente — extender polydim_tests.py |

---

*SPEC_FORMATO_BINARIO_V0.md · TASK_021 · 2026-06-27 · Claude Sonnet 4.6*
*Fuentes: PAPER_V7 Sec 5 + CONSTITUCION_V6 Art. XI*
