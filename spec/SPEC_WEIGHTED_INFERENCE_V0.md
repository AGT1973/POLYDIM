# POLYDIM_DEST
# destination: polydim/spec/
# filename:    SPEC_WEIGHTED_INFERENCE_V0.md
# author:      ai.mpat.agt@gmail.com

# SPEC_WEIGHTED_INFERENCE_V0
**Version:** V0.1
**Date:** 2026-06-26
**Status:** APPROVED
**Resolves:** PEND_002
**Implementado en:** polydim_weighted_inference.py (polydim/core/)
**Tests:** 20/20 PASS

---

## 1. PROBLEMA QUE RESUELVE

Un `ObjectND` recibe pesos explícitos al construirse (`w=0.9`).
Cuando un objeto llega via PBF0 o MODO_S sin pesos reales (todos en 1.0),
o cuando el contexto de uso cambia con el tiempo, ¿qué peso corresponde
a cada dimensión?

**PEND_002** definía: *"inferir pesos óptimos por contexto de uso.
Candidato: activación relativa en el manifold como proxy de relevancia."*

---

## 2. ALGORITMO — DOS MODOS

### 2.1 One-shot: `infer_weights(obj)`

**Entrada:** `ObjectND` con dimensiones declaradas (con o sin pesos explícitos).
**Salida:** `dict {dim_name: float}` donde los valores suman ≈ 1.

```
Para cada dim d en obj._props:
    act[d]  = obj.activacion(d)               # proyección sobre sub-espacio [0,1]
    net[d]  = max(0, act[d] - UMBRAL)         # eliminar piso de ruido (~0.510)

total = sum(net.values())
if total < 1e-10:
    w[d] = 1/len(dims)  # uniforme si no hay señal diferencial
else:
    raw[d] = net[d] / total                   # normalizar
    w[d]   = max(raw[d], MIN_W)               # floor = 0.05
    w[d]  /= sum(w.values())                  # re-normalizar post-floor
```

**Fundamento geométrico:** La proyección `_proj(hv, sub_d)` mide cuánto
del hipervector total "apunta" hacia el subespacio de la dimensión d.
Dimensiones con mayor contenido codificado → mayor activación → mayor peso inferido.

**Complejidad:** O(|dims| × N) — una proyección por dimensión.

---

### 2.2 Usage-tracked: `WeightedInferenceEngine`

Combina dos señales con blend α:

```
w_combined[d] = α × w_manifold[d] + (1−α) × w_usage[d]

α = 0.70   (manifold domina; uso modula)
```

**Signal 1 — activación en manifold** (`w_manifold`):
Igual que one-shot. Captura la geometría intrínseca del objeto.

**Signal 2 — frecuencia de uso** (`w_usage`):
```
observe(obj, dim)  →  usage_count[dim] += 1
w_usage[d] = usage_count[d] / sum(usage_count.values())
```

El historial está limitado a 100 observaciones (FIFO) para evitar drift.

---

## 3. PARÁMETROS Y CONSTANTES

| Constante | Valor | Justificación |
|---|---|---|
| `UMBRAL` | 0.510 | Mismo que runtime: `0.5 + 2*(1/2√N)` |
| `ALPHA` | 0.70 | Manifold explica 70%, uso 30% |
| `MIN_W` | 0.05 | Evita que una dimensión quede en exactamente 0 |
| `MAX_USAGE_MEMORY` | 100 | Cap de historial anti-drift |

---

## 4. API PÚBLICA

```python
from polydim_weighted_inference import (
    infer_weights,          # one-shot
    apply_inferred_weights, # mutar obj._w + invalidar cache
    reweight,               # infer + apply en una llamada
    WeightedInferenceEngine,# usage-tracked
    activation_report,      # debug: raw/net/declared/inferred por dim
    UMBRAL, ALPHA, MIN_W
)
```

### Ejemplo mínimo

```python
from polydim_runtime_v03 import Space, ObjectND
from polydim_weighted_inference import infer_weights, reweight

sp  = Space("MI_IA")
obj = ObjectND(sp).add("DIM_SQL", {"tabla": "ventas"}, w=1.0) \
                  .add("DIM_PYTHON", {"fn": "aggregate"}, w=1.0)

# Inferencia one-shot
pesos = infer_weights(obj)
# → {"DIM_SQL": 0.61, "DIM_PYTHON": 0.39}  (ejemplo)

# Aplicar y reconstruir hipervector
obj = reweight(obj)
```

### Ejemplo con motor de uso

```python
from polydim_weighted_inference import WeightedInferenceEngine

engine = WeightedInferenceEngine(sp)

# Simular 3 consultas: 2 a SQL, 1 a Python
engine.observe(obj, "DIM_SQL")
engine.observe(obj, "DIM_SQL")
engine.observe(obj, "DIM_PYTHON")

pesos = engine.infer(obj)
# SQL recibe más peso por 2:1 de uso (modulado al 30%)
# → {"DIM_SQL": 0.67, "DIM_PYTHON": 0.33}  (ejemplo)
```

---

## 5. REPORTE DE ACTIVACIÓN (DEBUG)

```python
from polydim_weighted_inference import activation_report
rep = activation_report(obj)
# {
#   "DIM_SQL": {
#     "activation_raw":  0.7421,  # _proj(hv, sub_DIM_SQL)
#     "activation_net":  0.2321,  # max(0, raw - UMBRAL)
#     "weight_declared": 1.0000,  # obj._w["DIM_SQL"]
#     "weight_inferred": 0.6100   # output de infer_weights
#   },
#   "DIM_PYTHON": { ... }
# }
```

---

## 6. INTEGRACIÓN CON RUNTIME V0.8

`polydim_weighted_inference.py` opera sobre cualquier `ObjectND` que exponga:
- `obj._props` — dims declaradas
- `obj.activacion(dim)` — proyección
- `obj._w` — dict de pesos mutables
- `obj._cache` — cache a invalidar (se pone en `None`)

Compatible con V0.3+ sin modificaciones al runtime.
Para usar con `from_pbf0()`, invocar `reweight()` post-reconstrucción:

```python
obj = ObjectND.from_pbf0(data, space)
obj = reweight(obj)   # inferir pesos desde la geometría recibida
```

---

## 7. LIMITACIONES

**L_001 — Solo dimensiones declaradas:**
`infer_weights()` solo opera sobre `obj._props`. No descubre dimensiones
no declaradas aunque tengan activación > UMBRAL. Para descubrimiento usar
`obj.dims_activas()` primero y luego `add()` las nuevas.

**L_002 — Activación vs. relevancia semántica:**
La activación geométrica no es idéntica a relevancia semántica. Un objeto
puede tener alta proyección sobre DIM_SQL por GEO_ID accidental, no por contenido real.
`activation_net` (resta del UMBRAL) mitiga esto pero no lo elimina.

**L_003 — α fijo:**
El blend α=0.70 es una constante. Contextos donde el uso es muy informativo
(bots especializados) se beneficiarían de α más bajo. No está expuesto como
parámetro en la API pública actual.

**L_004 — No aprendizaje online:**
El `WeightedInferenceEngine` actualiza conteos de uso pero no actualiza los
pesos del Space ni del objeto de forma persistente. Cada sesión comienza desde 0.

---

## 8. APERTURA A PENDS FUTUROS

**PEND_ONLINE_WEIGHTS:** Aprendizaje online de pesos — actualizar α dinámicamente
según la correlación entre activación y respuesta correcta (feedback externo).
Requiere un mecanismo de reward señal no definido aún.

**PEND_CROSS_SPACE_WEIGHTS:** Inferir pesos desde un objeto recibido de otro Space
(post-ALIGN). La proyección sobre subespacios propios da activación, pero la
calibración puede diferir entre Spaces con distintos seeds.

**PEND_BATCH_INFERENCE:** Inferencia eficiente de pesos para N objetos a la vez
usando operaciones matriciales (matriz de activaciones [N_objetos x N_dims]).

---

## 9. TESTS (20/20 PASS)

| Test | Qué verifica |
|---|---|
| T01-T04 | `infer_weights` retorna dict válido con suma ≈ 1, valores en (0,1] |
| T05 | Dimensión con w declarado alto → peso inferido mayor |
| T06 | Objeto sin dims → `{}` |
| T07-T09 | `apply_inferred_weights` muta `_w` e invalida cache |
| T10-T11 | `reweight` es conveniente y retorna obj |
| T12-T14 | `WeightedInferenceEngine` retorna pesos válidos |
| T15 | Dimensión no declarada ignorada en `observe` |
| T16-T19 | `activation_report` estructura correcta |
| T20 | `reset()` limpia historial |

---

## 10. STATUS

PEND_002 → RESUELTO
TASK_022 → DONE (polydim_weighted_inference.py)
TASK_027 → DONE (esta spec)
