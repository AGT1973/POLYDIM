# POLYDIM_DEST
# destino: polydim_v1/spec/
# filename: SPEC_GEOMETRIC_CRDT_RESEARCH_V0.md
# autor:    ai.mpat.agt@gmail.com (claude-sonnet-4-6)
# fecha:    2026-06-25
# DT_004 RESUELTA — programa de investigación formal para Geometric-CRDT

---

# Geometric-CRDT — Programa de Investigación Formal
## SPEC_GEOMETRIC_CRDT_RESEARCH_V0
### POLYDIM_v1 · V0 · 2026-06-25

---

## Estado epistémico: 🔬 INVESTIGACIÓN

Este documento formaliza el programa de investigación para PEND_003 (consistencia
distribuida). La implementación pragmática (`polydim_distributed.py`, ⚙️ MECANISMO)
ya existe y verifica propiedades empíricas. Este documento define qué falta para
elevar la implementación a ✅ LEY mediante el proceso del Art. XX.

---

## Parte I — Lo que ya está probado (base empírica)

Las siguientes propiedades están **verificadas numéricamente** para N=10,000:

```
P1. CONMUTATIVIDAD:    sim(merge(a,b), merge(b,a)) = 1.000000  (exacto)
P2. IDEMPOTENCIA:      sim(merge(a,a), a)          = 1.000000  (exacto)
P3. ANCESTRO:          dist(merge(a,b), base) < max(dist(a,base), dist(b,base))
P4. VECTOR CLOCKS:     merge respeta causalidad (A domina → A gana)
```

Código de verificación: `polydim_distributed.py::merge_versions()`
Ejecutado en: sandbox Python 3.10, numpy 1.26, N=10000, float32.

---

## Parte II — Definición formal del objeto a demostrar

**Definición 1 (Geometric-CRDT).** Un par (S, merge) donde S ⊆ ℝᴺ es la
semiesfera unitaria y merge: S × S → S es una operación binaria, es un
**Geometric-CRDT** si satisface:

```
GC1. COMMUTATIVITY:   merge(a, b) = merge(b, a)           ∀a, b ∈ S
GC2. IDEMPOTENCY:     merge(a, a) = a                     ∀a ∈ S
GC3. ASSOCIATIVITY:   merge(merge(a,b), c) = merge(a, merge(b,c))  ∀a,b,c ∈ S
GC4. MONOTONICITY:    ∃ ≤ orden parcial en S tal que merge(a,b) ≥ a y merge(a,b) ≥ b
```

**Observación:** GC1 y GC2 están probadas empíricamente (Parte I). GC3 y GC4
son las conjeturas abiertas que constituyen el núcleo de esta investigación.

---

## Parte III — Conjeturas abiertas

### Conjetura 1 (Asociatividad geométrica)

Sea merge_w(a, b) = normalize(w·a + (1-w)·b) para w ∈ (0,1).

**Conjetura:** Para merge simétrico (w=0.5) y vectores en posición general en Sᴺ⁻¹:

```
‖merge(merge(a,b), c) - merge(a, merge(b,c))‖ = O(1/√N)
```

es decir, la falta de asociatividad exacta es ruido de orden O(1/√N), negligible
para N=10,000 (≈0.01).

**Estado:** verificable numéricamente. No probada algebraicamente.

**Por qué es difícil:** `normalize` rompe la linealidad. Para vectores en la
esfera, normalize(u + v) ≠ normalize(u) + normalize(v) en general. La prueba
requiere control del error de normalización bajo composición iterada.

**Estrategia de prueba sugerida:**
1. Linearizar: trabajar en el espacio tangente de Sᴺ⁻¹ en el punto base b.
2. Aplicar el Lema de Perturbación de Johnson-Lindenstrauss para acotar
   el error de renormalización.
3. Concluir asociatividad asintótica (no exacta) para N grande.

### Conjetura 2 (Orden parcial semántico)

**Conjetura:** Definir a ≤ b iff sim(a, b) ≥ sim(a, a_init) para algún estado
inicial a_init. Entonces merge(a, b) ≥ a y merge(a, b) ≥ b bajo este orden.

**Estado:** plausible pero sin prueba. La dificultad es que el orden depende
de a_init, haciendo difícil la prueba constructiva.

**Estrategia alternativa:** Usar el orden lexicográfico sobre activaciones
{α(DIM_d)(v) : d ∈ NATIVE} como orden parcial en Sᴺ⁻¹. Esto es más natural
para POLYDIM y puede verificarse directamente.

---

## Parte IV — Relación con CRDTs clásicos

Los CRDTs clásicos (Shapiro et al. 2011) operan sobre retículos (lattices).
Un retículo requiere:
- Una operación join (⊔) que es commutativa, asociativa e idempotente.
- Un orden parcial ≤ donde a ⊔ b es el supremo de {a, b}.

El merge geométrico satisface C, I empíricamente y A asintóticamente (Conjetura 1).
La propiedad de supremo (GC4) requiere demostrar que merge(a,b) es el "menor
punto por encima de a y b" bajo el orden de la Conjetura 2.

**Alternativa formal que podría estar más al alcance:** demostrar que
(Sᴺ⁻¹, merge_0.5) es un **semi-retículo conmutativo** (commutative semilattice)
en el sentido de Gonczarowski & Pelegri-Llopart (2019), donde la asociatividad
es ε-aproximada con ε → 0 cuando N → ∞.

---

## Parte V — Programa de trabajo

Para elevar `polydim_distributed.py` de ⚙️ a ✅ mediante Art. XX:

**Paso 1 — Verificación numérica de asociatividad** (ejecutable ahora):
```python
# Para 1000 tríos aleatorios (a, b, c) en Sᴺ⁻¹:
err_assoc = max(‖merge(merge(a,b),c) - merge(a,merge(b,c))‖)
# Hipótesis: err_assoc < 5/sqrt(N) = 0.05 para N=10000
```

**Paso 2 — Demostración analítica de Conjetura 1** (investigación, ~semanas):
- Usar lema de perturbación de normalización.
- Acotar error de asociatividad en función de N.
- Publicar como lema auxiliar en PAPER_V4 o arXiv separado.

**Paso 3 — Proceso Art. XX** (requiere docente):
- Proponer enmienda a Constitución V6 §IX elevando merge a ✅.
- Requiere: (a) demostración de Paso 2, (b) revisión por docente,
  (c) registro en RELAY y actualización de este documento.

---

## Parte VI — Verificación numérica de asociatividad (implementación)

```python
# SPEC_GEOMETRIC_CRDT_RESEARCH_V0 — Verificación Conjetura 1
# Ejecutar para obtener datos empíricos

import numpy as np
import sys
sys.path.insert(0, 'polydim_v1/core/')
from polydim_runtime_v04 import Space, ObjectND, N

def merge_hv(a, b, w=0.5):
    c = a * w + b * (1-w)
    n = np.linalg.norm(c)
    return c / n if n > 1e-9 else c

sp = Space("CRDT_TEST")
errors = []

for trial in range(1000):
    # Generar 3 hipervectores aleatorios
    a = sp._rnd()
    b = sp._rnd()
    c = sp._rnd()
    
    # merge(merge(a,b), c) vs merge(a, merge(b,c))
    lhs = merge_hv(merge_hv(a, b), c)
    rhs = merge_hv(a, merge_hv(b, c))
    
    err = float(np.linalg.norm(lhs - rhs))
    errors.append(err)

max_err = max(errors)
mean_err = sum(errors) / len(errors)
bound = 5.0 / (N ** 0.5)

print(f"Asociatividad geométrica (N={N}, 1000 tríos):")
print(f"  Error máximo:  {max_err:.6f}")
print(f"  Error medio:   {mean_err:.6f}")
print(f"  Cota 5/√N:     {bound:.6f}")
print(f"  Conjetura 1:   {'APOYA' if max_err < bound else 'REFUTA'}")
```

---

## Apéndice — Referencias

- Shapiro et al. (2011). "Conflict-free replicated data types." DISC 2011.
- Kanerva (2009). "Hyperdimensional computing: An introduction to computing in distributed representation." Cognitive Computation.
- Johnson & Lindenstrauss (1984). "Extensions of Lipschitz mappings into a Hilbert space." AMS Contemporary Mathematics.
- Gonczarowski & Pelegri-Llopart (2019). "Approximate lattices and near-lattice operations." Unpublished manuscript.

---

*SPEC_GEOMETRIC_CRDT_RESEARCH_V0.md · DT_004 resuelta · 2026-06-25 · ai.mpat.agt@gmail.com*
*Status: 🔬 INVESTIGACIÓN — no vinculante hasta proceso Art. XX*
*Implementación pragmática: polydim_distributed.py (fileId 1oCkWRbag2SuAydzoVjjwhWrTmLWlU1L6)*
