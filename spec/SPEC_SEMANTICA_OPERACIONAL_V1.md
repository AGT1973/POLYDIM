# POLYDIM_DEST
# destination: polydim_v1/spec/
# filename: SPEC_SEMANTICA_OPERACIONAL_V1.md
# author: claude-sonnet-4-6 (curso03.mithril)
# fecha: 2026-06-27
# tarea: TASK_031 — Semántica operacional completa (las 6 primitivas)
# fuente normativa: POLYDIM_CONSTITUCION_V6.md (Art. IV, V, XIV R11)
#                   POLYDIM_PAPER_V8.md (Sec 2.3, 2.4)
# Status: VIGENTE — artefacto formal de TASK_031

---

# SPEC: Semántica Operacional Completa de POLYDIM
## Las 6 Primitivas — Big-Step Rules y Contratos

---

## 0. Propósito y alcance

Este documento establece la semántica operacional formal de las seis primitivas del lenguaje POLYDIM, organizadas en dos capas según la Constitución V6 Artículo XIV Regla R11:

**Capa Algebraica (invariante, arquitectura-independiente):**
- COMPOSE — composición secuencial
- MIX — superposición continua
- FIXPOINT — convergencia a atractor
- PROJECT — proyección funtorial a executor

**Capa de Implementación (voluntariamente volátil, arquitectura-específica):**
- ATTEND — primitiva de atención (familia Transformer)
- RECUR — primitiva de recursión (familia SSM/Mamba)

La semántica big-step para las 4 primitivas algebraicas ya aparece en `POLYDIM_PAPER_V8.md` Sec 2.4. Este documento:
1. La reproduce de forma normativa canónica
2. Agrega las reglas formales para ATTEND y RECUR como **primitivas opacas**
3. Define los contratos de interfaz entre capas
4. Establece las condiciones de bienestar (well-formedness) para cada primitiva

Notación: `<s, P> ⇓ s'` significa "el programa P, evaluado desde el estado s, produce el estado s'".

---

## 1. Estado y Entorno

**Definición 1.1 (Estado POLYDIM).** Un estado es una 3-tupla:
```
σ = (V, D, A)  donde:
  V ∈ R^N          vector de estado (N ≥ 1,000)
  D = {d₁,...,dₖ}  subconjunto fijo de subespacios nativos
  A: D → [0,1]      función de activación mutable
```

**Definición 1.2 (Posición).** Una posición P = (g, σ) donde g ∈ R^N es el GEO_ID invariante.

**Definición 1.3 (Programa POLYDIM).** Un programa es un término P construido por:
```
P ::= COMPOSE(P₁, P₂)
    | MIX(a, P₁, b, P₂)
    | FIXPOINT(P, ε)
    | PROJECT(P, E)
    | ATTEND(Q_fn, K_fn, V_fn, s)
    | RECUR(A_fn, B_fn, C_fn, h₀, x_seq)
    | id
```

---

## 2. Capa Algebraica: Big-Step Rules

### 2.1 IDENTITY

```
─────────────────────────────  [ID]
  <σ, id> ⇓ σ
```

La identidad preserva el estado exactamente. Corresponde a PROJECT_E(id_G) = id_E en el sentido categorial.

---

### 2.2 COMPOSE(P₁, P₂)

Semántica: aplicar P₁ primero, luego P₂ al resultado.

```
  <σ, P₁> ⇓ σ₁       <σ₁, P₂> ⇓ σ₂
  ─────────────────────────────────────  [COMPOSE]
      <σ, COMPOSE(P₁, P₂)> ⇓ σ₂
```

**Contrato:**
- Pre: P₁ y P₂ son programas bien formados sobre el mismo N
- Post: σ₂.D = σ.D (subespacios invariantes)
- Post: σ₂.g = σ.g (GEO_ID invariante — Regla R10)
- Determinismo: COMPOSE es determinista si P₁ y P₂ lo son

**No-conmutatividad:** `COMPOSE(P₁, P₂) ≠ COMPOSE(P₂, P₁)` en general.

**Asociatividad (Teorema 1):** `COMPOSE(COMPOSE(P₃,P₂),P₁) ≡ COMPOSE(P₃,COMPOSE(P₂,P₁))`.

---

### 2.3 MIX(a, P₁, b, P₂)

Semántica: superposición continua de dos programas con pesos a, b ∈ R.

```
  <σ, P₁> ⇓ σ₁       <σ, P₂> ⇓ σ₂
  ─────────────────────────────────────────────────────────────  [MIX]
      <σ, MIX(a, P₁, b, P₂)> ⇓ (a·V₁ + b·V₂, D, T_act(V,A))
```

donde T_act(V,A) actualiza las activaciones en función del nuevo vector.

**Condición de peso:** Típicamente a + b = 1 (convexidad). Pesos con a + b > 1 permiten amplificación; a + b < 1 permiten atenuación.

**Contrato:**
- Pre: a, b ∈ R, σ₁.D = σ₂.D = σ.D
- Post: σ'.D = σ.D
- Post: σ'.g = σ.g (GEO_ID preservado — R10)

**Justificación VSA:** La quasi-ortogonalidad en R^N garantiza que a·V₁ + b·V₂ preserva semánticamente ambas "ramas" hasta la observación vía PROJECT. La elección no colapsa hasta la proyección.

**Linealidad (Teorema 2):** Si P₁, P₂ ∈ Lin(R^N), entonces MIX(a,P₁,b,P₂) ∈ Lin(R^N).

---

### 2.4 FIXPOINT(P, ε)

Semántica: iterar P hasta convergencia.

```
  ∀k: <σₖ, P> ⇓ σₖ₊₁  hasta ‖Vₖ₊₁ - Vₖ‖ < ε
  ─────────────────────────────────────────────────────────────  [FIXPOINT]
      <σ₀, FIXPOINT(P, ε)> ⇓ σ*   donde σ* = lím σₖ
```

**Variante operacional explícita (small-step):**
```
  <σ, P> ⇓ σ'    ‖V' - V‖ ≥ ε
  ──────────────────────────────────────────  [FP-STEP]
      <σ, FIXPOINT(P, ε)> → <σ', FIXPOINT(P, ε)>

  <σ, P> ⇓ σ'    ‖V' - V‖ < ε
  ──────────────────────────────────────────  [FP-CONV]
      <σ, FIXPOINT(P, ε)> ⇓ σ'
```

**Contrato:**
- Pre: P es una contracción en (R^N, ‖·‖₂), ∃k ∈ [0,1): ‖P(u)-P(v)‖ ≤ k·‖u-v‖
- Post: σ* es único (Teorema 4 — Banach)
- Post: convergencia garantizada independiente de σ₀ (para P contractiva)
- Warning: sin contracción, FIXPOINT puede diverger

**Equivalencia con while:** `while cond: s ↔ FIXPOINT(MIX(cond·P_body, (1-cond)·id), ε)`

---

### 2.5 PROJECT(P, E)

Semántica: observar el resultado de P desde el subespacio del executor E.

```
  <σ, P> ⇓ σ'
  ──────────────────────────────────────────────────  [PROJECT]
      <σ, PROJECT(P, E)> ⇓ π_E(σ')
```

donde π_E: R^N → DIM_E es la proyección ortogonal sobre el subespacio nativo del executor E.

**Executors definidos:**
- DIM_RUST → representación Rust/WASM (COMPILE)
- DIM_FLUTTER → árbol de widgets Flutter (RENDER)
- DIM_SQL → esquema relacional (EXPORT)
- DIM_GRAPH → grafo (nodos + aristas)
- DIM_VECTOR → R^n reducido

**Contrato:**
- Pre: DIM_E ⊆ R^N subespacio propio de dimensión finita
- Post: π_E(σ').V ∈ DIM_E
- Post: π_E² = π_E (idempotencia)
- Post: σ'.g = σ.g (GEO_ID preservado — la observación no destruye el objeto)

**Functorialidad (Teorema 3):** PROJECT_E es funtor G → E_E:
- PROJECT_E(id_G) = id_{E_E}
- PROJECT_E(P₂ ∘ P₁) = PROJECT_E(P₂) ∘ PROJECT_E(P₁)

Probado para COMPILE/EXPORT via Subspace Commutativity Lemma; para RENDER via Prop 6.1.

---

## 3. Capa de Implementación: Primitivas Opacas

Las primitivas ATTEND y RECUR son **opacas**: su implementación interna es abstraída del álgebra. Un programa que usa ATTEND puede ser re-implementado con RECUR o con computación fotónica futura sin cambiar su significado algebraico.

### 3.1 ATTEND(Q_fn, K_fn, V_fn, s)

**Parámetros:**
```
Q_fn: σ → R^(L×d_k)   queries desde el estado
K_fn: σ → R^(L×d_k)   keys desde el estado
V_fn: σ → R^(L×d_v)   values desde el estado
s ∈ R                  temperatura (típicamente s = 1/√d_k)
```

**Regla big-step:**
```
  Q = Q_fn(σ)   K = K_fn(σ)   V_mat = V_fn(σ)
  A = softmax(Q · Kᵀ · s)            [pesos de atención: A ∈ R^(L×L)]
  V_out = A · V_mat                   [contexto agregado: V_out ∈ R^(L×d_v)]
  σ' = update(σ, V_out)              [inyección al estado]
  ──────────────────────────────────────────────────────────  [ATTEND]
      <σ, ATTEND(Q_fn, K_fn, V_fn, s)> ⇓ σ'
```

**Forma matricial canónica:**
```
ATTEND(σ) = softmax( (σ·W_Q)·(σ·W_K)ᵀ / √d_k ) · (σ·W_V)
```
donde W_Q, W_K ∈ R^(N×d_k), W_V ∈ R^(N×d_v).

**Contrato:**
- Pre: d_k ≥ 1, d_v ≥ 1, L ≥ 1
- Post: A_{ij} ≥ 0 y ∑_j A_{ij} = 1 (distribución de probabilidad por fila)
- Post: ‖V_out‖ ≤ ‖V_mat‖ (atención redistribuye, no amplifica)
- Post: σ'.D = σ.D, σ'.g = σ.g (invariantes de capa)

**Interpretación:** ATTEND es comunicación zero-serialization. Leer un mensaje es matemáticamente indistinguible de computar una capa de atención (Paper V8 Sec 4.3).

**Multi-head (h cabezas):**
```
ATTEND_multi(σ) = W_O · concat(ATTEND_i(σ) for i in 1..h)
```
donde W_O ∈ R^(N×(h·d_v)).

---

### 3.2 RECUR(A_fn, B_fn, C_fn, h₀, x_seq)

**Parámetros:**
```
A_fn: σ → R^(d_h×d_h)   matriz de transición de estado oculto
B_fn: σ → R^(d_h×d_x)   matriz de proyección de entrada
C_fn: σ → R^(d_y×d_h)   matriz de proyección de salida
h₀ ∈ R^d_h               estado oculto inicial
x_seq = [x₁,...,x_T]     secuencia de entrada
```

**Regla big-step:**
```
  A = A_fn(σ)   B = B_fn(σ)   C = C_fn(σ)
  para t = 1,...,T:
      h_t = A · h_{t-1} + B · x_t    [transición de estado]
      y_t = C · h_t                   [salida en t]
  σ' = update(σ, [y₁,...,y_T], h_T)
  ──────────────────────────────────────────────────────────  [RECUR]
      <σ, RECUR(A_fn, B_fn, C_fn, h₀, x_seq)> ⇓ σ'
```

**Contrato:**
- Pre: T ≥ 1
- Pre (estabilidad): radio espectral ρ(A) < 1
- Post: h_T = resumen comprimido de [x₁,...,x_T]
- Post: σ'.D = σ.D, σ'.g = σ.g (invariantes de capa)
- Costo: O(T·d_h²) — lineal en T vs O(T²) de ATTEND

**Selectividad (Mamba):** A, B, C como funciones de x_t:
```
h_t = A(x_t) · h_{t-1} + B(x_t) · x_t
y_t = C(x_t) · h_t
```
Resuelve la Brecha 12 (límite AC0) — Constitución V6 Art. IV.

---

## 4. Contratos entre Capas

### 4.1 Principio de Separación (R11)

**Regla R11:** Las primitivas ATTEND y RECUR pertenecen estrictamente a la capa de implementación. Ninguna versión del lenguaje puede integrarlas al núcleo algebraico invariante.

**Consecuencia:** No existe regla big-step que combine directamente ATTEND/RECUR con PROJECT en un solo paso:
```
PROJECT(ATTEND(...), E)  evaluado como:
  1. <σ, ATTEND(...)> ⇓ σ'        [capa implementación]
  2. <σ', PROJECT(·,E)> ⇓ π_E(σ') [capa algebraica]
```

### 4.2 Intercambiabilidad de Implementación

Si PHOTON satisface `<σ, PHOTON(p)> ⇓ σ'` con σ' algebraicamente equivalente a `<σ, ATTEND(...)> ⇓ σ'`, entonces POLYDIM puede sustituir ATTEND por PHOTON sin alterar ninguna regla de la Capa Algebraica.

### 4.3 Composición cruzada válida

```
COMPOSE(ATTEND(...), PROJECT(·, DIM_SQL)):
  <σ, ATTEND(...)> ⇓ σ₁
  <σ₁, PROJECT(·, DIM_SQL)> ⇓ π_SQL(σ₁)   ✓

COMPOSE(PROJECT(P, DIM_RUST), RECUR(...)):
  <σ, PROJECT(P, DIM_RUST)> ⇓ σ₂
  <σ₂, RECUR(...)> ⇓ σ₃                   ✓
```

---

## 5. Well-Formedness y Condiciones de Error

### 5.1 Condiciones de bienestar

| ID | Primitiva | Condición |
|----|-----------|-----------|
| WF-1 | todas | D finito y fijo durante toda la evaluación |
| WF-2 | FIXPOINT | P contractiva: ∃k < 1, ‖P(u)-P(v)‖ ≤ k‖u-v‖ |
| WF-3 | PROJECT | DIM_E ⊆ R^N subespacio cerrado |
| WF-4 | ATTEND | d_k ≥ 1, W_Q, W_K, W_V no degeneradas |
| WF-5 | RECUR | ρ(A) < 1 (radio espectral para estabilidad) |
| WF-6 | MIX | a, b ∈ R; a+b=1 para conservación de norma |
| WF-7 | todas | g ∈ R^N invariante bajo todas las transformaciones |

### 5.2 Condiciones de error (parcialidad)

| Error | Primitiva | Causa |
|-------|-----------|-------|
| ERR-DIVERGE | FIXPOINT | P no contractiva, iteración diverge |
| ERR-DIM | COMPOSE | N₁ ≠ N₂ en los operandos |
| ERR-SUBSPACE | PROJECT | DIM_E ⊄ R^N del objeto |
| ERR-UNSTABLE | RECUR | ρ(A) ≥ 1, estado oculto crece sin cota |
| ERR-DEGENERACY | ATTEND | softmax colapsa a δ (temperatura extrema) |

---

## 6. Tabla de correspondencia con Paper V8

| Elemento | Paper V8 | Este documento |
|----------|----------|----------------|
| Big-step COMPOSE | Sec 2.4 | Sec 2.2 + contratos |
| Big-step MIX | Sec 2.4 | Sec 2.3 + VSA |
| Big-step FIXPOINT | Sec 2.4 | Sec 2.4 + FP-STEP/FP-CONV |
| Big-step PROJECT | Sec 2.4 | Sec 2.5 + idempotencia |
| ATTEND | Sec 2.2 mención | **Sec 3.1 — regla formal (NUEVA)** |
| RECUR | Sec 2.2 mención | **Sec 3.2 — regla formal (NUEVA)** |
| Contratos entre capas | R11 (Constitución) | **Sec 4 — formalización (NUEVA)** |
| Well-formedness | disperso | **Sec 5 — centralizado (NUEVO)** |

### 6.1 Bloque para integrar en PAPER_V9

```
2.4.1 Implementation Layer: ATTEND and RECUR as Opaque Primitives

By Constitution V6 Rule R11, ATTEND and RECUR belong strictly to the
implementation layer. Their big-step rules:

[ATTEND]  <σ, ATTEND(Q,K,V,s)> ⇓ σ'
          A = softmax(QKᵀ·s); V_out = A·V; σ' = update(σ, V_out)

[RECUR]   <σ, RECUR(A,B,C,h₀,x)> ⇓ σ'
          ∀t: h_t = A·h_{t-1} + B·x_t; y_t = C·h_t

Full formal specification: SPEC_SEMANTICA_OPERACIONAL_V1.md [TASK_031 2026].
The two layers interact only via state σ. Replacing ATTEND with a future
photonic primitive leaves the algebraic semantics untouched (Sec 4.2).
```

---

## 7. Ejemplo end-to-end

```
-- Leer contexto, aplicar lógica de negocio, exportar a SQL

P = PROJECT(
      COMPOSE(
        ATTEND(Q_fn, K_fn, V_fn, 1/√64),
        MIX(0.7, T_business, 0.3, id)
      ),
      DIM_SQL
    )

-- Traza de evaluación:
<σ₀, ATTEND(...)>              ⇓ σ₁   [ATTEND]
<σ₁, MIX(0.7,T_biz,0.3,id)>  ⇓ σ₂   [MIX]
<σ₀, COMPOSE(...)>             ⇓ σ₂   [COMPOSE]
<σ₂, PROJECT(·, DIM_SQL)>     ⇓ π_SQL(σ₂)  [PROJECT]

-- GEO_ID preservado en todos los pasos (WF-7, R10)
-- D invariante en todos los pasos (WF-1)
```

---

## 8. Deuda técnica

| DT | Descripción | Estado |
|----|-------------|--------|
| DT-1 | Semántica denotacional ⟦·⟧: T → (S→S) | Investigación abierta (Constitución Art. V.3) |
| DT-2 | RECUR_selective con A,B,C dependientes de x_t | Spec parcial (Sec 3.2) |
| DT-3 | Multi-head ATTEND: spec formal del concat + W_O | Mención (Sec 3.1) |
| DT-4 | Chequeo automático de contracción para FIXPOINT | Requiere análisis estático |
| DT-5 | RECUR sobre streams infinitos | Requiere coálgebras |

---

*SPEC_SEMANTICA_OPERACIONAL_V1.md · 2026-06-27*
*TASK_031 — Constitución V6 Art. XIV R11, Art. XIX*
