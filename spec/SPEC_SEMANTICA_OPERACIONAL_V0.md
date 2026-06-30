# POLYDIM_DEST
# destination: polydim_v1/spec/
# filename: SPEC_SEMANTICA_OPERACIONAL_V0.md
# autor: claude-sonnet-4-6
# fecha: 2026-06-27
# tarea: TASK_031
# fuentes normativas:
#   - POLYDIM_CONSTITUCION_V6.md Art. V (fileId 1LneDD0D8fHREiZN-_Xw-Ax9AzxfF6tKv)
#   - POLYDIM_CONSTITUCION_V6.md Art. IV.1 y IV.2
#   - POLYDIM_PAPER_V7.md Sec 2 (fileId 1nKKcnluDM3PuUgJI_KcF5Hf2oW1JNmo8)
# tier: ⚙️ MECANISMO para capa algebraica (big-step) + ⚙️ MECANISMO para
#        ATTEND/RECUR como primitivas opacas + 🔬 INVESTIGACION para
#        semántica denotacional ⟦T⟧ (sección 4, abierta según Art. V.3)

---

# SPEC — Semántica Operacional Completa de POLYDIM
## Las 6 primitivas: reglas big-step + primitivas opacas + orientación denotacional

**Estado:** VIGENTE · V0 · 2026-06-27
**Corresponde a:** TASK_A del Artículo XIX de la Constitución V6

---

## 0. Propósito y alcance

La Constitución V6 Art. V define la semántica operacional big-step para las
4 primitivas algebraicas (COMPOSE, MIX, FIXPOINT, PROJECT) y menciona ATTEND
y RECUR como "primitivas opacas" de la capa de implementación. Sin embargo,
no existe un artefacto formal en `spec/` que consolide las 6 primitivas
en un solo documento con reglas de evaluación, criterios de validez y
orientación hacia la semántica denotacional.

Este artefacto cierra esa brecha. Es el insumo directo para:
- La VM Rust (TASK_023/TASK_D): necesita las reglas exactas de evaluación.
- El paper POLYDIM_PAPER_V7 Sec 2 (ya completa, pero este documento la
  formaliza a nivel de spec independiente).
- Sesiones futuras de prueba/verificación.

---

## 1. Definiciones base (Constitución V6 Art. V.1)

```
Estado S = (V, D, A)
  V ∈ R^N        vector de posición en el espacio latente (N ≥ 1,000)
  D              conjunto finito de subspacios nativos observadores (fijo por objeto)
  A: D → [0,1]   función de activación (mutable)

Posición P = (g, S)
  g ∈ R^N        GEO_ID — invariante bajo toda transformación admisible (Regla R10)
  S = (V, D, A)  estado mutable

Transformación T: S → S
  En la capa algebraica: T es una función lineal T_geo: R^N → R^N
  que modifica V y puede actualizar A como función del nuevo V,
  dejando D y g invariantes.
  Forma LoRA (estándar de serialización): T_geo = W_0 + U·V^T, r << N
```

**Notación big-step:** `⟨s, T⟩ ⇒ s'` se lee "el estado s, bajo la
transformación T, evalúa al estado s'".

---

## 2. Reglas de evaluación — Capa algebraica invariante (Art. IV.1)

### 2.1 Regla general (T arbitraria)

```
⟨s, T⟩ ⇒ T(s)
```

Toda transformación admisible aplicada a un estado produce el estado imagen.

### 2.2 COMPOSE(T₁, T₂) — Composición algebraica

**Definición operacional:**
```
COMPOSE(T₁, T₂) = T₂ ∘ T₁
```

**Regla big-step:**
```
⟨s, T₁⟩ ⇒ s₁      ⟨s₁, T₂⟩ ⇒ s₂
─────────────────────────────────────────
⟨s, COMPOSE(T₁, T₂)⟩ ⇒ s₂
```

**Propiedades normativas:**
- No conmutativa: T₂ ∘ T₁ ≠ T₁ ∘ T₂ en general.
- Asociativa: COMPOSE(COMPOSE(T₃,T₂),T₁) = COMPOSE(T₃,COMPOSE(T₂,T₁))
  — Teorema 1, Constitución V6 Art. XV.
- El orden importa: T₁ se aplica primero.

**Representación matricial (LoRA):**
```
COMPOSE((W₀^(1) + U^(1)·V^(1)T), (W₀^(2) + U^(2)·V^(2)T))
= W₀^(2)·W₀^(1) + U^(2)·(V^(2)T·W₀^(1)) + (W₀^(2)·U^(1))·V^(1)T + ...
```
Nota: COMPOSE de dos matrices LoRA no es necesariamente LoRA de rango r —
puede requerir re-compresión. La VM puede diferir esta compresión.

---

### 2.3 MIX(α, T₁, β, T₂) — Superposición continua

**Definición operacional:**
```
MIX(α, T₁, β, T₂) = α·T₁ + β·T₂
```
(típicamente α + β = 1; el lenguaje no lo impone como invariante)

**Regla big-step:**
```
────────────────────────────────────────────────────────────────────────
⟨s, MIX(α, T₁, β, T₂)⟩ ⇒ α·T₁(s) + β·T₂(s)
```

**Propiedades normativas:**
- Si T₁, T₂ ∈ Lin(R^N), entonces MIX(α,T₁,β,T₂) ∈ Lin(R^N)
  — Teorema 2, Constitución V6 Art. XV.
- MIX reemplaza el `if/else` binario. La "elección" no ocurre en MIX:
  ocurre al proyectar (PROJECT). Ambas ramas coexisten en el espacio
  de alta dimensión (propiedad VSA: cuasi-ortogonalidad).
- Cuando α=1, β=0: equivale a T₁ pura.
- Cuando α=β=0.5: superposición equiponderada.

**Representación matricial (LoRA):**
```
MIX(α, W₀^(1)+U^(1)V^(1)T, β, W₀^(2)+U^(2)V^(2)T)
= (α·W₀^(1) + β·W₀^(2)) + [α·U^(1) | β·U^(2)]·[V^(1) | V^(2)]^T
```
La suma de dos matrices LoRA de rango r produce rango ≤ 2r (concatenación
de factores). Compresión opcional por la VM.

---

### 2.4 FIXPOINT(T, ε) — Convergencia iterativa

**Definición operacional:**
```
s₀ = estado inicial
s_{k+1} = T(s_k)
Detener cuando ‖s_{k+1} − s_k‖ < ε
Resultado: s_{k+1}
```

**Regla big-step:**
```
s_{k+1} = T(s_k)        ‖s_{k+1} − s_k‖ < ε
──────────────────────────────────────────────
⟨s_k, FIXPOINT(T, ε)⟩ ⇒ s_{k+1}
```

**Propiedades normativas:**
- Reemplaza los loops `for`/`while` imperativos.
- **Condición de aplicabilidad (Constitución V6 Art. XV, Teorema 4 —
  Banach):** FIXPOINT garantiza existencia y unicidad del punto fijo
  si y solo si T es una contracción: ‖T(u)−T(v)‖ ≤ k·‖u−v‖ con k<1.
  Verificar esta condición es responsabilidad del programador/compilador,
  no del runtime (que puede no terminar si T no contrae).
- La norma usada para ‖·‖ es ‖·‖₂ (euclidiana) salvo especificación
  contraria del ejecutor.
- ε > 0 es el umbral de convergencia; su valor afecta la precisión y
  el costo de cómputo.

**Contrato de la VM:**
- Máximo de iteraciones: la VM DEBE tener un límite de iteraciones configurable
  (por defecto: max_iter = 10,000) para evitar loops infinitos cuando T no
  contrae. Al alcanzar el límite, reportar FIXPOINT_DIVERGENCE y devolver
  el último s_k.

---

### 2.5 PROJECT(T, E) — Proyección a executor (funtor)

**Definición operacional:**
```
PROJECT_E(T) := T|_E    (restricción de la acción de T al subespacio DIM_E)
PROJECT_E(P) := π_E(P)  (proyección ortogonal de la posición al subespacio E)
```

**Regla big-step (esquema general):**
```
F = funtor asociado al executor E
──────────────────────────────────────────────────
⟨s, PROJECT(T, E)⟩ ⇒ F(T)(s)
```

**Los tres contratos operativos (fragmentación de PROJECT, Art. IV.1):**

```
COMPILE(T, DIM_target)
  Executor E = DIM_RUST o DIM_WASM
  Produce: binario estático ejecutable en hardware convencional
  Regla: π_E(T₂ ∘ T₁) = π_E(T₂) ∘ π_E(T₁)
         [por Subspace Commutativity Lemma, DIM_RUST es T-invariante]

RENDER(T, DIM_FLUTTER)
  Executor E = DIM_FLUTTER
  Produce: árbol de widgets Flutter nativo
  Regla: φ(COMPOSE(T₂,T₁)) = Column(φ(T₁), φ(T₂))
         [por Proposición 6.1, isomorfismo monoidal estricto φ: T→F,
          probado en PROPOSITION_6_1_PROOF_V1.md]

EXPORT(T, DIM_external)
  Executor E = DIM_SQL, DIM_GRAPH, u otro sistema no-tensorial
  Produce: representación en el sistema externo (query SQL, nodo de grafo, etc.)
  Regla: F_SQL(T₂ ∘ T₁) = F_SQL(T₂) ∘ F_SQL(T₁)
         [por Subspace Commutativity Lemma, DIM_SQL es T-invariante]
```

**Propiedades normativas — Teorema 3 (PROJECT es un funtor, Art. XV):**

```
PROJECT preserva identidad:
  PROJECT_E(id_G) = id_E

PROJECT preserva composición:
  PROJECT_E(T₂ ∘ T₁) = PROJECT_E(T₂) ∘ PROJECT_E(T₁)
```

Probado incondicionalmente para los 3 contratos en PAPER_V7 y
THEOREM3_PROOF_V1.md. Ver fuentes normativas al inicio de este archivo.

**Condición de invariancia (diseño del lenguaje, no asunción externa):**
Las transformaciones COMPILE y EXPORT son, por definición del lenguaje,
transformaciones que tienen DIM_RUST y DIM_SQL como subspacios invariantes
respectivamente (T(DIM_E) ⊆ DIM_E). Esta es una consecuencia de la
especificación, no una hipótesis adicional.

---

## 3. Reglas de evaluación — Capa de implementación (Art. IV.2)

Las primitivas ATTEND y RECUR son **primitivas opacas** desde la perspectiva
de las reglas del núcleo algebraico. Esto significa:

- El núcleo no necesita saber *cómo* ATTEND calcula su resultado, solo que
  produce un nuevo estado a partir de uno dado.
- Las reglas big-step para ATTEND y RECUR se derivan directamente de sus
  ecuaciones de definición (Art. IV.2).

### 3.1 ATTEND(Q, K, V, s) — Atención cruzada (arquitectura transformer)

**Definición:**
```
ATTEND(Q, K, V, s) = softmax( Q·K^T / √d ) · V
```
donde Q, K, V son las matrices de query/key/value y d es la dimensión de K.

**Regla big-step (primitiva opaca):**
```
result = softmax(Q·K^T / √d) · V
──────────────────────────────────────────────────────────
⟨s, ATTEND(Q, K, V, s)⟩ ⇒ s_nuevo  donde s_nuevo.V = result
```

**Nota constitucional (Regla R11):**
ATTEND pertenece a la capa de implementación, no a la capa algebraica
invariante. Una VM mínima puede omitir ATTEND e implementar solo las 4
primitivas algebraicas (Sec. 2). La VM completa debe implementar ATTEND
para ejecutar programas que lo usen.

**Extensión laxo-monoidal (no vinculante):**
La no-linealidad del softmax impide que ATTEND sea un funtor estricto en el
mismo sentido que PROJECT. ATTEND es un funtor *laxo*: preserva composición
solo hasta una transformación de ajuste η: F(T₂)∘F(T₁) → F(T₂∘T₁).
Esta caracterización es ⚙️ MECANISMO en desarrollo — la formalización completa
requiere un suplementario independiente (ver THEOREM3_PROOF_V1.md Parte V
para la caracterización actual).

---

### 3.2 RECUR(A, B, C, h, x) — Recurrencia SSM/Mamba

**Definición:**
```
h(t+1) = A·h(t) + B·x(t)    [actualización del estado oculto]
y(t)   = C·h(t)              [proyección a la salida]
```
donde A, B, C son matrices de parámetros, h es el estado oculto, x es
la entrada en el paso t.

**Regla big-step (primitiva opaca, para un paso t):**
```
h_nuevo = A·h(t) + B·x(t)
y       = C·h_nuevo
──────────────────────────────────────────────────────────────────
⟨(s, h(t), x(t)), RECUR(A,B,C)⟩ ⇒ (s_nuevo, h_nuevo, y)
```

Para una secuencia de T pasos: aplicar la regla anterior T veces.

**Relación con FIXPOINT:**
RECUR convergido hacia un estado estacionario puede expresarse como
FIXPOINT de la transformación h ↦ A·h + B·x (si A es contracción, i.e.,
spectral_radius(A) < 1). En ese caso:
```
lim_{t→∞} h(t) = (I - A)^{-1} · B · x   [si ‖A‖ < 1]
```

**Nota constitucional (Regla R11):**
RECUR pertenece a la capa de implementación. Se incluye porque la atención
softmax tiene limitaciones teóricas documentadas (no puede satisfacer
simultáneamente composición monoidal estricta y descenso a cocientes lógicos
no triviales), y RECUR aborda esos casos de lógica profunda y composición
secuencial muy larga que ATTEND solo no puede garantizar.

---

## 4. Semántica denotacional — Orientación (🔬 INVESTIGACIÓN ABIERTA)

🔬 **INVESTIGACIÓN — explícitamente NO vinculante (Constitución V6 Art. V.3,
Regla R13). Registrada para continuidad, no para ser implementada ahora.**

La semántica operacional big-step de las secciones 2 y 3 define *cómo*
evalúa un programa POLYDIM. La semántica denotacional define *qué significa*
una transformación T como objeto matemático independiente de la evaluación.

**El operador ⟦·⟧ a definir:**
```
⟦T⟧ : T → ?
```
donde `?` es la categoría de objetos matemáticos que captura la semántica
de T. Las candidatas son:

**Candidata D1 — Operador lineal acotado (caso algebraico puro):**
```
⟦T⟧ ∈ L(R^N, R^N)    (operadores lineales acotados sobre R^N)
⟦COMPOSE(T₁,T₂)⟧ = ⟦T₂⟧ ∘ ⟦T₁⟧
⟦MIX(α,T₁,β,T₂)⟧ = α·⟦T₁⟧ + β·⟦T₂⟧
⟦FIXPOINT(T,ε)⟧ = punto fijo de ⟦T⟧ (si existe y es único — Banach)
⟦PROJECT_E(T)⟧ = π_E ∘ ⟦T⟧   [el funtor F_E de Teorema 3]
```
Esta es la candidata más sencilla y la más cercana a lo ya demostrado.
**Gap:** ATTEND involucra softmax (no lineal) y RECUR tiene productos
estado-dependientes — ninguno encaja limpiamente en L(R^N, R^N).

**Candidata D2 — Morfismo en Para(Vect) (caso general, incluyendo no-lineal):**
La 2-categoría Para(Vect) tiene como morfismos funciones parametrizadas
f: A × P → B donde P es el espacio de parámetros. ATTEND y RECUR encajan
aquí naturalmente: sus matrices Q,K,V y A,B,C son los parámetros.
```
⟦ATTEND(Q,K,V)⟧ ∈ Para(Vect)(R^N, R^N)   [morfismo parametrizado]
⟦RECUR(A,B,C)⟧  ∈ Para(Vect)(R^N×R^N, R^N)
```
Esta candidata aparece en la Constitución V6 modificada que existe en
docs_v6/ (no canónica) — su formalización requiere trabajo de Track 1.

**Tarea concreta para quien tome esta investigación:**
1. Demostrar que D1 funciona para la capa algebraica (4 primitivas).
2. Caracterizar el gap con ATTEND: ¿es un funtor laxo? ¿un morfismo en
   Para(Vect)? ¿algo más?
3. Si se adopta D2, revisar si Teorema 3 sigue siendo válido en Para(Vect).
4. Producir SPEC_SEMANTICA_DENOTACIONAL_V0.md cuando haya suficiente claridad.

**Status de la Constitución:** Art. V.3 marca esto explícitamente como
investigación abierta con la etiqueta 🔬. La semántica operacional de
las secciones 2 y 3 es suficiente para la VM y el paper — la denotacional
es el siguiente nivel de rigor para cs.PL.

---

## 5. Resumen operativo para la VM Rust (TASK_023/TASK_D)

```
Primitiva         Regla de evaluación              Costo (LoRA, rango r)
─────────────────────────────────────────────────────────────────────────
COMPOSE(T₁,T₂)   T₂(T₁(s))                       O(N·r) por aplicación
MIX(α,T₁,β,T₂)  α·T₁(s) + β·T₂(s)              O(N·r) por término
FIXPOINT(T,ε)    T^n(s) hasta ‖Δs‖ < ε           O(N·r·n) para n iters
PROJECT_E(T)     F_E(T)(s) según contrato E       O(N·r) + costo del executor
ATTEND(Q,K,V,s)  softmax(QK^T/√d)·V              O(N²) para atención densa
RECUR(A,B,C,h,x) A·h + B·x                       O(N²) si A,B,C densas
```

**Invariante que la VM DEBE preservar tras cualquier operación:**
```
dist(T(GEO_ID), GEO_ID) < ε_geo   [Regla R10, Teorema 5]
```
Esto se verifica con los tests BUG_002 del conjunto `polydim_tests.py`
(fileId 1FTNK7eBNHjIoc8Z1yoqvCkU6IXVYeBGX, 29/29 siempre deben pasar).

---

## 6. Referencias cruzadas normativas

| Concepto | Fuente canónica |
|---|---|
| Definiciones S, P, T | Constitución V6 Art. V.1 (fileId 1LneDD0D8fHREiZN-_Xw-Ax9AzxfF6tKv) |
| Big-step COMPOSE/MIX/FIXPOINT/PROJECT | Constitución V6 Art. V.2 |
| Dos capas de primitivas | Constitución V6 Art. IV.1 y IV.2 |
| ATTEND y RECUR como opacas | Constitución V6 Art. IV.2 + Regla R11 |
| Teorema 1 (asoc. COMPOSE) | Constitución V6 Art. XV |
| Teorema 2 (lineal. MIX) | Constitución V6 Art. XV |
| Teorema 3 (PROJECT funtor) | Constitución V6 Art. XV + THEOREM3_PROOF_V1.md (1W41O6eNTKIjLHPoswBRVfgmNQDQZIC32) |
| Teorema 4 (Banach, FIXPOINT) | Constitución V6 Art. XV |
| Teorema 5 (GEO_ID invariante) | Constitución V6 Art. XV + Regla R10 |
| Proposición 6.1 (Flutter ISO) | PROPOSITION_6_1_PROOF_V1.md (1OhQpZZ0md-NygRcxXSa2bcU3LbfpNgzu) |
| Subspace Commutativity Lemma | PAPER_V7 Sec 2.5 (1nKKcnluDM3PuUgJI_KcF5Hf2oW1JNmo8) |
| Formato binario LoRA | SPEC_FORMATO_BINARIO_V0.md (1JWIZH2AsKr8vCQhg6t4JE_EOUk3P4WIE) |
| Semántica denotacional (abierta) | Constitución V6 Art. V.3 (🔬) |

---

*SPEC_SEMANTICA_OPERACIONAL_V0.md · TASK_031 · 2026-06-27 · Claude Sonnet 4.6*
*Fuentes: Constitución V6 Art. IV+V + PAPER_V7 Sec 2 + pruebas suplementarias*
