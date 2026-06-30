# POLYDIM_DEST
# destino: polydim_v1/spec/
# nombre: SPEC_SEMANTICA_OPS_V0.md
# autor: claude-sonnet-4-6 (curso03.mithril@gmail.com)
# fecha: 2026-06-28
# tarea: TASK_031 — Semántica operacional completa (las 6 primitivas)
# fuente-primaria: POLYDIM_CONSTITUCION_V6.md (fileId 1LneDD0D8fHREiZN-_Xw-Ax9AzxfF6tKv)
# lock: LOCK_TASK_031.md (locks/, curso03.mithril, 2026-06-27)

---

# POLYDIM — Semántica Operacional Completa V0

> **Propósito**: este documento completa el Artículo V de la Constitución V6 en dos
> direcciones que allí quedan explícitamente incompletas:
>
> 1. Las reglas big-step para **ATTEND** y **RECUR** como "primitivas opacas" — el
>    Artículo V.2 dice que existen pero no las escribe.
> 2. Un marco de semántica **denotacional** ⟦T⟧ para las 6 primitivas — el Artículo
>    V.3 lo marca como 🔬 abierto e identifica como TASK_A (Artículo XIX).
>
> El documento usa las tres marcas de la Constitución V6:
> ⚔️ LEY — normativo, derivado de Art. V de V6.
> ⚙️ MECANISMO — especificación técnica en implementación.
> 🔬 INVESTIGACIÓN — propuesta formal, válida como hipótesis, NO vinculante.
>
> Este documento es de tipo ⚙️/🔬: no modifica la Constitución. Consolida
> lo que V6 Art. V establece y extiende donde V6 explícitamente lo permite.

---

## 1. Punto de partida: lo que V6 Art. V.2 ya establece

⚔️ **LEY** — estas cuatro reglas son la ley vigente. Se transcriben aquí
para que este documento sea autónomo, pero la fuente canónica es V6 Art. V.2.

```
Regla COMPOSE:

    ⟨s, T₁⟩ ⇒ s₁        ⟨s₁, T₂⟩ ⇒ s₂
    ──────────────────────────────────────
    ⟨s, COMPOSE(T₁, T₂)⟩ ⇒ s₂


Regla MIX:

    ────────────────────────────────────────────────────────
    ⟨s, MIX(α₁, T₁, α₂, T₂)⟩ ⇒ α₁·T₁(s) + α₂·T₂(s)


Regla FIXPOINT:

    s_{k+1} = T(s_k)        ‖s_{k+1} − s_k‖ < ε
    ──────────────────────────────────────────────
    ⟨s_k, FIXPOINT(T, ε)⟩ ⇒ s_{k+1}


Regla PROJECT (esquema general, instanciada por COMPILE/RENDER/EXPORT):

    F = funtor asociado al executor E
    ──────────────────────────────────────────────────────────
    ⟨s, PROJECT(T, E)⟩ ⇒ F(T)(s)
```

Nota: el Artículo V.2 cierra con: "ATTEND y RECUR (capa de implementación)
tienen sus propias reglas, derivadas directamente de sus ecuaciones de definición
en el Artículo IV.2, y se tratan como 'primitivas opacas' desde el punto de
vista de estas reglas". Este documento provee esas reglas.

---

## 2. Reglas big-step para ATTEND y RECUR como primitivas opacas

⚙️ **MECANISMO** — derivadas directamente de las ecuaciones de definición
del Artículo IV.2 de la Constitución V6. Son ley en la capa de implementación.

### 2.1 Semántica de estado extendido para la capa de implementación

Para modelar ATTEND y RECUR con reglas operacionales, el estado S = (V, D, A)
del Artículo V.1 se extiende localmente con el contexto de la capa:

```
S_impl = (V, D, A, h, ctx)

donde:
  V   ∈ R^N       vector de posición (igual que S)
  D               conjunto de subespacios visibles (igual que S)
  A               activaciones (igual que S)
  h   ∈ R^d_h    estado de memoria recurrente para RECUR (inicialmente cero)
  ctx             contexto de atención para ATTEND (K, V_attn matrices)
```

Esta extensión es estrictamente local a la capa de implementación (Art. IV.2).
El núcleo algebraico (Art. IV.1) opera sobre S, no sobre S_impl.

### 2.2 Regla ATTEND

De la definición del Artículo IV.2:
```
ATTEND(Q, K, V_attn, s) = softmax( Q·K^T / √d ) · V_attn
```

La regla big-step es:

```
Regla ATTEND:

    Q = W_Q · V        K = W_K · V        V_attn = W_V · V
    α_attn = softmax( Q · K^T / √d )
    V_out = α_attn · V_attn
    s' = (V + V_out, D, A, h, ctx)        [actualización residual]
    ──────────────────────────────────────────────────────────────
    ⟨s, ATTEND(W_Q, W_K, W_V)⟩ ⇒ s'
```

**Por qué "primitiva opaca"**: la semántica del núcleo algebraico no necesita
saber que el resultado involucra softmax ni matrices Q, K, V. Solo necesita
saber que ATTEND: S_impl → S_impl es una función total que produce un nuevo
estado a partir del estado dado. La forma interna es responsabilidad de la capa
de implementación y puede ser reemplazada sin tocar el núcleo (Art. XVII).

**Precondición de bienestar**: d > 0 (dimensión del subespacio de atención).
Si d = 0, ATTEND es la identidad: s' = s.

### 2.3 Regla RECUR

De la definición del Artículo IV.2:
```
RECUR:  h(t+1) = A·h(t) + B·x(t)
        y(t)   = C·h(t)
```

La regla big-step es:

```
Regla RECUR:

    x = V                              [entrada: vector de posición actual]
    h' = A · h + B · x                [actualización del estado recurrente]
    y  = C · h'                        [proyección al espacio de salida]
    V' = V + y                         [actualización residual del estado]
    s' = (V', D, A_act, h', ctx)
    ──────────────────────────────────────────────────────────────────────
    ⟨s, RECUR(A, B, C)⟩ ⇒ s'
```

donde A_act son las activaciones del estado (para no confundir notación
con la matriz A de RECUR, que es la matriz de recurrencia).

**Por qué "primitiva opaca"**: igual que ATTEND. El núcleo solo ve
RECUR: S_impl → S_impl. La presencia de estado recurrente h es interna.

**Precondición de bienestar**: A debe tener radio espectral ρ(A) < 1 para
garantizar estabilidad de la secuencia recurrente. Si no se verifica, RECUR
puede diverger. Esta condición es responsabilidad del programador (igual que
la condición de contracción de FIXPOINT, Art. XV, Teorema 4).

### 2.4 Las seis reglas en un mismo cuadro

```
⚔️ Capa algebraica (invariante):
┌─────────────────────────────────────────────────────────────────┐
│ COMPOSE(T₁, T₂)  : s  →  T₂(T₁(s))                           │
│ MIX(α₁,T₁,α₂,T₂): s  →  α₁·T₁(s) + α₂·T₂(s)                │
│ FIXPOINT(T, ε)   : s  →  T^n(s)  hasta convergencia           │
│ PROJECT(T, E)    : s  →  F_E(T)(s)   [F_E = funtor executor]  │
└─────────────────────────────────────────────────────────────────┘

⚙️ Capa de implementación (reemplazable, Art. XVII):
┌─────────────────────────────────────────────────────────────────┐
│ ATTEND(W_Q,W_K,W_V): s → s + softmax(QK^T/√d)·V_attn         │
│ RECUR(A, B, C)      : s → s + C·(A·h + B·V)  [h actualizado] │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Marco de semántica denotacional ⟦T⟧

🔬 **INVESTIGACIÓN** — lo que sigue es una propuesta formal, validable como
hipótesis. El Artículo V.3 de V6 lo marca explícitamente como tarea abierta
(TASK_A). Este documento NO cierra esa tarea: propone el marco y lo deja
listo para que una sesión futura escriba las demostraciones formales.

**Por qué hace falta**: la semántica operacional del §2 dice *cómo* evalúa cada
primitiva, pero no *qué objeto matemático es* cada transformación. Un paper de
cs.PL necesita ambas. La semántica denotacional ⟦T⟧ mapea cada programa al
objeto matemático que denota, independientemente de la estrategia de evaluación.

### 3.1 El operador ⟦·⟧

Se propone el operador de interpretación:

```
⟦·⟧ : Expr_POLYDIM → (S → S)
```

que mapea cada expresión del lenguaje a una función de estados. Para que
⟦·⟧ sea una semántica denotacional válida debe satisfacer:

```
(Composicionalidad)  ⟦COMPOSE(T₁, T₂)⟧(s) = ⟦T₂⟧(⟦T₁⟧(s))
(Correctitud)        si ⟨s, T⟩ ⇒ s'  entonces  ⟦T⟧(s) = s'
```

La segunda propiedad es el teorema de adecuación: la semántica denotacional
es correcta respecto de la operacional. Su demostración requiere verificar
cada regla del §2.

### 3.2 Interpretación denotacional de la capa algebraica

```
⟦COMPOSE(T₁, T₂)⟧  =  ⟦T₂⟧ ∘ ⟦T₁⟧

⟦MIX(α₁, T₁, α₂, T₂)⟧(s)  =  α₁ · ⟦T₁⟧(s) + α₂ · ⟦T₂⟧(s)

⟦FIXPOINT(T, ε)⟧  =  lím_{n→∞} ⟦T⟧^n       [si T es contracción]

⟦PROJECT(T, E)⟧  =  F_E ∘ ⟦T⟧
```

**Observación sobre linealidad**: si T₁ y T₂ son lineales,
⟦COMPOSE(T₁, T₂)⟧ es la composición de operadores lineales (producto matricial).
⟦MIX(α₁, T₁, α₂, T₂)⟧ es combinación lineal de operadores lineales, también lineal.
El lenguaje preserva linealidad si todas las primitivas son lineales — propiedad
relevante para análisis de complejidad y para la capa LoRA (Art. XI).

**FIXPOINT como límite de una sucesión en B(S)**: si ⟦T⟧ es una contracción en
el espacio de Banach (S, ‖·‖), el Teorema 4 de V6 garantiza que el límite existe
y es único. La semántica denotacional de FIXPOINT es ese punto fijo s* = ⟦T⟧(s*).

### 3.3 Interpretación denotacional de la capa de implementación

El tratamiento de ATTEND y RECUR como "primitivas opacas" en la semántica
operacional (§2) tiene su contraparte denotacional:

```
⟦ATTEND(W_Q, W_K, W_V)⟧(s)  =  s + f_attn(s; W_Q, W_K, W_V)
```

donde f_attn: S → R^N es la función de atención. f_attn no es lineal (contiene
softmax), por lo que ⟦ATTEND⟧ pertenece a la clase de operadores no lineales
acotados de Lipschitz si los pesos son acotados — esto es verificable y es una
condición suficiente para que FIXPOINT(ATTEND, ε) converja cuando la constante
de Lipschitz L < 1.

```
⟦RECUR(A, B, C)⟧(s)  =  s + C·(A·h(s) + B·V(s))
```

donde h(s) es el estado recurrente del contexto. ⟦RECUR⟧ es lineal en s si
A, B, C son fijas (lo que ocurre cuando los pesos están fijos, caso del programa
ya compilado). La dependencia de h introduce historia — ⟦RECUR⟧ es un operador
con memoria en el sentido de los sistemas de control lineales.

### 3.4 Teorema de adecuación (enunciado, demostración pendiente)

🔬 **Conjetura formal — requiere demostración rigurosa para convertirse en ley.**

> **Teorema de Adecuación (TASK_031 — pendiente):**
> Para toda expresión T y todo estado s, si ⟨s, T⟩ ⇒ s' según las reglas
> del §2, entonces ⟦T⟧(s) = s'.

La demostración procedería por inducción estructural sobre T:
- Caso COMPOSE: inmediato por definición de ⟦COMPOSE⟧ = ⟦T₂⟧ ∘ ⟦T₁⟧.
- Caso MIX: por linealidad de la interpretación.
- Caso FIXPOINT: requiere el Teorema de Banach (T contracción → límite único).
- Caso PROJECT: requiere mostrar que F_E es efectivamente un funtor (Teorema 3,
  ya demostrado en PAPER_V8 para los tres contratos COMPILE/RENDER/EXPORT).
- Caso ATTEND: f_attn es continua y acotada → ⟦ATTEND⟧ está bien definido.
- Caso RECUR: si ρ(A) < 1, la serie de potencias de A converge → ⟦RECUR⟧ definido.

**Estado**: los casos algebraicos (COMPOSE, MIX, FIXPOINT, PROJECT) son
estándar y sus pruebas son cortas. Los casos ATTEND y RECUR dependen de
propiedades de continuidad de softmax y de la condición espectral ρ(A) < 1.
Ninguno requiere matemática nueva — solo escribir la prueba formal.

### 3.5 Consecuencia para el paper

Si el Teorema de Adecuación se demuestra en una sesión futura, el paper gana:

1. **Sección de semántica formal completa**: operacional (§2) + denotacional (§3) +
   adecuación (§3.4) — el trifecta que cs.PL exige para un lenguaje serio.
2. **Conexión con Teorema 3**: PROJECT es funtor (Teorema 3, ya en PAPER_V8) →
   la adecuación del caso PROJECT es un corolario, no una prueba nueva.
3. **Respuesta a la sección "Amenazas a la Validez" (TASK_032)**: la condición
   ρ(A) < 1 para RECUR y L < 1 para ATTEND son amenazas identificadas y acotadas,
   no omisiones — fortalece la posición del paper ante revisores de cs.PL.

---

## 4. Registro de lo que queda abierto

| # | Item abierto | Estado | Quién debe cerrarlo |
|---|---|---|---|
| O1 | Demostración formal del Teorema de Adecuación | 🔬 pendiente | Sesión futura (TRACK 1) |
| O2 | Verificación empírica: medir L (Lipschitz) de ATTEND sobre bootstrap V0.3 | ⚙️ pendiente | TRACK 3 (primitivas) |
| O3 | Condición ρ(A) < 1 para RECUR en programas reales | ⚙️ pendiente | TRACK 3 |
| O4 | Extensión de ⟦·⟧ a la sintaxis de superficie (cuando exista, Art. XII.5) | ⚙️ futuro | TRACK 1 post-sintaxis |

---

## 5. Cómo usar este documento

Este archivo vive en `spec/` y complementa directamente a:

- **POLYDIM_CONSTITUCION_V6.md** (Art. V) — fuente canónica de las 4 reglas algebraicas
- **POLYDIM_PAPER_V8.md** (Sec. 2.5, Teorema 3) — prueba de funtorialidad de PROJECT
- **POLYDIM_THEOREM3_PROOF_V1.md** — prueba detallada de Teorema 3

Para **demostrar el Teorema de Adecuación** (O1 arriba): leer este archivo,
leer V6 Art. V completo, y escribir la prueba por inducción estructural.
El caso PROJECT ya está resuelto por PAPER_V8 §2.5.

Para **escribir la sección "Amenazas a la Validez" del paper** (TASK_032):
las condiciones ρ(A) < 1 y L < 1 identificadas en §3.3 son el material
central de esa sección — no son debilidades del lenguaje sino precondiciones
explícitas y verificables.

---

*SPEC_SEMANTICA_OPS_V0.md · 2026-06-28 · TASK_031 · claude-sonnet-4-6 (curso03.mithril)*
*Estado: DRAFT — Adecuación pendiente de demostración formal (ver O1)*
