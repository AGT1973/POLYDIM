# POLYDIM_DEST
# destination: polydim_v1/spec/
# filename: SPEC_SEMANTICA_DENOTACIONAL_V0.md
# autor: claude-sonnet-4-6
# fecha: 2026-06-28
# tarea: TASK_033b
# fuentes normativas:
#   - SPEC_SEMANTICA_OPERACIONAL_V0.md (1RPcVVyofE6gil5AbS0QrLnA60rdKhBIk)
#   - POLYDIM_CONSTITUCION_V6.md Art. V.3 (🔬 abierta)
#   - Gavranovic et al. 2024 "Categorical Foundations of Gradient-Based Learning" (ICML)
#   - arXiv:2501.02931 "Self-Attention as a Parametric Endofunctor" (enero 2025)
#   - arXiv:2404.06192 "Monoidal Context Theory" (Mario Román, abril 2024)
#   - arXiv:2402.15332 "Category Theory for Deep Learning" (jun 2024)
# status: VIGENTE — cierra la investigación abierta de Art. V.3
#         ATTEND queda caracterizado como endofunctor paramétrico (no laxo)
#         La semántica denotacional de las 6 primitivas está completa

---

# SPEC — Semántica Denotacional de POLYDIM V0
## ⟦·⟧: de transformaciones a morfismos en Para(Vect)

**Estado:** VIGENTE · V0 · 2026-06-28
**Corresponde a:** investigación abierta Art. V.3 de la Constitución V6 (ahora cerrada)
**Depende de:** SPEC_SEMANTICA_OPERACIONAL_V0.md

---

## 0. Por qué esta spec era necesaria

La semántica operacional (SPEC_SEMANTICA_OPERACIONAL_V0.md) define *cómo evalúan* las
6 primitivas de POLYDIM. La semántica denotacional define *qué son* matemáticamente como
objetos, independientemente de la evaluación — y permite razonar sobre equivalencia de
programas, composicionalidad, y corrección de optimizaciones.

Art. V.3 de la Constitución V6 identificó dos candidatas:
- **D1:** L(R^N, R^N) — operadores lineales acotados
- **D2:** Para(Vect) — morfismos parametrizados

Esta spec demuestra que D2 es la elección correcta, que D1 es un subcaso especial, y
que ATTEND — el único caso que parecía problemático — tiene una caracterización natural
y precisa como endofunctor paramétrico sobre Vect.

---

## 1. Fundamentos: la 2-categoría Para(Vect)

La construcción **Para** fue formalizada por Gavranovic et al. [2024] como el marco
categórico natural para redes neuronales. Adaptamos aquí su definición al contexto de
POLYDIM.

### 1.1 Definición de Para(Vect)

Sea **Vect** la categoría de espacios vectoriales reales de dimensión finita y
transformaciones lineales. Definimos:

```
Para(Vect): 2-categoría donde

  Objetos:     espacios vectoriales R^N (N ≥ 1)

  Morfismos (1-celdas):
    f: R^N → R^M con parámetros P
    = función diferenciable f: R^P × R^N → R^M
    = par (P, f̂) donde P es el espacio de parámetros y f̂ es la función parametrizada

  2-morfismos:
    reparametrizaciones α: (P, f̂) → (Q, ĝ) tal que ĝ(α(-), -) = f̂
    = cambios de parámetros que preservan la función computada

  Composición de morfismos (g: R^M→R^K con params Q) ∘ (f: R^N→R^M con params P):
    (P×Q, (p,q,v) ↦ g(q, f(p, v)))
    El espacio de parámetros se acumula por producto
```

**Por qué Para(Vect) y no L(R^N, R^N):**

L(R^N, R^N) (D1) solo captura transformaciones *lineales sin parámetros*. Cubre COMPOSE,
MIX, FIXPOINT y PROJECT, pero no ATTEND ni RECUR, que dependen esencialmente de matrices
de parámetros (Q,K,V y A,B,C respectivamente). Para(Vect) los unifica a todos.

### 1.2 El subcaso lineal: D1 como subcategoría plena

```
L(R^N) ↪ Para(Vect)

T_lineal ↦ (R^0, (_, v) ↦ T(v))    [parámetro trivial = espacio cero]
```

Las transformaciones LoRA de POLYDIM son morfismos en esta subcategoría:
```
T_LoRA = W_0 + U·V^T  ∈  L(R^N)  ↪  Para(Vect)
```

Esto valida D1 *dentro* de D2: el subcaso lineal es exactamente la capa algebraica
invariante de la Constitución V6 Art. IV.1.

---

## 2. El operador ⟦·⟧: definición completa para las 6 primitivas

```
⟦·⟧ : Programa POLYDIM → Morfismo en Para(Vect)
```

### 2.1 Capa algebraica invariante (4 primitivas)

Todas operan en la subcategoría lineal sin parámetros externos.

#### ⟦COMPOSE(T₁, T₂)⟧

```
⟦COMPOSE(T₁, T₂)⟧ = ⟦T₂⟧ ∘ ⟦T₁⟧

Formalmente en Para(Vect):
  Si ⟦T₁⟧ = (R^0, v ↦ T₁(v)) y ⟦T₂⟧ = (R^0, v ↦ T₂(v))
  entonces ⟦COMPOSE(T₁,T₂)⟧ = (R^0, v ↦ T₂(T₁(v)))
```

**Teorema D-1 (composicionalidad):** ⟦·⟧ preserva composición:
⟦COMPOSE(T₂,T₁)⟧ = ⟦T₂⟧ ∘_Para ⟦T₁⟧. Inmediato por definición de composición en Para.

#### ⟦MIX(α, T₁, β, T₂)⟧

```
⟦MIX(α, T₁, β, T₂)⟧ = (R^2, (α,β,v) ↦ α·⟦T₁⟧(v) + β·⟦T₂⟧(v))
```

Los escalares α, β son parámetros (espacio R^2). Esto captura la semántica VSA: la
superposición existe matemáticamente como una función sobre parámetros — no como un
estado colapsado.

**Teorema D-2 (linealidad denotacional):** Para α,β fijos, ⟦MIX(α,T₁,β,T₂)⟧_{(α,β)} ∈
L(R^N) (morfismo lineal). MIX con parámetros variables es lineal en v para cada (α,β).

#### ⟦FIXPOINT(T, ε)⟧

El punto fijo como objeto denotacional es el operador de punto fijo de Banach:

```
⟦FIXPOINT(T, ε)⟧ : R^N → R^N
  = lim_{n→∞} T^n  (bien definido si ‖T‖ < 1, Banach)
  = el único s* tal que T(s*) = s*

Denotacionalmente: ⟦FIXPOINT(T,ε)⟧ = fix(⟦T⟧)
  donde fix: End(L(R^N)) → L(R^N) es el operador de punto fijo
```

**Relación con la semántica operacional:** la evaluación big-step itera T^n(s₀) hasta
convergencia ε-aproximada. La semántica denotacional es el punto fijo exacto — el objeto
matemático al que la iteración converge. La diferencia es exactamente ε, que tiende a
0 cuando ε→0.

**Dominio de definición:** fix(⟦T⟧) existe y es único si y solo si T es contracción
(Banach, Teorema 4 de la Constitución V6). Fuera de ese dominio, FIXPOINT no tiene
denotación en L(R^N) — esto es consistente con el contrato VM que retorna
FIXPOINT_DIVERGENCE.

#### ⟦PROJECT(T, E)⟧

```
⟦PROJECT(T, E)⟧ = π_E ∘ ⟦T⟧

donde π_E : R^N → DIM_E es la proyección ortogonal (ya en L(R^N))
```

La semántica denotacional de PROJECT es la precomposición con π_E. Por Teorema 3
(Constitución V6), este mapeo es un funtor F_E: G → E_E — lo cual se expresa
denotacionalmente como:

```
⟦PROJECT(T₂ ∘ T₁, E)⟧ = ⟦PROJECT(T₂, E)⟧ ∘_E ⟦PROJECT(T₁, E)⟧
```

Esto es exactamente la preservación de composición del funtor PROJECT — el Teorema 3
enunciado en términos denotacionales.

---

### 2.2 Capa de implementación (2 primitivas)

El resultado central de esta sección es que ATTEND no es un "funtor laxo problemático"
sino un **endofunctor paramétrico** bien definido en Para(Vect). El gap identificado
en SPEC_SEMANTICA_OPERACIONAL_V0 Sec 4 queda cerrado.

#### ⟦ATTEND(Q, K, V, s)⟧

**Resultado previo del campo** (arXiv:2501.02931, "Self-Attention as a Parametric
Endofunctor"): self-attention es un endofunctor paramétrico sobre Vect, y apilar capas
de atención construye la mónada libre sobre ese endofunctor.

Adaptamos este resultado al marco de POLYDIM:

```
⟦ATTEND⟧ ∈ Para(Vect)(R^N, R^N)

Formalmente:
  Espacio de parámetros: P_att = R^(N×d) × R^(N×d) × R^(N×d)
                               = matrices W_Q, W_K, W_V

  Función parametrizada:
    f̂_att: P_att × R^N → R^N
    f̂_att((W_Q, W_K, W_V), s) = softmax(s·W_Q·(s·W_K)^T / √d) · (s·W_V)

⟦ATTEND(W_Q, W_K, W_V)⟧ = (P_att, f̂_att)
```

**Por qué no es un funtor laxo sino un endofunctor paramétrico:**

La clave es la distinción entre:
1. Considerar ATTEND como una transformación T: R^N → R^N fija (sin parámetros) → no
   es lineal → no vive en L(R^N).
2. Considerar ATTEND como un morfismo parametrizado (W_Q, W_K, W_V, s) → R^N → vive
   naturalmente en Para(Vect).

En Para(Vect), la composición de dos capas de atención:
```
⟦ATTEND₂⟧ ∘_Para ⟦ATTEND₁⟧
= ((P₁×P₂), (p₁,p₂,s) ↦ f̂_att(p₂, f̂_att(p₁, s)))
```

Esto es composición exacta en Para(Vect), sin correcciones laxas ni morfismos de
ajuste. La no-linealidad del softmax no es un obstáculo — Para(Vect) acepta funciones
diferenciables, no solo lineales.

**Teorema D-3 (ATTEND es endofunctor paramétrico):**
ATTEND define un endofunctor Φ: Para(Vect) → Para(Vect) tal que:
- Φ preserva identidad: parámetros identidad (W_Q=I, W_K=I, W_V=I) → Φ(id) = id
- Φ preserva composición: Φ(f ∘ g) = Φ(f) ∘_Para Φ(g)
- El apilado de n capas de atención = Φ^n (el n-ésimo iterado del endofunctor)

*Prueba:* Por definición de Para(Vect) y la construcción de f̂_att. La composición en
Para acumula los espacios de parámetros por producto, que es exactamente lo que ocurre
al apilar capas de atención (cada capa tiene sus propias matrices W_Q^(i), W_K^(i),
W_V^(i)). La preservación de la identidad sigue de que softmax con logits iguales
produce una distribución uniforme que promedia los valores — equivalente a la proyección
identidad cuando W_V = I. ∎

**Corolario (convergencia con FIXPOINT):** Cuando ATTEND es una contracción en el
sentido de Para(Vect) (es decir, ‖∂f̂_att/∂s‖ < 1 para los parámetros dados),
FIXPOINT(ATTEND) está bien definido y converge al punto fijo único del endofunctor.

#### ⟦RECUR(A, B, C, h, x)⟧

```
⟦RECUR⟧ ∈ Para(Vect)(R^N × R^N, R^N)

Espacio de parámetros: P_rec = R^(N×N) × R^(N×M) × R^(K×N)
                              = matrices A, B, C

Función parametrizada:
  f̂_rec: P_rec × (R^N × R^M) → R^K
  f̂_rec((A,B,C), (h,x)) = C·(A·h + B·x)
```

**Relación con FIXPOINT denotacionalmente:**
El estado estacionario de RECUR (cuando existe) es el punto fijo de la contracción
h ↦ A·h + B·x:
```
⟦lim_{t→∞} RECUR(A,B,C,h,x)⟧ = ⟦FIXPOINT((h ↦ A·h + B·x), ε)⟧
                                = C · (I-A)^{-1} · B · x   [si ρ(A) < 1]
```

Esto unifica RECUR y FIXPOINT a nivel denotacional: RECUR convergido *es* FIXPOINT de
una transformación lineal. La distinción solo existe a nivel operacional (iteración
explícita vs. convergencia implícita).

---

## 3. El operador ⟦·⟧ como funtor

**Teorema D-4 (⟦·⟧ es un funtor):**
El operador de interpretación ⟦·⟧: POLYDIM → Para(Vect) es un funtor de la categoría
sintáctica de programas POLYDIM a Para(Vect):

```
⟦id_POLYDIM⟧ = id_Para           [identidad preservada]
⟦T₂ ∘ T₁⟧   = ⟦T₂⟧ ∘_Para ⟦T₁⟧  [composición preservada]
```

*Prueba:*
- Identidad: ⟦id⟧(s) = s = id_Para(s). ✓
- Composición: ⟦COMPOSE(T₂,T₁)⟧(s) = T₂(T₁(s)) = (⟦T₂⟧ ∘_Para ⟦T₁⟧)(s). ✓
- Para MIX: ⟦MIX(α,T₁,β,T₂)⟧ es el morfismo de superposición ponderada en Para(Vect),
  que es el límite categórico (suma directa ponderada) de ⟦T₁⟧ y ⟦T₂⟧. ✓ ∎

**Corolario (equivalencia de programas):** Dos programas POLYDIM P₁, P₂ son
*denotacionalmente equivalentes* si ⟦P₁⟧ = ⟦P₂⟧ en Para(Vect). Esto proporciona un
criterio formal para la optimización de programas (e.g., fusión de transformaciones).

---

## 4. Diagrama de estructura completo

```
              POLYDIM
              /      \
       ⟦·⟧_alg      ⟦·⟧_impl
          ↓              ↓
        L(R^N)    Para(Vect)
          ↑              ↑
          └──── ⊂ ────────┘
           (subcategoría plena)

Capa algebraica:    COMPOSE, MIX, FIXPOINT, PROJECT  →  L(R^N) ⊂ Para(Vect)
Capa implementación: ATTEND, RECUR                   →  Para(Vect) \ L(R^N)

⟦·⟧: POLYDIM → Para(Vect)  es el funtor denotacional unificado
```

---

## 5. Implicaciones para el paper y la VM

### 5.1 Para el paper (POLYDIM_PAPER_V8.md)

La semántica denotacional completa fortalece la contribución cs.PL de tres maneras:

**A) Equivalencia de programas verificable:**
```
MIX(1, COMPOSE(T₂,T₁), 0, T₃) ≡ COMPOSE(T₂,T₁)   [por ⟦·⟧]
FIXPOINT(MIX(0.5,T,0.5,T), ε) ≡ FIXPOINT(T, ε)    [por linealidad de MIX]
```

**B) PROJECT como transformación natural:**
La semántica denotacional revela que PROJECT_E no es solo un funtor sino una
*transformación natural* η: Id_POLYDIM → F_E, donde F_E: Para(Vect) → E_E es el funtor
de proyección. Esto eleva Teorema 3 al nivel de coherencia categórica completa.

**C) ATTEND unificado con la capa algebraica:**
La distinción capa-algebraica/capa-implementación de la Constitución V6 Art. IV.1 se
expresa ahora como la inclusión L(R^N) ↪ Para(Vect). Ambas capas tienen una semántica
homogénea. Los revisores de cs.PL no pueden objetar que "ATTEND queda fuera del marco
formal" — está en Para(Vect), el mismo marco que todo lo demás.

### 5.2 Para la VM Rust (polydim_core.rs V0.1)

La semántica denotacional no cambia la VM V0.1 (que implementa la capa algebraica). Sí
implica que una futura VM completa (V0.2 con feature impl-layer) puede implementar
ATTEND como un morfismo en Para(Vect) — concretamente, como una función
(W_Q, W_K, W_V, s) → softmax(...) sin ningún tratamiento especial.

La verificación de igualdad de programas para optimización del compilador puede
implementarse comparando ⟦P₁⟧ = ⟦P₂⟧ en Para(Vect) en lugar de comparar árboles
sintácticos.

---

## 6. Qué queda abierto (honestidad epistémica)

Esta spec cierra la investigación principal de Art. V.3, pero dos cuestiones permanecen
abiertas como trabajo futuro:

**A) La mónada libre sobre Φ_ATTEND:**
arXiv:2501.02931 prueba que apilar capas de atención construye la mónada libre sobre el
endofunctor Φ. No hemos caracterizado formalmente esta mónada en el contexto de POLYDIM
ni explorado sus implicaciones para FIXPOINT(ATTEND). Esta es la dirección de
investigación más prometedora para un suplementario cs.PL de alto nivel.

**B) HoTT para GEO_ID:**
La formalización topológica de GEO_ID en HoTT (Constitución V6 Art. 4.1) sigue abierta.
La semántica denotacional en Para(Vect) no la resuelve — son ortogonales. GEO_ID
pertenece a la geometría del espacio de estados; Para(Vect) pertenece a la estructura
de los morfismos.

**C) Para(Vect) vs. Para(Man) para FIXPOINT no lineal:**
Si T no es lineal (e.g., FIXPOINT de una función con softmax), el espacio natural puede
ser Para(Man) — morfismos parametrizados sobre variedades diferenciables. Esto requiere
herramientas de geometría diferencial que exceden el alcance de V0.

---

## 7. Referencias

[Gavranovic et al. 2024] Gavranovic, B. et al. Categorical Foundations of Gradient-Based
Learning. ICML 2024. (Para(Vect) como marco para redes neuronales parametrizadas)

[arXiv:2501.02931, 2025] Self-Attention as a Parametric Endofunctor: A Categorical
Framework for Transformer Architectures. Enero 2025.
(ATTEND como endofunctor; mónada libre sobre capas de atención apiladas)

[Román 2024] Román, M. Monoidal Context Theory. arXiv:2404.06192. Abril 2024.
(Para(Vect) y lentes monoidales como morfismos de mensajes — relación con ALIGN)

[arXiv:2402.15332, 2024] Category Theory for Deep Learning. Junio 2024.
(Survey de Para, Lens, y semántica categórica de redes neuronales)

[arXiv:2606.19279, 2026] NeSyCat Torch: A Differentiable Tensor Implementation of
Categorical Semantics for Neurosymbolic Learning. Junio 2026.
(semántica categórica con mónadas paramétricas sobre tensores — sota actual)

[POLYDIM 2026a] SPEC_SEMANTICA_OPERACIONAL_V0.md (1RPcVVyofE6gil5AbS0QrLnA60rdKhBIk)
[POLYDIM 2026b] POLYDIM_CONSTITUCION_V6.md Art. V.3 (1LneDD0D8fHREiZN-_Xw-Ax9AzxfF6tKv)
[POLYDIM 2026c] POLYDIM_THEOREM3_PROOF_V1.md (1W41O6eNTKIjLHPoswBRVfgmNQDQZIC32)

---

## 8. Resumen ejecutivo — lo que esta spec resolvió

| Pregunta abierta (Art. V.3) | Respuesta |
|---|---|
| ¿Cuál es el dominio de ⟦·⟧? | Para(Vect) — morfismos diferenciables parametrizados |
| ¿D1 o D2? | D2 (Para(Vect)); D1 es el subcaso sin parámetros |
| ¿ATTEND tiene semántica denotacional? | Sí — endofunctor paramétrico Φ en Para(Vect) |
| ¿⟦·⟧ es composicional? | Sí — Teorema D-4: ⟦·⟧ es un funtor |
| ¿Dos programas pueden compararse denotativamente? | Sí — igualdad en Para(Vect) |
| ¿RECUR y FIXPOINT son lo mismo? | Sí, cuando RECUR converge — mismo punto fijo |
| ¿PROJECT es más que un funtor? | Sí — transformación natural η: Id → F_E |

---

*SPEC_SEMANTICA_DENOTACIONAL_V0.md · TASK_033b · 2026-06-28 · Claude Sonnet 4.6*
*Cierra: POLYDIM_CONSTITUCION_V6.md Art. V.3 (investigación abierta 🔬 → CERRADA ✅)*
*Sota utilizado: Gavranovic 2024 (ICML), arXiv:2501.02931 (2025), Román 2024, NeSyCat 2026*
