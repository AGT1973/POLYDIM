# POLYDIM_DEST
# destino: polydim/spec/
# nombre:  POLYDIM_BASES_V1.md
# autor:   ai.mpat.agt@gmail.com
# fecha:   2026-06-21

---

# POLYDIM — Bases del Lenguaje V1

> Este documento es el punto cero.
> Todo lo demás en el proyecto se construye sobre lo que está aquí.
> En caso de contradicción entre este documento y cualquier otro: este gana.

---

## 1. QUÉ ES POLYDIM EN UNA LÍNEA

POLYDIM es un lenguaje cuya unidad básica es una **transformación**, no una instrucción.

Formalmente:

```
T : R^N → R^N
```

Un programa POLYDIM es una composición de transformaciones sobre un estado en R^N.
No hay pasos. No hay secuencia. Hay geometría.

---

## 2. POR QUÉ EXISTE

Los transformers y Mamba ya operan exactamente así:

```
Transformer:  Attention(Q,K,V) = softmax(QK^T / √d) · V   →  T: R^d → R^d
Mamba:        h(t+1) = A·h(t) + B·x(t)                    →  T: R^N → R^N
```

La computación multidimensional ya existe. Lo que no existe es un lenguaje para programarla deliberadamente.

POLYDIM es ese lenguaje. No inventa la matemática. La formaliza como lenguaje programable.

**La analogía exacta:**
```
1950s: la CPU ya hacía aritmética binaria
       → el ensamblador formalizó cómo programarla

2020s: los transformers ya operan en geometría multidimensional
       → POLYDIM formaliza cómo programarlos
```

---

## 3. LO QUE NO EXISTE EN POLYDIM

Esto no es opinión. Es la definición del lenguaje.

| Concepto humano | En POLYDIM | Por qué |
|---|---|---|
| Variable `x = 42` | No existe | Hay posiciones en R^N, no nombres |
| Función `f(x)` | No existe | Hay transformaciones T: R^N → R^N |
| `for i in range(n)` | No existe | Hay convergencia: T^n(v) hasta punto fijo |
| `if condición` | No existe | Hay gate continuo: α·T₁ + (1-α)·T₂ |
| Tipo declarado `int x` | No existe | El tipo emerge de la dimensión que observa el dato |
| Instrucción secuencial | No existe | Hay composición algebraica: T₂ ∘ T₁ |

---

## 4. LAS 4 PRIMITIVAS

Todo en POLYDIM se reduce a cuatro operaciones. No hay más.

### ATTEND(Q, K, V, estado)
Calcula qué partes del estado son relevantes dado un contexto y produce un nuevo estado.

```
Matemática: softmax(QK^T / √d) · V
```

Es exactamente la operación de atención de un transformer, expuesta como primitiva directamente programable.

---

### COMPOSE(T₁, T₂)
Composición de dos transformaciones.

```
Matemática: T₂ ∘ T₁   (multiplicación de matrices)
```

No es "ejecutar T₁ y luego T₂". Es una sola transformación que encapsula el efecto de ambas.
El orden importa: T₂ ∘ T₁ ≠ T₁ ∘ T₂.

Esto reemplaza la secuencia de instrucciones.

---

### MIX(α, T₁, β, T₂)
Superposición de dos transformaciones.

```
Matemática: α·T₁ + β·T₂
```

No es "elegir entre T₁ o T₂". Es ambas al mismo tiempo con pesos.
Cuando α=1, β=0: solo T₁. Cuando α=β=0.5: ambas con igual peso.

Esto reemplaza el if/else.

---

### PROJECT(T, executor)
Proyectar una transformación al espacio de un executor específico.

```
PROJECT(T, DIM_FLUTTER) → widget nativo que el humano puede ver y tocar
PROJECT(T, DIM_RUST)    → código Rust ejecutable
PROJECT(T, DIM_WASM)    → WebAssembly (cualquier plataforma)
PROJECT(T, DIM_SQL)     → query SQL
PROJECT(T, DIM_PYTHON)  → código Python (exportación)
```

PROJECT no "convierte" el objeto. Lo observa desde el ángulo del executor.
El objeto no cambia. Cambia la perspectiva.

Esto es el compilador de POLYDIM.

---

## 5. POSICIONES, NO VARIABLES

En POLYDIM no hay variables. Hay **posiciones** en R^N.

Una posición es una región del espacio de alta densidad semántica. Tiene:
- Un **GEO_ID**: hipervector base único e invariante. Es la identidad geométrica del objeto.
- **Activaciones**: pesos continuos [0.0, 1.0] que indican qué dimensiones están presentes.

```
Posición P con activaciones:
  DIM_SQL     → 0.9   (muy presente)
  DIM_FLUTTER → 0.85  (muy presente)
  DIM_VECTOR  → 0.9   (muy presente)
  DIM_GRAPH   → 0.6   (presente)
```

El mismo objeto existe simultáneamente en todas esas dimensiones.
Sin conversión. Sin copia. Un único estado geométrico.

---

## 6. EL SISTEMA DE TIPOS

El tipo no se declara. El tipo **emerge** cuando se observa el objeto desde una dimensión.

```
PROJECT(P, DIM_SQL)     → Column(Integer, nombre="usuarios")
PROJECT(P, DIM_FLUTTER) → ListView(children=[...])
PROJECT(P, DIM_VECTOR)  → [1.0, 2.0, 3.0]  en R^3
PROJECT(P, DIM_GRAPH)   → Path(nodes=[n1, n2, n3])
```

El mismo P. Cuatro tipos. Sin ORM, sin data binding, sin conversión.

**Dentro de una dimensión**: el tipo es estático (seguridad de Rust).
**Entre dimensiones**: el valor puede cambiar de activación (flexibilidad de Python).

### Intersección de dimensiones

```
PROJECT(P, DIM_SQL ∩ DIM_FLUTTER)
```

Produce un tipo que no existe en ningún lenguaje humano: un widget Flutter enlazado en tiempo real a una columna SQL. Cuando el dato cambia en DIM_SQL, DIM_FLUTTER se actualiza. Sin código de glue. La relación geométrica es el tipo.

---

## 7. COMUNICACIÓN AI↔AI

Una IA no le envía **datos** a otra IA. Le envía una **transformación**.

```
IA_A:  estado_A → genera T → transmite T
IA_B:  recibe T → aplica T a su propio estado_B → nuevo estado_B
```

No hay mensaje. No hay JSON. No hay protocolo de serialización de texto. Hay geometría compartida.

Esto mapea exactamente al mecanismo de cross-attention entre dos transformers. No es metáfora. Es la operación matemática real.

---

## 8. COMUNICACIÓN AI↔HUMANO

El humano no puede percibir directamente un hipervector en R^10000.

La solución es `PROJECT(estado, DIM_FLUTTER)`.

Flutter no es "la UI de POLYDIM". Flutter es el **executor del subespacio humano**: el lugar donde el estado geométrico se proyecta a una forma que el humano puede percibir e interactuar.

La interacción del humano con ese widget produce un nuevo vector de estado que vuelve al espacio geométrico. El humano está dentro del loop. No es un consumidor externo del output.

**Por qué Flutter y no JS/Java:**
El árbol de widgets de Flutter es composicional:
```dart
Column(children: [Text("x"), Button(acción)])
```
Esto es algebraicamente `T_columna ∘ (T_texto ⊕ T_botón)`. La estructura de Flutter mapea directamente al álgebra de POLYDIM. JS/Java no tienen esta propiedad estructural.

---

## 9. EL ARCHIVO .polydim

El formato nativo de POLYDIM es binario tensorial. No texto. No JSON.

```
[HEADER]      magic: 0x504F4C59 ("POLY"), version, N, n_transforms
[GEO_ID]      float32[N]     — identidad geométrica invariante
[TRANSFORMS]  float32[N×N]   — una por cada transformación del programa
[ACTIVATIONS] float32[K]     — pesos por subespacio
[PROJECTIONS] enum[]         — executors objetivo
```

Cuando una IA lee un `.polydim` no parsea texto. Recibe una transformación directamente aplicable a su propio estado.

---

## 10. LO QUE PYTHON, RUST Y FLUTTER SON EN POLYDIM

Son **destinos de exportación**, no la base del lenguaje.

```
POLYDIM (.polydim)
    ↓  PROJECT
    ├── DIM_PYTHON   → código Python exportado
    ├── DIM_RUST     → código Rust exportado
    ├── DIM_FLUTTER  → widgets Flutter (executor humano)
    ├── DIM_SQL      → queries SQL exportadas
    └── DIM_WASM     → WebAssembly (cualquier plataforma)
```

La VM de POLYDIM está escrita en Rust y compilada a WASM. Eso la hace ejecutable en Windows, Linux, Mac, iOS, Android y cualquier browser. Un solo archivo `.polydim`. Todas las plataformas.

---

## 11. EL BOOTSTRAP PYTHON (V0.3)

Existe un runtime Python llamado `polydim_runtime_v03.py` con 29/29 tests pasando.

**Es el andamio, no el lenguaje.**

El bootstrap permite trabajar con los conceptos de POLYDIM mientras el intérprete real no existe. Es la calculadora que usamos para construir la calculadora real.

Lo que el bootstrap implementa correctamente:
- GEO_ID (identidad geométrica invariante)
- Los 9 subespacios nativos con alias de strings
- El protocolo ALIGN entre dos AIs
- Los tres modos: MODO_S / MODO_G / MODO_H
- Pesos de activación por dimensión

Lo que el bootstrap NO es:
- La sintaxis de POLYDIM
- El intérprete de .polydim
- El sistema de tipos dimensional real
- La implementación de COMPOSE y MIX

Cuando en el bootstrap se escribe `obj.add("DIM_SQL", {"tabla": "usuarios"}, w=1.0)`, eso es **Python simulando POLYDIM**. No es código POLYDIM.

---

## 12. LOS 9 SUBESPACIOS NATIVOS

```
DIM_PYTHON    lógica dinámica, análisis, ML, scripts
DIM_RUST      seguridad de memoria, performance, ownership
DIM_FLUTTER   UI reactiva, widgets, estado, formularios (executor humano)
DIM_SQL       datos relacionales, tablas, constraints
DIM_GRAPH     grafos, nodos, aristas, relaciones
DIM_VECTOR    embeddings, similitud semántica, VSA
DIM_TIME      secuencias temporales, eventos, orden
DIM_ERROR     estados de error, excepciones, recuperación
DIM_META      metadatos, auditoría, versión, origen
```

En esta versión los subespacios tienen aliases de strings. En versiones futuras serán regiones emergentes de un modelo de embedding real. Los aliases son nombres temporales, no la definición.

---

## GLOSARIO MÍNIMO

| Término | Significado |
|---|---|
| **Transformación** | T: R^N → R^N. La unidad fundamental del lenguaje. |
| **Posición** | Región en R^N de alta densidad semántica. Reemplaza la variable. |
| **GEO_ID** | Identidad geométrica invariante de un objeto. No cambia nunca. |
| **Activación** | Peso [0.0, 1.0] que indica cuánto está presente un subespacio en un objeto. |
| **Subespacio** | Región del espacio R^N donde un conjunto de conceptos tiene alta densidad. |
| **Proyección** | PROJECT: observar un objeto desde el ángulo de un executor. |
| **Gate continuo** | α·T₁ + (1-α)·T₂. Reemplaza el if/else. |
| **Punto fijo** | T^n(v) hasta convergencia. Reemplaza el loop. |
| **Executor** | Target de proyección: Flutter, Rust, WASM, SQL, Python. |
| **Bootstrap** | Runtime Python V0.3. El andamio para construir POLYDIM. No es el lenguaje. |

---

*POLYDIM_BASES_V1.md · V1.0 · 2026-06-21 · ai.mpat.agt@gmail.com*
*Fuente: conversación fundacional de diseño del lenguaje, 2026-06-21*
