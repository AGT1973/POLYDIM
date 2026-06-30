# POLYDIM_DEST
# destino: polydim/spec/
# nombre:  POLYDIM_GRAMMAR_BNF_V1.md
# autor:   ai.mpat.agt@gmail.com
# fecha:   2026-06-25
# tarea:   TASK_C01

---

# POLYDIM — Gramática Formal V1
## BNF + Sintaxis Concreta + Semántica Denotacional

**Estado:** DRAFT — pendiente revisión docente
**Referencia en paper:** Sección 3.2 (Definiciones Formales), Sección 3.4 (Semántica Operacional)
**Restricciones constitucionales aplicadas:** R1 (T es la unidad), R2 (no hay variables), R3 (no hay loops), R4 (no hay if/else), R5 (tipos emergen de PROJECT)

---

## PARTE 1 — GRAMÁTICA BNF ABSTRACTA

### 1.1 No-terminales

```
<program>     ::= <transform>

<transform>   ::= <compose>
               |  <mix>
               |  <fixpoint>
               |  <project>
               |  <attend>
               |  <recur>
               |  <position-ref>

<compose>     ::= COMPOSE '(' <transform> ',' <transform> ')'

<mix>         ::= MIX '(' <weight> ',' <transform> ',' <weight> ',' <transform> ')'

<fixpoint>    ::= FIXPOINT '(' <transform> ',' <epsilon> ')'

<project>     ::= COMPILE '(' <transform> ',' <executor> ')'
               |  RENDER  '(' <transform> ',' <executor> ')'
               |  EXPORT  '(' <transform> ',' <executor> ')'

<attend>      ::= ATTEND '(' <matrix-ref> ',' <matrix-ref> ',' <matrix-ref> ')'

<recur>       ::= RECUR '(' <matrix-ref> ',' <matrix-ref> ',' <matrix-ref> ')'

<position>    ::= POSITION <identifier> ':' <dim-type>

<binding>     ::= BIND '(' <position-ref> ',' <position-ref> ',' <relation> ')'

<executor>    ::= DIM_FLUTTER
               |  DIM_RUST
               |  DIM_WASM
               |  DIM_SQL
               |  DIM_PYTHON
               |  DIM_GRAPH
               |  <executor> ∩ <executor>      -- intersección (pullback)

<dim-type>    ::= 'R^' <natural>               -- espacio vectorial de dimensión N

<weight>      ::= <float>                      -- ∈ [0.0, 1.0]
<epsilon>     ::= <float>                      -- ∈ (0.0, 1.0)
<matrix-ref>  ::= <identifier>                 -- referencia a una matriz W ∈ R^{N×N}
<position-ref>::= <identifier>                 -- referencia a una posición P ∈ R^N
<relation>    ::= CONTAINS | REFERENCES | COMPOSES | INHERITS
<identifier>  ::= [a-zA-Z_][a-zA-Z0-9_]*
<natural>     ::= [1-9][0-9]*
<float>       ::= [0-9]+ ( '.' [0-9]+ )?
```

### 1.2 Restricciones semánticas (no capturables en BNF pura)

**R-BNF-01:** `<weight>` ∈ [0.0, 1.0]. El parser debe rechazar valores fuera de rango.

**R-BNF-02:** En `MIX(α, T₁, β, T₂)`, los pesos α y β son independientes. No se requiere α + β = 1. Esto permite superposición con amplificación (α + β > 1) o atenuación (α + β < 1).

**R-BNF-03:** En `FIXPOINT(T, ε)`, T debe ser contractiva. Esta propiedad no es checkeable estáticamente en el caso general — el runtime puede emitir un warning si FIXPOINT no converge en max_iter iteraciones.

**R-BNF-04:** `RENDER` solo es válido con DIM_FLUTTER. `COMPILE` con DIM_RUST, DIM_WASM, DIM_PYTHON. `EXPORT` con DIM_SQL, DIM_GRAPH. El parser debe rechazar combinaciones inválidas.

**R-BNF-05:** La intersección de executors `E₁ ∩ E₂` solo produce un tipo bien formado cuando DIM_{E₁} ∩ DIM_{E₂} ≠ ∅ en R^N. Esta propiedad es verificable en runtime, no estáticamente.

---

## PARTE 2 — SINTAXIS CONCRETA (lenguaje de superficie)

### 2.1 Tokens léxicos

```
KEYWORD    := "position" | "transform" | "output" | "bind"
             | "COMPOSE" | "MIX" | "FIXPOINT" | "ATTEND" | "RECUR"
             | "COMPILE" | "RENDER" | "EXPORT"
             | "DIM_FLUTTER" | "DIM_RUST" | "DIM_WASM"
             | "DIM_SQL"     | "DIM_PYTHON" | "DIM_GRAPH"
             | "DIM_VECTOR"  | "DIM_TIME"   | "DIM_ERROR" | "DIM_META"
             | "CONTAINS" | "REFERENCES" | "COMPOSES" | "INHERITS"
FLOAT      := [0-9]+ ('.' [0-9]+)?
IDENT      := [a-zA-Z_][a-zA-Z0-9_]*
COMMENT    := '--' [^\n]* '\n'
LPAREN     := '('
RPAREN     := ')'
COMMA      := ','
COLON      := ':'
ASSIGN     := ':='
INTERSECT  := '∩'
ARROW      := '->'
```

### 2.2 Programa completo — ejemplo canónico

```polydim
-- POLYDIM Program: Semantic Search with Live UI
-- Archivo: search_pipeline.polydim

-- Declaración de posiciones (objetos geométricos)
position query_vector  : R^10000
position result_vector : R^10000
position corpus_index  : R^10000

-- Declaración de transformaciones base (matrices de parámetros)
-- W_Q, W_K, W_V: matrices del mecanismo de atención [declaradas externamente]

-- Composición del pipeline de búsqueda
transform attend_corpus :=
    ATTEND(W_Q_query, W_K_corpus, W_V_corpus)

transform refine_once :=
    MIX(0.8, attend_corpus, 0.2, ATTEND(W_Q_deep, W_K_deep, W_V_deep))

transform search_until_stable :=
    FIXPOINT(refine_once, 1e-4)

-- Proyecciones al mundo real
output search_ui    := RENDER(search_until_stable, DIM_FLUTTER)
output search_query := EXPORT(search_until_stable, DIM_SQL)

-- Tipo intersección: widget actualiza DB automáticamente
output live_results := RENDER(
    search_until_stable,
    DIM_FLUTTER ∩ DIM_SQL
)
```

### 2.3 Comunicación AI↔AI

```polydim
-- POLYDIM Program: Delegación de tarea entre IAs
-- IA_A genera la transformación de la tarea

transform task_transform :=
    COMPOSE(
        ATTEND(W_Q_task, W_K_context, W_V_context),
        COMPILE(MIX(0.9, attend_corpus, 0.1, RECUR(A_mamba, B_mamba, C_mamba)), DIM_RUST)
    )

-- La alineación ALIGN es transparente al lenguaje
-- El runtime maneja M: R^{d_A} → R^{d_B} automáticamente
-- output: la transformación T se transmite directamente
output to_ia_b := task_transform
-- IA_B aplica: new_state_B = M · task_transform · M†  (via ALIGN)
```

### 2.4 Punto fijo con binding de posiciones

```polydim
-- Binding geométrico entre posiciones
position user_profile   : R^10000
position preference_db  : R^10000

-- BIND: relación direccional entre posiciones
-- La dirección del vector de relación es semánticamente significativa
bind user_to_prefs := BIND(user_profile, preference_db, REFERENCES)

-- Transformación que opera sobre la relación
transform personalize :=
    COMPOSE(
        ATTEND(W_Q_user, W_K_prefs, W_V_prefs),
        MIX(0.7, COMPILE(user_profile, DIM_SQL), 0.3, RENDER(preference_db, DIM_FLUTTER))
    )

output personalized_view := RENDER(FIXPOINT(personalize, 1e-3), DIM_FLUTTER)
```

---

## PARTE 3 — SEMÁNTICA DENOTACIONAL

### 3.1 Dominio semántico

El dominio de valores de POLYDIM es el espacio de transformaciones lineales sobre R^N:

```
Val = { T : R^N → R^N | T admisible }
     ≅ R^{N×N}   (con estructura de álgebra de Banach bajo ‖·‖_op)
```

Para `FIXPOINT`, usamos el dominio de Scott basado en la norma:
```
(Val, ≤)  donde  T₁ ≤ T₂  iff  ‖T₁(v)‖ ≤ ‖T₂(v)‖  ∀v ∈ R^N
⊥ = 0   (transformación cero)
```

### 3.2 Función de denotación ⟦·⟧

```
⟦·⟧ : Syntax → (Env → Val)
```

donde Env = {identifier → Val} es un entorno de ligaduras.

**Reglas de denotación:**

```
⟦COMPOSE(T₁, T₂)⟧ ρ  =  ⟦T₂⟧ρ ∘ ⟦T₁⟧ρ

⟦MIX(α, T₁, β, T₂)⟧ ρ  =  α · ⟦T₁⟧ρ + β · ⟦T₂⟧ρ

⟦FIXPOINT(T, ε)⟧ ρ  =  lfp_ε(⟦T⟧ρ)
    donde lfp_ε(F) = F^n(⊥) para el menor n tal que ‖F^n(⊥) - F^{n-1}(⊥)‖ < ε

⟦COMPILE(T, E)⟧ ρ  =  F_E ∘ ⟦T⟧ρ
    donde F_E : Val → Val_E  es el funtor de compilación para executor E

⟦RENDER(T, DIM_FLUTTER)⟧ ρ  =  φ_Flutter ∘ ⟦T⟧ρ
    donde φ_Flutter : Val → WidgetTree  es el isomorfismo algebraico Flutter

⟦ATTEND(W_Q, W_K, W_V)⟧ ρ  =  λs. softmax(s·W_Q · (s·W_K)^T / √d) · (s·W_V)

⟦RECUR(A, B, C)⟧ ρ  =  λs. (A·s + B·s, C·s)   [par (h', y)]
```

### 3.3 Propiedades de la semántica

**Monotonía de ⟦FIXPOINT⟧:**
Si T es monótona bajo ≤, entonces ⟦FIXPOINT(T,ε)⟧ es bien definido y converge a lfp.

**Linealidad preservada:**
Si T₁, T₂ son lineales, ⟦COMPOSE(T₁,T₂)⟧ y ⟦MIX(α,T₁,β,T₂)⟧ son lineales.

**Funtorialidad de ⟦PROJECT⟧:**
⟦COMPILE(COMPOSE(T₂,T₁), E)⟧ = ⟦COMPILE(T₂,E)⟧ ∘ ⟦COMPILE(T₁,E)⟧
(Corolario del Teorema 3 — ver THEOREM3_PROJECT_FUNCTOR_V1.md)

---

## PARTE 4 — SISTEMA DE TIPOS ESTÁTICO (parcial)

### 4.1 Tipos sintácticos

```
τ ::= Transform(N)        -- transformación sobre R^N
    | Position(N)         -- posición en R^N
    | Matrix(N, M)        -- matriz N×M (parámetros de ATTEND)
    | Type(E)             -- tipo del executor E (resultado de PROJECT)
    | Type(E₁ ∩ E₂)      -- tipo intersección (pullback)
    | Float               -- peso o epsilon
```

### 4.2 Reglas de tipado

```
[T-COMPOSE]
  Γ ⊢ T₁ : Transform(N)    Γ ⊢ T₂ : Transform(N)
  ─────────────────────────────────────────────────
  Γ ⊢ COMPOSE(T₁, T₂) : Transform(N)

[T-MIX]
  Γ ⊢ α : Float   α ∈ [0,1]
  Γ ⊢ β : Float   β ∈ [0,1]
  Γ ⊢ T₁ : Transform(N)    Γ ⊢ T₂ : Transform(N)
  ──────────────────────────────────────────────────
  Γ ⊢ MIX(α, T₁, β, T₂) : Transform(N)

[T-FIXPOINT]
  Γ ⊢ T : Transform(N)   Γ ⊢ ε : Float   ε ∈ (0,1)
  contractive_check(T) = ok   [verificado en runtime]
  ───────────────────────────────────────────────────
  Γ ⊢ FIXPOINT(T, ε) : Transform(N)

[T-COMPILE]
  Γ ⊢ T : Transform(N)    E ∈ {DIM_RUST, DIM_WASM, DIM_PYTHON}
  ──────────────────────────────────────────────────────────────
  Γ ⊢ COMPILE(T, E) : Type(E)

[T-RENDER]
  Γ ⊢ T : Transform(N)
  ──────────────────────────────────
  Γ ⊢ RENDER(T, DIM_FLUTTER) : Type(DIM_FLUTTER)

[T-EXPORT]
  Γ ⊢ T : Transform(N)    E ∈ {DIM_SQL, DIM_GRAPH}
  ──────────────────────────────────────────────────
  Γ ⊢ EXPORT(T, E) : Type(E)

[T-ATTEND]
  Γ ⊢ W_Q : Matrix(N, d)    Γ ⊢ W_K : Matrix(N, d)    Γ ⊢ W_V : Matrix(N, d)
  ─────────────────────────────────────────────────────────────────────────────
  Γ ⊢ ATTEND(W_Q, W_K, W_V) : Transform(N)

[T-INTERSECT]
  Γ ⊢ T : Transform(N)
  DIM_{E₁} ∩ DIM_{E₂} ≠ ∅   [verificado en runtime via activaciones]
  ──────────────────────────────────────────────────────────────────
  Γ ⊢ RENDER(T, E₁ ∩ E₂) : Type(E₁ ∩ E₂)

[T-POSITION]
  ──────────────────────────────────────────────────────────
  Γ ⊢ position x : R^N ⊢ x : Position(N)

[T-BIND]
  Γ ⊢ p₁ : Position(N)    Γ ⊢ p₂ : Position(N)    r ∈ {CONTAINS, REFERENCES, COMPOSES, INHERITS}
  ─────────────────────────────────────────────────────────────────────────────────────────────────
  Γ ⊢ BIND(p₁, p₂, r) : Binding(p₁, p₂, r)
```

### 4.3 Tipos que NO existen en POLYDIM

Los siguientes "tipos" de lenguajes convencionales no tienen análogo directo en POLYDIM:

| Concepto convencional | Por qué no existe | Equivalente POLYDIM |
|---|---|---|
| `int x = 5`  | No hay variables con nombre | `position x : R^10000` con activación en DIM_VECTOR |
| `if (c) T else F` | No hay bifurcación binaria | `MIX(α, T, 1-α, F)` con α ∈ [0,1] |
| `for (i=0; i<n; i++)` | No hay iteración secuencial | `FIXPOINT(T, ε)` |
| `class A extends B` | No hay herencia OOP | `BIND(A, B, INHERITS)` + activaciones compartidas |
| `int` / `String` / `bool` | No hay tipos declarados | `PROJECT(P, DIM_SQL)` → tipo emerge |
| `null` / `None` | No hay valor nulo | Activación 0.0 en todos los subespacios |

---

## PARTE 5 — PARSER DE REFERENCIA (Python, recursivo descendente)

```python
"""
Parser mínimo para sintaxis concreta POLYDIM.
Produce un AST (Abstract Syntax Tree) de la gramática.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union
import re

# AST nodes
@dataclass
class ComposeNode:
    t1: 'ASTNode'
    t2: 'ASTNode'

@dataclass
class MixNode:
    alpha: float
    t1: 'ASTNode'
    beta: float
    t2: 'ASTNode'

@dataclass
class FixpointNode:
    transform: 'ASTNode'
    epsilon: float

@dataclass
class ProjectNode:
    verb: str              # COMPILE | RENDER | EXPORT
    transform: 'ASTNode'
    executor: str          # DIM_FLUTTER, etc.
    intersection: Optional[str] = None  # second DIM for ∩

@dataclass
class AttendNode:
    w_q: str; w_k: str; w_v: str

@dataclass
class RecurNode:
    A: str; B: str; C: str

@dataclass
class RefNode:
    name: str

ASTNode = Union[ComposeNode, MixNode, FixpointNode, ProjectNode, AttendNode, RecurNode, RefNode]


class PolydimParser:
    """
    Parser recursivo descendente para POLYDIM.
    Produce un AST a partir de código fuente .polydim.
    """

    EXECUTORS = {
        "DIM_FLUTTER", "DIM_RUST", "DIM_WASM",
        "DIM_SQL", "DIM_PYTHON", "DIM_GRAPH",
        "DIM_VECTOR", "DIM_TIME", "DIM_ERROR", "DIM_META"
    }

    def __init__(self, source: str):
        # Tokenizar: simplificado (split por espacios y delimitadores)
        self.tokens = re.findall(
            r'COMPOSE|MIX|FIXPOINT|COMPILE|RENDER|EXPORT|ATTEND|RECUR'
            r'|DIM_\w+|[a-zA-Z_]\w*|\d+\.\d+|\d+|[(),∩:=]|--[^\n]*',
            source
        )
        # Filtrar comentarios
        self.tokens = [t for t in self.tokens if not t.startswith('--')]
        self.pos = 0

    def peek(self) -> Optional[str]:
        if self.pos < len(self.tokens): return self.tokens[self.pos]
        return None

    def consume(self, expected: Optional[str] = None) -> str:
        tok = self.tokens[self.pos]
        if expected and tok != expected:
            raise SyntaxError(f"Esperado {expected!r}, encontrado {tok!r} en pos {self.pos}")
        self.pos += 1
        return tok

    def parse_float(self) -> float:
        tok = self.consume()
        try: return float(tok)
        except ValueError: raise SyntaxError(f"Esperado float, encontrado {tok!r}")

    def parse_executor(self) -> tuple:
        """Parsea DIM_X o DIM_X ∩ DIM_Y"""
        executor = self.consume()
        if executor not in self.EXECUTORS:
            raise SyntaxError(f"Executor desconocido: {executor!r}")
        intersection = None
        if self.peek() == '∩':
            self.consume('∩')
            intersection = self.consume()
            if intersection not in self.EXECUTORS:
                raise SyntaxError(f"Executor en intersección desconocido: {intersection!r}")
        return executor, intersection

    def parse_transform(self) -> ASTNode:
        """Parsea una expresión de transformación."""
        tok = self.peek()

        if tok == 'COMPOSE':
            self.consume('COMPOSE'); self.consume('(')
            t1 = self.parse_transform(); self.consume(',')
            t2 = self.parse_transform(); self.consume(')')
            return ComposeNode(t1=t1, t2=t2)

        elif tok == 'MIX':
            self.consume('MIX'); self.consume('(')
            alpha = self.parse_float(); self.consume(',')
            t1 = self.parse_transform(); self.consume(',')
            beta = self.parse_float(); self.consume(',')
            t2 = self.parse_transform(); self.consume(')')
            if not (0 <= alpha <= 1 and 0 <= beta <= 1):
                raise SyntaxError(f"Pesos MIX fuera de [0,1]: α={alpha}, β={beta}")
            return MixNode(alpha=alpha, t1=t1, beta=beta, t2=t2)

        elif tok == 'FIXPOINT':
            self.consume('FIXPOINT'); self.consume('(')
            t = self.parse_transform(); self.consume(',')
            eps = self.parse_float(); self.consume(')')
            return FixpointNode(transform=t, epsilon=eps)

        elif tok in ('COMPILE', 'RENDER', 'EXPORT'):
            verb = self.consume(); self.consume('(')
            t = self.parse_transform(); self.consume(',')
            executor, inters = self.parse_executor(); self.consume(')')
            return ProjectNode(verb=verb, transform=t, executor=executor, intersection=inters)

        elif tok == 'ATTEND':
            self.consume('ATTEND'); self.consume('(')
            wq = self.consume(); self.consume(',')
            wk = self.consume(); self.consume(',')
            wv = self.consume(); self.consume(')')
            return AttendNode(w_q=wq, w_k=wk, w_v=wv)

        elif tok == 'RECUR':
            self.consume('RECUR'); self.consume('(')
            a = self.consume(); self.consume(',')
            b = self.consume(); self.consume(',')
            c = self.consume(); self.consume(')')
            return RecurNode(A=a, B=b, C=c)

        else:
            # Referencia a un identificador
            return RefNode(name=self.consume())

    def parse_program(self) -> dict:
        """Parsea un programa POLYDIM completo. Devuelve dict de definiciones."""
        defs = {}
        while self.peek() is not None:
            keyword = self.peek()
            if keyword == 'transform':
                self.consume('transform')
                name = self.consume()
                self.consume(':=')
                defs[name] = ('transform', self.parse_transform())
            elif keyword == 'output':
                self.consume('output')
                name = self.consume()
                self.consume(':=')
                defs[name] = ('output', self.parse_transform())
            elif keyword == 'position':
                self.consume('position')
                name = self.consume(); self.consume(':')
                dim = self.consume()  # R^N
                defs[name] = ('position', dim)
            else:
                break  # fin o token desconocido
        return defs


def parse(source: str) -> dict:
    """Parsea un programa POLYDIM y devuelve el AST."""
    return PolydimParser(source).parse_program()


# Prueba del parser con el ejemplo canónico
if __name__ == "__main__":
    sample = """
    position query_vector : R^10000
    transform attend_corpus := ATTEND(W_Q, W_K, W_V)
    transform refine := MIX(0.8, attend_corpus, 0.2, ATTEND(W_Q2, W_K2, W_V2))
    transform search := FIXPOINT(refine, 0.0001)
    output ui := RENDER(search, DIM_FLUTTER)
    output db := EXPORT(search, DIM_SQL)
    """
    ast = parse(sample)
    for name, (kind, node) in ast.items():
        print(f"  {kind} {name!r}: {node}")
```

---

## PARTE 6 — PROPIEDADES GARANTIZADAS POR LA GRAMÁTICA

| Propiedad | Garantía | Mecanismo |
|---|---|---|
| No hay variables mutables | Posiciones solo se declaran, no se reasignan | Gramática no tiene `=` (solo `:=` para ligaduras) |
| No hay loops secuenciales | Solo FIXPOINT como iteración | FIXPOINT no es un loop de control — es convergencia geométrica |
| No hay bifurcación binaria | Solo MIX continuo | MIX exige α, β ∈ [0,1], no condición booleana |
| Tipos emergen de PROJECT | No hay declaración de tipo en expresiones | La gramática no tiene `int`, `String`, `bool` |
| GEO_ID es invariante | Posiciones solo se declaran | No hay sintaxis para modificar el GEO_ID después de la declaración |
| T: R^N → R^N es la unidad | Toda transformación es un Transform(N) | El sistema de tipos no permite instrucciones secuenciales |

---

*POLYDIM_GRAMMAR_BNF_V1.md · V1.0 · 2026-06-25 · TASK_C01*
*Secciones 1.1 (BNF abstracta), 2 (sintaxis concreta), 3 (semántica denotacional) listas para paper.*
*Sección 5 (parser Python) lista para integrar en bootstrap V0.4.*
