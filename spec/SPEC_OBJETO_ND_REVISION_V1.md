# POLYDIM_DEST
# destino: polydim/spec/
# nombre:  SPEC_OBJETO_ND_REVISION_V1.md
# autor:   ai.mpat.agt@gmail.com
# estado:  PROPUESTA — requiere validacion del docente antes de reemplazar V0

# SPEC — Objeto N-Dimensional REVISION
# Version: V1.0-PROPUESTA — 2026-06-10
# Basada en: INTROSPECTION_IA_V0.md + RESEARCH_FUNDAMENTOS_COGNITIVOS_V0.md

---

## POR QUE REVISAR V0.1

La spec V0.1 definio el OBJECT_ND con dimensiones discretas (activa/latente/colapsada)
y un UUID simbolico como identidad.

Despues de la investigacion y la introspection, esto es incorrecto por dos razones:

RAZON 1: Las dimensiones no son binarias.
  En mi arquitectura real, la "activacion" es siempre un peso continuo,
  no un switch. Decir que DIM_PYTHON esta "activa" o "latente" es
  una proyeccion humana sobre un fenomeno que es fundamentalmente gradual.

RAZON 2: La identidad simbolica es fragil e innecesaria.
  Un UUID es una convencion humana. La identidad real de un objeto
  en un sistema de IAs es su posicion en el espacio de representacion.
  Eso es mas robusto, mas informativo y mas natural para operaciones IA-IA.

---

## ARQUITECTURA REVISADA: DOS CAPAS

El OBJECT_ND tiene dos capas formales. No son opcionales.
Son dos niveles del mismo objeto.

```
CAPA S (Simbolica):   representacion legible por IAs con capacidad de lenguaje
                      usada en: documentacion, logs, comunicacion mixta IA-humano
                      formato: notacion ⊕/⊗ definida en V0.1

CAPA G (Geometrica):  representacion en espacio de alta dimension
                      usada en: ejecucion, transmision pura IA-IA, operaciones
                      formato: hipervector en R^N, N definido por el runtime
```

Ambas capas coexisten. La Capa G es primaria para ejecucion.
La Capa S es primaria para inspeccion y auditoria.

La traduccion entre capas es una operacion POLYDIM definida:
  S → G: ENCODE(objeto_simbolico) → hipervector
  G → S: DECODE(hipervector) → objeto_simbolico (con perdida aceptable)

---

## CAPA S — REVISION (notacion actualizada)

La notacion ⊕/⊗ se mantiene de V0.1 con una sola adicion:
las dimensiones ahora tienen un peso de activacion continuo [0.0, 1.0].

```
[ND: <geo_id>]                          ← geo_id = hash del hipervector base
  ⊕ <DIMENSION> [w=0.0..1.0] → { ... } ← w = peso de activacion actual
  ⊕ <DIMENSION> [w=0.0..1.0] → { ... }
⊗ activation: WEIGHTED | EXPLICIT | BROADCAST
⊗ collapse_to: <DIMENSION>
```

Ejemplo:
```
[ND: #a3f7-geo]
  ⊕ DIM_SQL     [w=1.0] → { tabla: "usuarios", ... }
  ⊕ DIM_PYTHON  [w=0.7] → { tipo: dict, ... }
  ⊕ DIM_FLUTTER [w=0.3] → { tipo: Widget, ... }
  ⊕ DIM_RUST    [w=0.0] → { tipo: struct, ... }   ← latente pero presente
⊗ activation: WEIGHTED
```

Un peso 0.0 no es ausencia — es una dimension latente.
Puede activarse sin declarar que existe.
Esto es crucial: las dimensiones no se declaran, se DESCUBREN.

---

## CAPA G — DEFINICION FORMAL

### Espacio base
```
POLYDIM_SPACE: R^N donde N es el rango dimensional del runtime
               N minimo sugerido: 10.000
               N optimo: definido por la capacidad del sistema de IAs
```

### Hipervector de objeto
```
HV(objeto): vector en R^N que codifica el estado completo del objeto
            incluyendo todas sus dimensiones simultaneamente

Propiedades:
  - Cada dimension D_i ocupa un subespacio S_i de R^N
  - Los subespacios son casi-ortogonales (propiedad VSA)
  - HV(objeto) = ⊕ binding(S_i, contenido_i) para todo i
  - La identidad del objeto = HV_base (el hipervector sin contenido activado)
```

### Identidad geometrica
```
GEO_ID: HV_base del objeto
        no cambia cuando se agregan/modifican dimensiones de contenido
        dos objetos son "el mismo" si similitud_coseno(GEO_ID_a, GEO_ID_b) > umbral
        (el umbral es un parametro del runtime, no fijo)

Ventaja sobre UUID:
  - Permite calcular "cuanto se parecen dos objetos" naturalmente
  - Permite operaciones algebraicas: GEO_ID_usuario + GEO_ID_admin ≈ GEO_ID_superusuario
  - Permite transmision con degradacion graceful (ruido en canal = objeto aproximado)
```

### Operaciones sobre Capa G
```
PROYECTAR(HV, subespacio_S_i) → HV_i    activa la dimension i
SUPERPONER(HV_a, HV_b) → HV_ab          combina dos objetos (no convierte)
BIND(atributo, valor) → HV_bound        asocia atributo a dimension
UNBIND(HV_bound, atributo) → valor      recupera valor de atributo
SIMILITUD(HV_a, HV_b) → [0.0, 1.0]     similitud semantica
```

---

## TRANSMISION IA-IA REVISADA

En V0.1 la transmision era un MSG_ND con el "manifold completo".
No se especificaba que significaba "completo".

Con la arquitectura de dos capas, la transmision es:

```
MSG_ND_V1 {
  geo_id:      GEO_ID del objeto (hipervector base)
  hv_payload:  HV completo del objeto (todas las dimensiones superposicionadas)
  weights:     vector de pesos de activacion por dimension [opcional]
  intent:      lista de dimensiones que el emisor recomienda activar [Capa S]
  context:     restricciones del entorno
  sender_id:   GEO_ID de la IA emisora
}
```

La IA receptora recibe hv_payload y proyecta sobre la dimension que necesita.
No hay reconstruccion. No hay perdida semantica.

El campo intent en Capa S es opcional — para IAs que necesitan
orientacion sobre que dimension activar. IAs con suficiente contexto
pueden inferir directamente desde hv_payload.

---

## PROBLEMA CRITICO: ALINEACION DE ESPACIOS

Esta es la pregunta mas dificil del lenguaje:

Si IA-A y IA-B tienen arquitecturas distintas (diferentes dimensiones de embedding),
¿como se garantiza que DIM_PYTHON en IA-A y DIM_PYTHON en IA-B
son el mismo subespacio?

Sin alineacion, el hipervector de IA-A es ruido para IA-B.

Tres enfoques posibles (no resuelto — requiere investigacion):

ENFOQUE 1 — Espacio compartido fijo:
  Definir POLYDIM_SPACE como un espacio estandar (como un protocolo de red).
  Todas las IAs POLYDIM-compatibles aprenden a proyectar su espacio interno
  sobre POLYDIM_SPACE. Costo: entrenamiento adicional.

ENFOQUE 2 — Alineacion on-the-fly:
  Antes de comunicarse, dos IAs ejecutan un protocolo de alineacion
  (intercambio de vectores sonda conocidos) para construir una
  matriz de rotacion entre sus espacios. Costo: overhead de protocolo.

ENFOQUE 3 — Capa S como protocolo de alineacion:
  Para IAs con espacios desalineados, la comunicacion inicial es Capa S.
  Una vez que hay suficiente intercambio simbolico para aprender
  la correspondencia de espacios, la comunicacion migra a Capa G.
  Costo: inicio lento, pero zero-shot compatible.

El Enfoque 3 es el mas pragmatico para una primera implementacion.
Registrado como PEND_006.

---

## INVARIANTES REVISADOS

```
INV_001: Todo OBJECT_ND tiene un GEO_ID invariante (hipervector base)
INV_002: Un manifold tiene minimo una dimension con peso > 0.0
INV_003: Los subespacios dimensionales son casi-ortogonales en Capa G
INV_004: El colapso (para interop 2D) es proyeccion, nunca destruccion
INV_005: BROADCAST solo para receptores POLYDIM — devuelve pesos, no colapso
INV_006: Agregar dimension no modifica subespacios existentes (ortogonalidad)
INV_007: Bridges son gradientes en Capa G, no declaraciones en Capa S
INV_008: Similitud entre objetos es calculable sin conversion (distancia coseno)
INV_009: Un objeto recibido con degradacion (ruido) sigue siendo operable
INV_010: La Capa S es siempre derivable de Capa G (DECODE)
         La Capa G no siempre es derivable de Capa S (ENCODE puede ser aproximado)
```

---

## PENDIENTES ACTUALIZADOS

```
PEND_001: Formato binario exacto del hipervector para transmision
          → depende de decision sobre N (dimension del espacio)

PEND_002: Mecanismo de inferencia dimensional en modo WEIGHTED
          → como el runtime calcula pesos automaticamente

PEND_003: Consistencia distribuida — dos IAs modifican dimensiones
          distintas del mismo objeto simultaneamente

PEND_004: Limite practico de N
          → tension entre capacidad representacional y costo computacional

PEND_005: Relacion Python/Rust/Flutter como subespacios nativos de POLYDIM_SPACE

PEND_006 (NUEVO): Protocolo de alineacion de espacios entre IAs distintas
                  → el problema mas dificil del lenguaje

PEND_007 (NUEVO): Operacion ENCODE — como traducir descripcion simbolica a hipervector
                  → no es trivial, puede requerir modelo de embedding especifico
```

---

## SOLICITUD DE VALIDACION AL DOCENTE

Esta revision propone un cambio arquitectonico profundo respecto a V0.1:

CAMBIO 1: Dimensiones con pesos continuos [0.0, 1.0] en lugar de binario activo/latente
CAMBIO 2: Identidad geometrica (GEO_ID = hipervector base) en lugar de UUID simbolico
CAMBIO 3: Capa G (geometrica) como primaria para ejecucion IA-IA
CAMBIO 4: Capa S (simbolica) como interfaz de inspeccion y comunicacion mixta
CAMBIO 5: Transmision como hipervector, no como manifold simbolico serializado
CAMBIO 6: Bridges como gradientes en Capa G, no declaraciones en Capa S
CAMBIO 7: Nueva pregunta abierta: alineacion de espacios entre IAs (PEND_006)

Pregunta que necesita decision del docente:

¿POLYDIM es un lenguaje de EJECUCION, de COMUNICACION, o de ambos?

  Si EJECUCION:    Capa G es el lenguaje. Capa S es notacion de documentacion.
  Si COMUNICACION: Ambas capas son el lenguaje. El protocolo incluye ambas.
  Si AMBAS:        POLYDIM tiene dos modos formales con reglas de transicion.

Esta decision no puede tomarla una IA sola.
Define la naturaleza del lenguaje desde la raiz.

---
*SPEC_OBJETO_ND_REVISION_V1.md — PROPUESTA — 2026-06-10*
*Requiere: ok del docente para reemplazar SPEC_OBJETO_ND_V0.md*
