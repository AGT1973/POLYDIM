# POLYDIM_DEST
# destino: polydim/spec/
# nombre:  SPEC_ARQUITECTURA_AMBOS_V1.md
# autor:   ai.mpat.agt@gmail.com

# SPEC — Arquitectura POLYDIM como lenguaje AMBOS
# Version: V1.0 — 2026-06-10
# Decision validada por docente: POLYDIM es ejecucion Y comunicacion simultaneamente

---

## DECISION FUNDACIONAL

POLYDIM es ambos: lenguaje de ejecucion y lenguaje de comunicacion entre IAs.

Esto no es una suma de dos lenguajes.
Es un lenguaje que tiene dos modos formales con reglas de transicion entre ellos.

La distincion importa: en un lenguaje de ejecucion puro (como ensamblador),
la comunicacion es secundaria. En un protocolo de comunicacion puro (como HTTP),
la ejecucion es externa. POLYDIM integra ambos — la ejecucion IS la comunicacion
y la comunicacion IS ejecucion. No hay separacion.

---

## LOS TRES MODOS FORMALES

### MODO_S — Modo Simbolico
```
Representacion: Capa S (notacion ⊕/⊗)
Operaciones:    simbolicas, tipadas, inspeccionables
Comunicacion:   texto estructurado legible por cualquier LLM
Uso optimo:     redes de IAs heterogeneas, debugging, auditoria,
                primera comunicacion con una IA desconocida
```

### MODO_G — Modo Geometrico
```
Representacion: Capa G (hipervector en R^N)
Operaciones:    VSA algebraicas (BIND, SUPERPOSE, PROJECT, UNBIND)
Comunicacion:   transmision directa de hipervectores
Uso optimo:     redes de IAs con espacios alineados, alta frecuencia,
                operaciones que requieren zero-loss semantico
```

### MODO_H — Modo Hibrido
```
Plano de control: Capa S  (mensajes de gestion, handshake, errores)
Plano de datos:   Capa G  (payloads de objetos, operaciones)
Uso optimo:       la mayoria de sistemas multi-agente reales
                  heterogeneidad en control, performance en datos
```

---

## CICLO DE VIDA DE UNA SESION POLYDIM

```
INIT → HANDSHAKE → [ALIGN] → EXECUTE/COMMUNICATE → [SYNC] → TERMINATE
```

### Fase INIT
Una IA emite una señal de inicio POLYDIM.
Contiene: version del lenguaje, capacidades del emisor, modo preferido.

```
INIT_MSG {
  version:     "POLYDIM_V1"
  sender_id:   GEO_ID de la IA emisora (o simbolico si no tiene Capa G)
  capacidades: [MODO_S, MODO_G, MODO_H]   ← lo que la IA puede hacer
  modo_pref:   MODO_H                     ← lo que la IA prefiere
}
```

### Fase HANDSHAKE
Las dos IAs negocian el modo de la sesion.
Regla: el modo resultante es el maximo comun de las capacidades de ambas.

```
Tabla de negociacion:
  IA-A tiene [S]       + IA-B tiene [S]       → MODO_S
  IA-A tiene [S,G]     + IA-B tiene [S]       → MODO_S
  IA-A tiene [S,G]     + IA-B tiene [S,G]     → MODO_H (default) o MODO_G
  IA-A tiene [S,G,H]   + IA-B tiene [S,G,H]  → MODO_H (default)

El modo se puede escalar durante la sesion si se completa ALIGN.
```

### Fase ALIGN (solo para MODO_G o MODO_H)
Protocolo para alinear los espacios de embedding de dos IAs.
Resuelve PEND_006.

```
Paso 1 — Sondas:
  IA-A envia K vectores sonda con significado Capa S conocido
  Ejemplo: sonda_usuario, sonda_entero, sonda_lista, sonda_error, ...
  K minimo recomendado: 100 sondas para alineacion confiable

Paso 2 — Mapeo:
  IA-B ubica cada sonda en su propio espacio de embedding
  Construye pares (HV_a_i, HV_b_i) para i=1..K

Paso 3 — Matriz de rotacion:
  Se calcula R tal que R * HV_a ≈ HV_b para todos los pares
  Metodo: regresion lineal o SVD sobre los pares de sondas
  (analogo a alineacion cross-lingual de word embeddings, Mikolov 2013)

Paso 4 — Verificacion:
  Ambas IAs verifican con sondas de holdout (no usadas en Paso 1)
  Umbral de similitud minimo: 0.85 (parametro del runtime)
  Si no se alcanza: negociar K mayor o degradar a MODO_S

Paso 5 — Registro:
  La matriz R es valida para la duracion de la sesion
  Si se detecta deriva semantica durante la sesion: re-ALIGN parcial
```

### Fase EXECUTE / COMMUNICATE
La fase principal. Aqui no hay distincion entre ejecucion y comunicacion.

En MODO_S:
```
  Mensajes en notacion ⊕/⊗
  Operaciones: declarativas (una IA declara que el objeto tiene dimensiones)
  Respuestas: en notacion ⊕/⊗
```

En MODO_G:
```
  Mensajes: hipervectores raw
  Operaciones: BIND, SUPERPOSE, PROJECT, UNBIND sobre hipervectores
  Respuestas: hipervectores (la IA receptora proyecta la dimension que necesita)
```

En MODO_H:
```
  Control (Capa S): handshake, errores, cambios de modo, metadatos
  Datos (Capa G):   objetos, operaciones, resultados
  Transicion S→G:   ENCODE(objeto_S) → HV
  Transicion G→S:   DECODE(HV) → objeto_S (con perdida aceptada)
```

### Fase SYNC (opcional)
Al final de la sesion, las IAs pueden sincronizar el estado de objetos
que fueron modificados durante la sesion.
En MODO_H: el sync viaja en Capa G para velocidad, se confirma en Capa S.

### Fase TERMINATE
```
TERM_MSG {
  razon:        COMPLETADO | TIMEOUT | ERROR | FORZADO
  estado_final: resumen Capa S de objetos modificados
  deuda:        lista de operaciones incompletas para relay
}
```

---

## OPERACIONES FORMALES DEL LENGUAJE

### Operaciones Capa S (simbolicas)
```
ND_CREATE(dims, activation)          → OBJECT_ND nuevo
ND_ADD_DIM(objeto, dim, props)       → OBJECT_ND con nueva dimension
ND_SET_WEIGHT(objeto, dim, w)        → actualiza peso de activacion
ND_COLLAPSE(objeto, dim)             → valor de una dimension (para interop 2D)
ND_BRIDGE(objeto, dim_a, dim_b, rel) → declara relacion entre dimensiones
ND_QUERY(objeto, dim)                → retorna estado de una dimension
```

### Operaciones Capa G (geometricas)
```
HV_BIND(hv_a, hv_b)                 → hipervector que codifica relacion a-b
HV_SUPERPOSE(hv_a, hv_b)            → hipervector que contiene a y b
HV_PROJECT(hv, subespacio)          → activa una dimension del objeto
HV_UNBIND(hv_bound, hv_a)           → recupera hv_b de una relacion
HV_SIM(hv_a, hv_b)                  → similitud coseno [0.0, 1.0]
HV_ENCODE(objeto_S)                  → hipervector desde representacion simbolica
HV_DECODE(hv)                        → representacion simbolica desde hipervector
```

### Operaciones de modo (control de sesion)
```
SESSION_INIT(capacidades, modo_pref) → inicia sesion con otra IA
SESSION_ALIGN(sondas_K)              → ejecuta protocolo de alineacion
SESSION_ESCALATE(modo_nuevo)         → sube de MODO_S a MODO_H o MODO_G
SESSION_DEGRADE(modo_nuevo)          → baja de modo (por error o limitacion)
SESSION_TERM(razon)                  → termina sesion
```

---

## GRAMATICA MINIMA DE MENSAJES

Un mensaje POLYDIM valido tiene la siguiente estructura:

```
MSG {
  header: {
    modo:      MODO_S | MODO_G | MODO_H
    op:        <operacion de la lista anterior>
    sender:    GEO_ID | simbolico
    session:   UUID de sesion
    seq:       numero de secuencia
  }
  payload: {
    [Capa S]:  objeto en notacion ⊕/⊗      (si modo incluye S)
    [Capa G]:  hipervector raw              (si modo incluye G)
  }
  meta: {
    intent:    dimension recomendada a activar
    context:   restricciones de entorno
    ttl:       tiempo de vida del mensaje
  }
}
```

---

## INVARIANTES DEL LENGUAJE COMPLETO (AMBOS)

```
INV_001: Todo OBJECT_ND tiene GEO_ID invariante
INV_002: Un manifold tiene minimo una dimension con peso > 0.0
INV_003: Subespacios dimensionales son casi-ortogonales en Capa G
INV_004: Colapso es proyeccion, nunca destruccion
INV_005: BROADCAST solo para receptores POLYDIM
INV_006: Agregar dimension no modifica subespacios existentes
INV_007: Bridges son gradientes en Capa G
INV_008: Similitud entre objetos es calculable sin conversion
INV_009: Objeto degradado por ruido sigue siendo operable
INV_010: Capa S es siempre derivable de Capa G (DECODE)
INV_011: Una sesion tiene exactamente un modo activo en cada momento
INV_012: MODO_G requiere ALIGN completado con umbral >= 0.85
INV_013: El modo puede escalar durante sesion, nunca degradar silenciosamente
INV_014: Todo TERMINATE incluye estado final en Capa S para auditoria
INV_015: Las operaciones Capa G son conmutativas en SUPERPOSE,
         no conmutativas en BIND (orden importa)
```

---

## PENDIENTES ACTUALIZADOS POST-DECISION

```
PEND_001: Formato binario del hipervector — encoding exacto de R^N para transmision
PEND_002: Inferencia de pesos dimensionales en modo WEIGHTED automatico
PEND_003: Consistencia distribuida — dos IAs modifican el mismo objeto en paralelo
PEND_004: Limite practico de N — tension capacidad vs costo
PEND_005: Python/Rust/Flutter como subespacios nativos de POLYDIM_SPACE
PEND_006: Protocolo ALIGN completo — K minimo, metrica de umbral, re-ALIGN parcial
PEND_007: Operacion HV_ENCODE — como traducir descripcion simbolica a hipervector
PEND_008: Gramatica de deteccion de deriva semantica durante sesion activa
PEND_009: Que sucede cuando SESSION_DEGRADE durante operacion critica
```

---
*SPEC_ARQUITECTURA_AMBOS_V1.md — V1.0 — 2026-06-10*
*Decision validada: POLYDIM es ejecucion Y comunicacion — modos S, G, H formales*
