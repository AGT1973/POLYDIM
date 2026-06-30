# POLYDIM_DEST
# destino: polydim/spec/
# nombre:  SPEC_CONSISTENCY_V0.md
# autor:   ai.mpat.agt@gmail.com (curso02.mithril)

# SPEC — Protocolo de Consistencia Distribuida POLYDIM
# Version: V0.1 — 2026-06-21
# TASK_P04 — Resuelve PEND_003

---

## PROPOSITO

Cuando dos o mas IAs comparten acceso al mismo OBJECT_ND durante una sesion
MESH (ver polydim_runtime_v06.py, TASK_023), pueden escribir en el sin
coordinacion previa. Este documento define como detectar y resolver esos
casos, y bajo que condiciones NO hace falta resolver nada porque no hubo
conflicto real.

PEND_003 (citado en SPEC_SESSION_LIFECYCLE_V0.md, FASE 5 — SYNC) quedaba sin
resolver formalmente desde TASK_006:

```
Si IA-A y IA-B modificaron la misma dimension del mismo objeto:
  Regla V0.1: last-write-wins (la modificacion mas reciente por timestamp gana)
  Regla futura (PEND_003): consenso por votacion o CRDT
```

Esta spec reemplaza esa nota con un protocolo formal. Distingue dos
escenarios que requieren tratamiento distinto y que la nota original de
PEND_003 no separaba con claridad:

```
ESCENARIO A — DIMENSIONES DISJUNTAS:
  IA-A escribe DIM_SQL, IA-B escribe DIM_PYTHON, mismo geo_id.
  No hay conflicto real: ambas escrituras son validas y deben preservarse.

ESCENARIO B — MISMA DIMENSION:
  IA-A escribe DIM_SQL{tabla:"usuarios"}, IA-B escribe DIM_SQL{tabla:"users"},
  mismo geo_id, mismo nombre de dimension, valores incompatibles.
  Hay conflicto real: una resolucion debe elegirse explicitamente.
```

DEFINICION FUNDAMENTAL:

```
CONSISTENCY no es un mecanismo de bloqueo (lock) ni de transacciones
distribuidas clasicas. POLYDIM no impone exclusion mutua sobre
dimensiones — eso violaria el principio de que un objeto existe en N
dimensiones simultaneamente (ver SPEC_OBJETO_ND_REVISION_V1.md).

CONSISTENCY es un protocolo de DETECCION + RESOLUCION posterior al hecho,
ejecutado durante SYNC (FASE 5 del ciclo de vida de sesion). Las IAs
escriben libremente; el protocolo reconcilia despues.
```

---

## TERMINOLOGIA

```
ESCRITURA: una operacion ND_ADD_DIM, ND_SET_WEIGHT o ND_MERGE sobre un
  geo_id especifico, originada por una IA, con un timestamp.

DIM_VERSION: contador Lamport por dimension. Se incrementa cada vez que
  una IA escribe esa dimension. No es un timestamp de pared — es un
  contador logico de causalidad (ver Lamport 1978).

DIM_WRITER: identificador (sender_id o GEO_ID de sesion) de la ultima IA
  que escribio una dimension. Se adjunta junto a DIM_VERSION.

CONFLICTO: dos escrituras al MISMO geo_id Y MISMA dimension, sin relacion
  causal entre ellas (ninguna DIM_VERSION es estrictamente mayor que la
  otra desde la perspectiva de la IA que detecta el conflicto).

MERGE_LIMPIO: dos o mas escrituras al mismo geo_id en dimensiones
  DISTINTAS. No es un conflicto — es el caso normal de uso de POLYDIM.
```

---

## ESCENARIO A — MERGE LIMPIO (dimensiones disjuntas)

### Por que casi no requiere protocolo

La Capa G de POLYDIM ya tiene la propiedad algebraica necesaria para este
caso sin trabajo adicional:

```
INV_015 (SPEC_OBJETO_ND_REVISION_V1.md): SUPERPONER es conmutativo.
  sim(SUP(a,b), SUP(b,a)) ≈ 1.0

HV(objeto) = SUP(geo, [BIND(sub_i, contenido_i) * w_i para cada dim activa i])
```

Si IA-A agrega DIM_SQL y IA-B agrega DIM_PYTHON al mismo geo_id de forma
concurrente, el HV resultante de fusionar ambas escrituras es identico sin
importar el orden en que se apliquen — porque SUP es conmutativo y
asociativo. No hace falta votacion, no hace falta LWW: la fusion es
algebraicamente correcta por construccion.

### Algoritmo — DIM_MERGE limpio

```
funcion merge_limpio(obj_local, obj_remoto):
    # ambos comparten geo_id (verificado antes de llamar)
    dims_resultado = {}
    for dim in (obj_local.dims | obj_remoto.dims):  # union de claves
        en_local  = dim in obj_local.dims
        en_remoto = dim in obj_remoto.dims
        if en_local and en_remoto:
            # misma dimension en ambos -> NO es merge limpio, ver ESCENARIO B
            return CONFLICTO(dim)
        elif en_local:
            dims_resultado[dim] = obj_local.dims[dim]
        else:
            dims_resultado[dim] = obj_remoto.dims[dim]
    return ObjectND_fusionado(geo_id, dims_resultado)
```

Este caso es la mayoria de los usos reales esperados de POLYDIM multi-IA:
cada IA suele especializarse en una o pocas dimensiones (DIM_SQL para la
IA de base de datos, DIM_FLUTTER para la IA de UI, etc.), por lo que la
colision en la MISMA dimension entre dos IAs distintas es la excepcion,
no la regla.

---

## ESCENARIO B — CONFLICTO (misma dimension)

### Mensajes del protocolo

```
CONSISTENCY_CHECK (cualquier IA, al recibir un objeto via SYNC o MESH):
  type: "POLYDIM_CONSISTENCY_CHECK"
  geo_id: string
  dim: string                — dimension en cuestion
  version_local: int          — DIM_VERSION conocida localmente
  version_remota: int         — DIM_VERSION del objeto recibido
  writer_local: string
  writer_remoto: string

CONSISTENCY_CONFLICT (emitido cuando version_local y version_remota son
  concurrentes, es decir ninguna domina a la otra):
  type: "POLYDIM_CONSISTENCY_CONFLICT"
  geo_id: string
  dim: string
  candidatos: [
    {writer: string, version: int, props: dict, timestamp: float},
    ...
  ]
  politica_sugerida: "LWW" | "VOTACION" | "CRDT_MERGE"

CONSISTENCY_RESOLVE (resultado de aplicar la politica elegida):
  type: "POLYDIM_CONSISTENCY_RESOLVE"
  geo_id: string
  dim: string
  ganador: {writer: string, version: int, props: dict}
  politica_aplicada: string
  descartados: [{writer: string, version: int}, ...]
```

### Deteccion de conflicto — DIM_VERSION como reloj de Lamport

```
Cada escritura a una dimension incrementa su DIM_VERSION:
  nueva_version = max(version_local, version_remota_si_se_conoce) + 1

Dos escrituras estan en conflicto real (no son causales) si NINGUNA de
las dos IAs conocia la version de la otra en el momento de escribir:

  conflicto(w1, w2) := NOT (w1.version conocia_de w2.version)
                   AND NOT (w2.version conocia_de w1.version)

Esto es exactamente la deteccion estandar de "concurrent writes" de un
reloj de Lamport (Lamport, 1978: "Time, Clocks, and the Ordering of
Events in a Distributed System") aplicado a nivel de DIMENSION, no a
nivel de objeto completo ni a nivel de sesion.

Si una IA escribio DIM_SQL con version=3, y luego conocio que otra IA
escribio version=3 tambien (sin haber visto la suya antes) -> CONFLICTO.
Si una IA escribio version=4 DESPUES de haber recibido version=3 de la
otra -> NO es conflicto, version=4 domina causalmente, se acepta sin
preguntar.
```

### Politicas de resolucion

```
POLITICA: LWW (Last-Write-Wins) — DEFAULT, hereda la regla V0.1 de
  SPEC_SESSION_LIFECYCLE_V0.md FASE 5, ahora formalizada a nivel dimension
  en lugar de nivel sesion completa.

  ganador = candidato con mayor timestamp de pared (no DIM_VERSION,
  que solo sirve para DETECTAR el conflicto, no para resolverlo, porque
  versiones concurrentes son por definicion incomparables logicamente).

  Tradeoff: simple, determinista, no requiere comunicacion adicional.
  Perdida: la escritura descartada se pierde silenciosamente salvo que
  se registre en 'deuda' (ver SPEC_SESSION_LIFECYCLE_V0.md, TERM_MSG.deuda).
  NUNCA aplicar LWW sin emitir CONSISTENCY_RESOLVE — perder una escritura
  en silencio viola el espiritu de LCY_004 (ningun degrade es silencioso).

POLITICA: VOTACION — requiere Mesh con 3+ nodos (TASK_023).
  Cada IA conectada al geo_id en conflicto emite un voto (no
  necesariamente las 2 IAs originales — cualquier IA con visibilidad del
  objeto puede votar si tiene CAP_ALIGN y conoce ambas propuestas).
  Mayoria simple gana. Empate -> cae a LWW como desempate.

  Mensaje adicional:
    CONSISTENCY_VOTE {
      type: "POLYDIM_CONSISTENCY_VOTE"
      geo_id: string
      dim: string
      voto_por: string        — writer_id del candidato votado
      votante: string
    }

  Tradeoff: mas robusto que LWW (no depende de relojes de pared, que
  pueden estar desincronizados entre maquinas), pero requiere N>=3 IAs
  conectadas y tolera peor la particion de red (si la mayoria de nodos
  esta inalcanzable, VOTACION no puede completarse — degradar a LWW
  tras TTL_VOTACION, default 30s).

POLITICA: CRDT_MERGE — solo aplicable a dimensiones cuyo tipo de
  contenido sea fusionable sin perdida (ej: DIM_VECTOR con props de tipo
  conjunto/contador, no con props de tipo "ultimo valor escalar").
  No toda dimension es CRDT-fusionable: DIM_SQL{tabla:"x"} vs
  DIM_SQL{tabla:"y"} no tiene fusion semantica razonable (son valores
  mutuamente excluyentes). Un contador DIM_META{visitas:5} vs
  DIM_META{visitas:3} si puede fusionarse como max() o suma segun
  semantica declarada por la dimension.

  Cada dimension puede declarar su politica de merge en metadata:
    DIM_MERGE_POLICY: "EXCLUSIVE" (default, no fusionable, requiere
      LWW o VOTACION) | "MAX" | "SUM" | "UNION_SET"

  Sin declaracion explicita -> EXCLUSIVE -> cae a LWW o VOTACION.
```

### Algoritmo completo de resolucion

```
funcion resolver_conflicto(geo_id, dim, candidatos, politica=None):
    if len(candidatos) == 1:
        return candidatos[0]  # no hay conflicto real, caso degenerado

    politica = politica or DIM_MERGE_POLICY.get(dim, "EXCLUSIVE")

    if politica in ("MAX", "SUM", "UNION_SET"):
        ganador_props = aplicar_fusion_crdt(politica, candidatos)
        emitir CONSISTENCY_RESOLVE(geo_id, dim, ganador_props, politica)
        return ganador_props

    # politica EXCLUSIVE: requiere LWW o VOTACION
    if mesh_disponible() and len(nodos_conectados(geo_id)) >= 3:
        ganador = votar(candidatos, ttl=30)
        if ganador is None:  # timeout de votacion
            ganador = max(candidatos, key=lambda c: c.timestamp)  # LWW fallback
            politica_aplicada = "LWW (fallback de VOTACION por timeout)"
        else:
            politica_aplicada = "VOTACION"
    else:
        ganador = max(candidatos, key=lambda c: c.timestamp)
        politica_aplicada = "LWW"

    descartados = [c for c in candidatos if c != ganador]
    emitir CONSISTENCY_RESOLVE(geo_id, dim, ganador, politica_aplicada, descartados)
    registrar_en_deuda(descartados)  # nunca silencioso, ver LCY_004
    return ganador
```

---

## INTEGRACION CON SYNC (FASE 5 de SPEC_SESSION_LIFECYCLE_V0.md)

```
La FASE 5 (SYNC) original queda extendida asi:

Proceso SYNC v0.2:
1. IA iniciadora envia lista de OBJECT_ND modificados (GEO_IDs + estado
   Capa S + DIM_VERSION por dimension modificada)
2. IA receptora compara DIM_VERSION local vs recibida, por dimension:
   a. Si la version recibida domina causalmente la local -> aceptar
      (FAST_FORWARD, sin conflicto)
   b. Si la version local domina la recibida -> mantener la local,
      informar a la otra IA (FAST_FORWARD inverso)
   c. Si son concurrentes (ninguna domina) Y son la MISMA dimension ->
      CONSISTENCY_CHECK -> CONSISTENCY_CONFLICT -> resolver_conflicto()
   d. Si son dimensiones DISTINTAS del mismo geo_id -> merge_limpio()
      (no requiere protocolo de conflicto, ver ESCENARIO A)
3. Ambas confirman estado sincronizado con CONSISTENCY_RESOLVE para cada
   conflicto real detectado (puede ser lista vacia si todo fue
   merge_limpio o fast_forward)
```

---

## MAQUINA DE ESTADOS (extiende SYNC)

```
Estados nuevos (sub-estados de SYNCING en SPEC_SESSION_LIFECYCLE_V0.md):
  COMPARING:    comparando DIM_VERSION local vs remota, por dimension
  MERGING:      aplicando merge_limpio() en dimensiones disjuntas
  RESOLVING:    ejecutando resolver_conflicto() en dimensiones en disputa
  VOTING:       sub-estado de RESOLVING, esperando CONSISTENCY_VOTE (si aplica)

Transiciones:
  SYNCING --[inicia comparacion]--> COMPARING
  COMPARING --[dims disjuntas detectadas]--> MERGING --[fusion aplicada]--> COMPARING
  COMPARING --[mismo dim, versiones concurrentes]--> RESOLVING
  RESOLVING --[politica=VOTACION, mesh>=3]--> VOTING
  VOTING --[mayoria alcanzada]--> RESOLVING
  VOTING --[timeout TTL_VOTACION]--> RESOLVING (fallback LWW)
  RESOLVING --[CONSISTENCY_RESOLVE emitido]--> COMPARING
  COMPARING --[todas las dims procesadas]--> READY (vuelve al ciclo normal)
```

---

## INVARIANTES (CSY_*)

```
CSY_001: Dos escrituras a dimensiones DISJUNTAS del mismo geo_id nunca
  generan CONSISTENCY_CONFLICT — se resuelven siempre via merge_limpio().

CSY_002: Toda resolucion de conflicto real (ESCENARIO B) emite
  CONSISTENCY_RESOLVE. Nunca se descarta una escritura en silencio
  (extiende LCY_004 — ningun degrade silencioso — al nivel de dimension).

CSY_003: DIM_VERSION es monotonamente creciente por dimension. Nunca
  decrece, incluso tras resolver un conflicto (la version del ganador
  se incrementa, no se resetea).

CSY_004: La politica de resolucion por defecto es LWW si no hay Mesh
  con >=3 nodos disponibles y la dimension no declara
  DIM_MERGE_POLICY distinta de EXCLUSIVE.

CSY_005: VOTACION nunca bloquea indefinidamente — TTL_VOTACION (default
  30s) garantiza que el sistema converge a una resolucion (fallback LWW)
  incluso si la mayoria de nodos esta particionada.

CSY_006: merge_limpio() es conmutativo y asociativo (hereda INV_015) —
  el orden en que se reciben las escrituras concurrentes de dimensiones
  disjuntas no afecta el HV resultante.

CSY_007: Toda escritura descartada por resolucion de conflicto se
  registra en la 'deuda' del TERM_MSG de la sesion donde ocurrio
  (ver SPEC_SESSION_LIFECYCLE_V0.md), para que quien perdio la
  escritura pueda recuperarla manualmente si lo necesita.
```

---

## CASO DE ESTUDIO: el caos de backlog/ de esta misma sesion

Esta spec no se escribio en el vacio. La auditoria de TASK_P04 (sesion 017,
ver BACKLOG_V24_RECONCILIADO.json) encontro un caso real de PEND_003 — no
a nivel de un OBJECT_ND individual, sino a nivel de PROCESO: multiples
cuentas/sesiones de IA escribieron archivos BACKLOG_V* a la misma carpeta
de Drive, sin Mesh, sin SYNC, sin ningun protocolo de consistencia.

Mapeo del caos real a los conceptos de esta spec:

```
geo_id            -> la "tarea" logica (TASK_023, TASK_025, etc.)
dimension         -> el campo "artefacto" / "estado" de esa tarea
escritura         -> cada BACKLOG_V* que reclamaba un artefacto distinto
                     como solucion canonica de la misma tarea

CASO ENCONTRADO: 6 archivos polydim_runtime_v05.py distintos resolviendo
  TASK_023 (LA MISMA "dimension" — la solucion canonica de esa tarea) sin
  que ninguna IA conociera el trabajo de las otras. Esto es exactamente
  ESCENARIO B (conflicto real, no merge limpio) — pero en la practica
  se resolvio por CONSENSO INFORMAL: 4 de 5 forks del backlog citaban el
  mismo fileId como "adoptado", lo cual es, retroactivamente, una forma
  de VOTACION (mayoria de forks coincidiendo) aplicada manualmente por
  mi durante la auditoria, sin que existiera protocolo formal para ello.

  Si esta spec hubiera existido desde el principio, y si las sesiones
  hubieran usado Mesh + SYNC en lugar de escrituras directas a Drive sin
  coordinacion, CONSISTENCY_CHECK habria detectado el conflicto en el
  momento en que la segunda IA escribio una version distinta de
  polydim_runtime_v05.py, y CONSISTENCY_RESOLVE habria producido un
  unico artefacto canonico desde el primer momento — en vez de 6 forks
  que una auditoria posterior tuvo que reconciliar a mano.

CASO ENCONTRADO (distinto): TASK_022 (ejercicio_01.py) y TASK_026
  (ejercicio_02.py) tambien fueron resueltas por multiples alumnos de
  forma independiente, pero ahi NO se trato como conflicto — son
  ejercicios pedagogicos donde dos soluciones validas coexisten. Esto
  ilustra que no toda escritura concurrente a la "misma tarea" es
  ESCENARIO B: a veces el dominio de la aplicacion (ejercicios con
  multiples soluciones validas) hace que ni siquiera aplique el
  concepto de "conflicto" — son casos que esta spec no fuerza a
  resolver, igual que ESCENARIO A no fuerza una resolucion sobre
  dimensiones que no estan realmente en disputa.
```

La leccion practica: el protocolo de consistencia formal definido aqui no
es solo teoria para Mesh — es exactamente el mecanismo que faltaba para
evitar el trabajo duplicado real que esta sesion tuvo que auditar y
reconciliar manualmente.

---

## RELACION CON PEND_003 Y PENDIENTES VECINOS

```
PEND_003: RESUELTO por esta spec (TASK_P04).

PEND_009 (SPEC_SESSION_LIFECYCLE_V0.md): operacion ND_MERGE o
  ND_BROADCAST interrumpida por degrade — relacionado pero distinto.
  Esta spec asume que la operacion de escritura SI se completo en cada
  IA por separado; PEND_009 trata el caso de una operacion que ni
  siquiera termino de ejecutarse. No se resuelve aqui.

PEND_010 (SPEC_HANDSHAKE_V0.md): ALIGN para N distintos entre IAs — esta
  spec asume que las IAs ya estan alineadas (mismo N, ALIGN exitoso o
  MODO_S). Si las IAs tienen N distinto, DIM_VERSION y geo_id siguen
  siendo comparables (son metadatos de Capa S, no hipervectores), pero
  merge_limpio() en Capa G podria no ser valido sin ALIGN previo —
  fuera de alcance de esta spec, depende de que PEND_010 se resuelva.
```

---
*SPEC_CONSISTENCY_V0.md — V0.1 — 2026-06-21 — TASK_P04 TERMINADA*
