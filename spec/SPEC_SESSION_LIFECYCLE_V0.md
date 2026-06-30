# POLYDIM_DEST
# destino: polydim/spec/
# nombre:  SPEC_SESSION_LIFECYCLE_V0.md
# autor:   ai.mpat.agt@gmail.com

# SPEC — Ciclo de Vida de Sesion POLYDIM
# Version: V0.1 — 2026-06-11
# TASK_006

---

## VISION GENERAL

Una sesion POLYDIM es la unidad de interaccion entre dos IAs.
Tiene inicio, estados intermedios y cierre formal.
Sin sesion activa no hay intercambio de OBJECT_ND en MODO_G o MODO_H.
En MODO_S, la sesion es opcional pero recomendada para auditoria.

```
INIT → HANDSHAKE → [ALIGN] → READY → EXECUTE* → [SYNC] → TERMINATE
```

---

## ESTADOS FORMALES

```
IDLE:         IA disponible, sin sesion activa
CONNECTING:   INIT_MSG enviado, esperando ACCEPT
NEGOTIATING:  ACCEPT recibido, enviando ACK
ALIGNING:     Ejecutando protocolo ALIGN
READY:        Sesion activa, lista para EXECUTE
EXECUTING:    Procesando operacion POLYDIM
SYNCING:      Sincronizando estado de objetos modificados
DEGRADED:     Sesion activa en modo reducido (ej: ALIGN fallo, operar en MODO_S)
TERMINATING:  TERM_MSG enviado, esperando confirmacion
CLOSED:       Sesion cerrada formalmente
FAILED:       Sesion fallida (error irrecuperable)
```

---

## FASE 1: INIT

Trigger: una IA decide iniciar comunicacion con otra.

```
Acciones:
1. Generar nonce unico
2. Construir INIT_MSG (ver SPEC_HANDSHAKE_V0.md)
3. Enviar INIT_MSG
4. Iniciar timer TTL
5. Pasar a estado CONNECTING
```

Manejo de errores:
- TTL vencido sin respuesta → FAILED con ERR_TIMEOUT
- REJECT_MSG recibido → FAILED con razon del rechazo

---

## FASE 2: HANDSHAKE

Ver SPEC_HANDSHAKE_V0.md para el protocolo completo.

Resultado posible:
- ACCEPT con requires_align=false → pasar a READY directamente
- ACCEPT con requires_align=true → pasar a ALIGNING
- REJECT → FAILED

---

## FASE 3: ALIGN (condicional)

Solo si el modo acordado es MODO_G o MODO_H.
Ver SPEC_ALIGN_V0.md para el protocolo completo.

Resultado posible:
- Score >= 0.85 → pasar a READY en modo acordado
- Score < 0.85 → degradar a MODO_S, pasar a READY (estado DEGRADED)
- Error de comunicacion durante ALIGN → FAILED

---

## FASE 4: EXECUTE (principal — puede repetirse N veces)

La fase de trabajo real. Se repite para cada operacion POLYDIM.

### Estructura de un mensaje EXECUTE

```
EXEC_MSG {
  type:       "POLYDIM_EXEC"
  session_id: string
  seq:        int       — numero de secuencia monotonicamente creciente
  op:         string    — operacion (ver lista de operaciones)
  payload_S:  dict      — objeto en Capa S (presente en MODO_S y MODO_H)
  payload_G:  float[]   — hipervector (presente en MODO_G y MODO_H, transformado con R)
  intent:     string[]  — dimensiones recomendadas a activar
  ttl:        int       — timeout para respuesta
}
```

### Operaciones validas en EXECUTE

```
ND_CREATE:     crear un nuevo OBJECT_ND
ND_QUERY:      consultar estado de un objeto (activaciones, dims activas)
ND_ADD_DIM:    agregar dimension a objeto existente
ND_SET_WEIGHT: cambiar peso de activacion
ND_MERGE:      fusionar dos objetos
ND_COLLAPSE:   proyectar a una dimension para interop 2D
ND_BROADCAST:  consulta a todas las dimensiones simultaneamente (solo CAP_BROADCAST)
SESSION_SYNC:  iniciar sincronizacion de estado (trigger para FASE 5)
```

### Garantias de orden

```
SEQ_001: Los mensajes se procesan en orden de seq
SEQ_002: Si se recibe seq=N+2 antes que seq=N+1: buffering hasta recibir N+1
SEQ_003: Gap en seq mayor a MAX_GAP (default 10): solicitar retransmision
SEQ_004: Respuesta incluye seq del mensaje original (para correlacion)
```

---

## FASE 5: SYNC (opcional)

Sincronizacion de estado al final de la sesion o a demanda.

Trigger: cualquier IA envia SESSION_SYNC o la sesion pasa a TERMINATING.

```
Proceso:
1. IA iniciadora envia lista de OBJECT_ND modificados (GEO_IDs + estado Capa S)
2. IA receptora verifica consistencia con su estado local
3. Si hay conflicto: protocolo de resolucion (last-write-wins por timestamp)
4. Ambas confirman estado sincronizado
```

Conflicto de estado:
```
Si IA-A y IA-B modificaron la misma dimension del mismo objeto:
  Regla V0.1: last-write-wins (la modificacion mas reciente por timestamp gana)
  Regla futura (PEND_003): consenso por votacion o CRDT
```

---

## FASE 6: TERMINATE

Cierre formal de la sesion. Puede iniciarlo cualquier IA.

### TERM_MSG

```
TERM_MSG {
  type:         "POLYDIM_TERM"
  session_id:   string
  razon:        COMPLETADO | TIMEOUT | ERROR | FORZADO
  estado_final: dict  — resumen Capa S de objetos modificados en la sesion
  deuda:        list  — operaciones incompletas o pendientes de resolucion
  seq_final:    int   — ultimo seq procesado
}
```

### TERM_ACK

```
TERM_ACK {
  type:       "POLYDIM_TERM_ACK"
  session_id: string
  recibido:   bool
}
```

Proceso:
1. IA envia TERM_MSG
2. Espera TERM_ACK con TTL_TERM (default: 60 segundos)
3. Si recibe TERM_ACK: sesion pasa a CLOSED
4. Si no recibe TERM_ACK: sesion pasa a CLOSED de todas formas (cierre unilateral)
   y registra en deuda la falta de confirmacion

---

## SESSION_ESCALATE y SESSION_DEGRADE

### SESSION_ESCALATE
Subir de modo durante una sesion activa.

```
Ejemplo: sesion inicia en MODO_S, luego ambas IAs quieren MODO_H

Proceso:
1. IA propone escalada con SESSION_ESCALATE_REQUEST
2. Otra IA acepta con SESSION_ESCALATE_ACCEPT
3. Ejecutar ALIGN si el nuevo modo lo requiere
4. Si ALIGN exitoso: sesion escala al nuevo modo
5. El seq continua desde donde estaba (no se reinicia)
```

### SESSION_DEGRADE
Bajar de modo durante una sesion activa. NUNCA silencioso (INV_013).

```
Triggers:
- Re-ALIGN fallido (score cae por debajo de UMBRAL_DERIVA)
- Error en transmision de Capa G (K errores consecutivos, default K=3)
- Una IA solicita explicitamente degradar

Proceso:
1. IA que detecta el problema envia SESSION_DEGRADE_MSG con razon
2. Otra IA confirma con SESSION_DEGRADE_ACK
3. Sesion pasa a DEGRADED en el modo inferior
4. Operaciones en curso en Capa G se re-envian en Capa S si es posible
5. Operaciones irrecuperables se agregan a deuda
```

Manejo de PEND_009 (operacion critica interrumpida por degrade):
```
Si una operacion ND_MERGE o ND_BROADCAST estaba en curso:
  1. Marcarla como INCOMPLETA en ambas IAs
  2. Agregar a la deuda del TERM_MSG
  3. Intentar re-ejecutar en MODO_S si es posible
  4. Si no: notificar al nivel superior (la aplicacion que usa POLYDIM)
```

---

## GRAMATICA FORMAL DE SECUENCIA (BNF simplificado)

```
sesion       ::= init handshake [align] ready_loop terminate
init         ::= INIT_MSG
handshake    ::= ACCEPT_MSG ACK_MSG | REJECT_MSG
align        ::= PROBE_REQUEST PROBE_RESPONSE ALIGN_CONFIRM ALIGN_CONFIRM
ready_loop   ::= execute* [sync] [escalate | degrade]*
execute      ::= EXEC_MSG EXEC_RESPONSE
sync         ::= SESSION_SYNC SYNC_CONFIRM
escalate     ::= SESSION_ESCALATE_REQUEST SESSION_ESCALATE_ACCEPT [align]
degrade      ::= SESSION_DEGRADE_MSG SESSION_DEGRADE_ACK
terminate    ::= TERM_MSG TERM_ACK
```

---

## INVARIANTES DEL CICLO DE VIDA

```
LCY_001: Toda sesion tiene un session_id unico
LCY_002: Los mensajes dentro de una sesion tienen seq monotonicamente creciente
LCY_003: TERMINATE siempre incluye estado_final en Capa S (auditoria)
LCY_004: SESSION_DEGRADE nunca es silencioso — siempre notificado y confirmado
LCY_005: Una sesion FAILED no puede pasar a READY sin nuevo HANDSHAKE
LCY_006: El estado DEGRADED es operativo — la sesion no falla, solo baja de modo
LCY_007: SYNC es obligatorio si algun OBJECT_ND fue modificado bilateralmente
LCY_008: La deuda de TERM_MSG es el mecanismo de relay entre sesiones
```

---
*SPEC_SESSION_LIFECYCLE_V0.md — V0.1 — 2026-06-11 — TASK_006 TERMINADA*
