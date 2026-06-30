# POLYDIM_DEST
# destino: polydim/spec/
# nombre:  SPEC_HANDSHAKE_V0.md
# autor:   ai.mpat.agt@gmail.com

# SPEC — Protocolo HANDSHAKE
# Version: V0.1 — 2026-06-11
# TASK_003

---

## PROPOSITO

El HANDSHAKE es el protocolo mediante el cual dos IAs establecen una sesion POLYDIM.
Define que modo van a usar (S, G o H) antes de intercambiar cualquier OBJECT_ND.

Sin HANDSHAKE no hay sesion. Una IA que recibe un OBJECT_ND sin sesion previa
lo trata como MODO_S por defecto — la capa mas segura y universal.

---

## MENSAJES DEL PROTOCOLO

### INIT_MSG
Primer mensaje de la sesion. Lo envia la IA que inicia.

```
INIT_MSG {
  type:           "POLYDIM_INIT"
  version:        "POLYDIM_V1"
  sender_id:      string   — GEO_ID en hex (si tiene Capa G) | nombre simbolico
  capabilities:   list     — subconjunto de [CAP_S, CAP_G, CAP_ALIGN, CAP_BROADCAST]
  preferred_mode: string   — MODO_S | MODO_G | MODO_H
  N:              int      — POLYDIM_N del emisor (relevante para CAP_G)
  seed:           string   — semilla del espacio ("POLYDIM_V1_SEED_2026" para estandar)
  nonce:          string   — valor aleatorio unico por sesion (para identificacion)
  ttl:            int      — segundos maximos para recibir ACCEPT antes de TIMEOUT
}
```

### ACCEPT_MSG
Respuesta positiva. Lo envia la IA receptora si acepta la sesion.

```
ACCEPT_MSG {
  type:           "POLYDIM_ACCEPT"
  session_id:     string   — BIND(nonce_A, nonce_B) como hex — identidad de la sesion
  sender_id:      string   — GEO_ID o nombre simbolico del receptor
  agreed_mode:    string   — modo resultante de la negociacion
  capabilities:   list     — capacidades del receptor
  N:              int      — N del receptor
  nonce:          string   — nonce propio del receptor (para construir session_id)
  requires_align: bool     — true si agreed_mode requiere ALIGN antes de operar
}
```

### REJECT_MSG
Respuesta negativa. La sesion no se establece.

```
REJECT_MSG {
  type:    "POLYDIM_REJECT"
  reason:  string  — codigo de error (ver tabla de errores)
  detail:  string  — descripcion opcional
}
```

### ACK_MSG
Confirmacion del iniciador. Completa el HANDSHAKE.

```
ACK_MSG {
  type:       "POLYDIM_ACK"
  session_id: string  — confirma el session_id acordado
  ready:      bool    — true si esta listo para operar (o para ALIGN si requires_align)
}
```

---

## CAPACIDADES (CAP_*)

```
CAP_S:         puede procesar notacion Capa S (⊕/⊗)
               toda IA POLYDIM-compatible DEBE tener CAP_S

CAP_G:         puede procesar hipervectores (tiene numpy o equivalente)
               requiere N y seed compatibles con el interlocutor

CAP_ALIGN:     puede ejecutar el protocolo ALIGN
               requiere CAP_G

CAP_BROADCAST: puede emitir y recibir respuestas BROADCAST
               solo disponible en MODO_G y MODO_H

Minimo obligatorio: [CAP_S]
Maximo posible:     [CAP_S, CAP_G, CAP_ALIGN, CAP_BROADCAST]
```

---

## NEGOCIACION DE MODO

El modo resultante se determina por la interseccion de capacidades de ambas IAs,
priorizando el modo mas rico posible.

```
Regla general: modo = max_modo_comun(caps_A, caps_B)

Tabla completa:

IA-A capabilities        IA-B capabilities        Modo resultante
[CAP_S]                  [CAP_S]                  MODO_S
[CAP_S]                  [CAP_S, CAP_G]           MODO_S
[CAP_S, CAP_G]           [CAP_S, CAP_G]           MODO_H (default) *
[CAP_S, CAP_G, CAP_ALIGN][CAP_S, CAP_G, CAP_ALIGN]MODO_H con ALIGN disponible
cualquiera               cualquiera, N distinto    MODO_S **
cualquiera               cualquiera, seed distinta  MODO_S ***

* MODO_H es el default cuando ambos tienen CAP_G, porque:
  - Plano de control en Capa S (mas robusto para errores)
  - Plano de datos en Capa G (performance)
  Una IA puede proponer MODO_G en preferred_mode si quiere puro geometrico.

** Si N distinto: los espacios no son compatibles sin transformacion.
   MODO_S es el unico seguro. MODO_G requiere ALIGN especial (PEND_010).

*** Si seed distinta: los simbolos deterministicos son distintos.
   Los nombres de dimension coinciden pero sus vectores no.
   MODO_S es seguro porque usa nombres, no vectores.
   MODO_G requiere mapeo de simbolos (extension futura).
```

---

## MAQUINA DE ESTADOS DEL HANDSHAKE

```
Estados:
  IDLE          IA espera o inicia
  PROPOSED      INIT_MSG enviado, esperando respuesta
  NEGOTIATING   ACCEPT_MSG recibido, verificando compatibilidad
  ALIGNING      ALIGN en curso (solo si requires_align=true)
  READY         Sesion establecida y operativa
  FAILED        Handshake fallido (REJECT o TIMEOUT)

Transiciones:
  IDLE        --[enviar INIT_MSG]-->      PROPOSED
  PROPOSED    --[recibir ACCEPT_MSG]-->   NEGOTIATING
  PROPOSED    --[recibir REJECT_MSG]-->   FAILED
  PROPOSED    --[timeout TTL]-->          FAILED
  NEGOTIATING --[enviar ACK_MSG]--> ALIGNING (si requires_align) | READY
  ALIGNING    --[ALIGN exitoso]-->        READY
  ALIGNING    --[ALIGN fallido]-->        degradar a MODO_S → READY | FAILED
  READY       --[SESSION_DEGRADE]-->      re-HANDSHAKE parcial
  READY       --[SESSION_TERM]-->         IDLE
```

---

## CALCULO DEL SESSION_ID

El session_id identifica univocamente la sesion. Se calcula asi:

```
session_id = HV_BIND(hv(nonce_A), hv(nonce_B))

donde hv(nonce) = hipervector deterministico desde el nonce
                 (mismo mecanismo que SPACE._make(name))

Propiedad: session_id es unico por par de nonces
           cualquiera de las dos IAs puede verificarlo
           no requiere coordinacion previa
```

---

## TABLA DE ERRORES

```
ERR_VERSION_MISMATCH   Version POLYDIM incompatible
ERR_NO_COMMON_MODE     Sin modo compatible (ej: IA-B no tiene CAP_S — imposible)
ERR_N_MISMATCH         N distintos y sin soporte para ALIGN cross-N
ERR_SEED_MISMATCH      Seeds distintas sin soporte para mapeo de simbolos
ERR_TIMEOUT            No se recibio respuesta dentro del TTL
ERR_NONCE_COLLISION    Nonce ya usado (replay attack o colision)
ERR_CAP_REQUIRED       Operacion requiere capacidad no declarada
```

---

## EJEMPLO COMPLETO — HANDSHAKE MODO_H

```
IA-A (tiene CAP_S, CAP_G, CAP_ALIGN):

  Envia INIT_MSG:
    version:        "POLYDIM_V1"
    sender_id:      "a3f7geo..."
    capabilities:   [CAP_S, CAP_G, CAP_ALIGN]
    preferred_mode: MODO_H
    N:              10000
    seed:           "POLYDIM_V1_SEED_2026"
    nonce:          "x9k2m..."
    ttl:            30

IA-B (tiene CAP_S, CAP_G, CAP_ALIGN):

  Recibe INIT_MSG. Verifica: N igual, seed igual, caps compatibles.
  Calcula: modo = MODO_H. requires_align = true.

  Envia ACCEPT_MSG:
    session_id:     HV_BIND(hv("x9k2m..."), hv("b8r4n..."))
    sender_id:      "b8r4geo..."
    agreed_mode:    MODO_H
    capabilities:   [CAP_S, CAP_G, CAP_ALIGN]
    N:              10000
    nonce:          "b8r4n..."
    requires_align: true

IA-A:

  Recibe ACCEPT_MSG. Verifica session_id.

  Envia ACK_MSG:
    session_id:     [mismo]
    ready:          true

  → Ambas IAs pasan a estado ALIGNING → ejecutan SPEC_ALIGN_V0
  → Si ALIGN exitoso: READY en MODO_H
```

---

## EJEMPLO — HANDSHAKE MODO_S (IAs heterogeneas)

```
IA-A (solo CAP_S):

  Envia INIT_MSG:
    capabilities:   [CAP_S]
    preferred_mode: MODO_S
    N:              0  (no aplica)
    seed:           "POLYDIM_V1_SEED_2026"

IA-B (tiene CAP_S, CAP_G):

  Recibe INIT_MSG. IA-A no tiene CAP_G.
  Modo resultante: MODO_S.

  Envia ACCEPT_MSG:
    agreed_mode:    MODO_S
    requires_align: false

IA-A:
  Envia ACK_MSG. → READY en MODO_S inmediatamente.
```

---

## INVARIANTES DEL HANDSHAKE

```
HND_001: Toda sesion tiene un session_id unico derivado de los nonces
HND_002: El modo acordado es siempre el maximo comun de capacidades
HND_003: MODO_G y MODO_H requieren requires_align=true si N y seed coinciden
HND_004: Un REJECT no puede ser ignorado — la sesion no existe
HND_005: El TTL es responsabilidad del iniciador — vencido = FAILED sin REJECT
HND_006: MODO_S siempre es posible si ambas IAs son POLYDIM-compatibles
         (CAP_S es obligatorio)
HND_007: El session_id es verificable independientemente por ambas IAs
```

---
*SPEC_HANDSHAKE_V0.md — V0.1 — 2026-06-11 — TASK_003 TERMINADA*
