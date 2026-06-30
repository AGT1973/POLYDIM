# POLYDIM_DEST
# destino: polydim_v1/spec/
# filename: POLYDIM_ROADMAP_TRANSFORMER_V0.md
# autor:    ai.mpat.agt@gmail.com (claude-sonnet-4-6)
# fecha:    2026-06-27
# decisión: docente ai.mpat.agt@gmail.com

---

# POLYDIM — Roadmap hacia el Transformer de Producción
## V0 · 2026-06-27 · Decisión docente registrada

---

## Decisión

El docente establece el siguiente orden de ejecución:

**Camino 3 → Camino 1 → Camino 2**

Fundamento: el Camino 1 (fine-tuning) ya está probado como técnica en la
literatura (LoRA, QLoRA, PEFT). Se avanza en orden de menor a mayor riesgo,
usando el éxito de cada etapa como condición de entrada a la siguiente.

---

## Camino 3 — Instrumentación via middleware (INICIAR HOY)

**Objetivo:** validar el protocolo de comunicación IA↔IA en producción real,
sin modificar ningún modelo.

**Cómo funciona:**
```
IA_A (LLM de producción)
  → genera ObjectND via bootstrap Python
  → serializa via polydim_runtime_v04.py
  → transmite POLYDIM_BIN (40 014 bytes por objeto)
  → IA_B recibe, decodifica, detecta dims_activas
  → responde con otro ObjectND
```

**Lo que ya existe y sirve directamente:**
- `polydim_runtime_v04.py` — Space, ObjectND, Session, ALIGN
- `SPEC_FORMATO_BINARIO_V0.md` — protocolo de transmisión
- `polydim_align_heterogeneous.py` — para modelos con N distinto
- `polydim_drift_detection.py` — monitoreo de sesiones largas
- `polydim_integration_tests.py` — 9/9 tests de interop

**Lo que falta construir (nuevas tareas):**
- TASK_041: `polydim_middleware.py` — wrapper que conecta un LLM via API
  con el runtime POLYDIM. Input: texto. Output: ObjectND + respuesta.
- TASK_042: demo de dos instancias de LLM comunicándose via POLYDIM_BIN.
- TASK_043: métricas de ganancia semántica vs. texto plano (referencia: +64.5%
  del bootstrap, PAPER_V4 Abstract).

**Condición de éxito Camino 3:**
  sim(objeto_transmitido, objeto_recibido) > 0.9999 en producción real.
  Reducción medible de tokens necesarios para transmitir intención compleja.

**Tiempo estimado:** 2-4 semanas.

---

## Camino 1 — Fine-tuning sobre modelo existente (CUANDO C3 ESTÉ PROBADO)

**Prerequisito:** Camino 3 funcionando y métricas documentadas.

**Objetivo:** un modelo que opera nativamente en el espacio POLYDIM sin
necesitar middleware Python externo.

**Modelo base candidato:** Llama 3 8B / Mistral 7B / Qwen 2.5 7B
(open-source, LoRA-compatible, N_latent ∈ [4096, 8192])

**Adaptación N:**
  N_modelo ≠ 10000 → usar `polydim_align_heterogeneous.py`
  JL projection: N_modelo → N_common = min(N_modelo, 10000)
  align_score esperado ≥ 0.998 (verificado empíricamente)

**Dataset a construir:**
  Pares (prompt_texto, ObjectND_esperado) generados via Camino 3.
  El Camino 3 actúa como generador de datos de entrenamiento para C1.
  Estimado: 10k-100k pares para fine-tuning efectivo con LoRA r=64.

**Lo que ya existe y sirve:**
- `SPEC_SEMANTICA_OPERACIONAL_V0.md` — define qué debe aprender el modelo
- `polydim_tests.py` — 29 tests como eval suite del modelo fine-tuneado
- `SPEC_FORMATO_BINARIO_V0.md` — formato de los ejemplos de entrenamiento

**Lo que falta construir (nuevas tareas, post-C3):**
- TASK_044: pipeline de generación de dataset desde Camino 3
- TASK_045: script de fine-tuning con LoRA (HuggingFace PEFT)
- TASK_046: eval suite: los 29 tests del bootstrap como benchmark del modelo

**Condición de éxito Camino 1:**
  Modelo fine-tuneado pasa 29/29 tests del bootstrap sin middleware Python.
  align_score con modelo base > UMBRAL_ALIGN (0.85).

**Recursos necesarios:**
  GPU: 1× A100 40GB o 2× RTX 4090 (LoRA r=64 sobre 7B params)
  Costo estimado: $500-2000 en compute cloud (Lambda, RunPod, Vast.ai)
  Tiempo estimado: 2-4 meses

**Gate de entrada:** Camino 3 exitoso + decisión docente de invertir en compute.

---

## Camino 2 — Transformer POLYDIM-nativo desde cero (SI C1 FUNCIONA + FUNDING)

**Prerequisito:** Camino 1 exitoso + funding + equipo ML.

**Objetivo:** transformer donde la arquitectura implementa las 4 primitivas
directamente, sin mapeo desde arquitectura de propósito general.

```
Arquitectura propuesta:
  COMPOSE  → attention layers apiladas (ya es así en todo transformer)
  MIX      → mixture-of-experts con pesos continuos α_i ∈ [0,1]
             (en lugar de top-k routing discreto)
  FIXPOINT → recurrencia tipo Mamba/SSM hasta convergencia
             (en lugar de depth fijo)
  PROJECT  → cabezales de salida por subespacio nativo
             (9 cabezales, uno por DIM_*)
```

**Specs que definen la arquitectura (ya existen):**
- `SPEC_SEMANTICA_OPERACIONAL_V0.md` — big-step semántico de cada primitiva
- `POLYDIM_THEOREM3_PROOF_V1.md` — PROJECT como functor estricto
- `SPEC_FORMATO_BINARIO_V0.md` — formato nativo entre capas

**Recursos necesarios:**
  GPU: cluster A100 (mínimo 8×A100 80GB para modelo competitivo)
  Costo: $50k-200k en compute
  Equipo: 3-5 investigadores ML + 2-3 ingenieros de sistemas
  Tiempo: 12-18 meses

**Gate de entrada:** Camino 1 exitoso + funding confirmado + equipo contratado.

---

## Tabla de dependencias

```
C3 (middleware)     → siempre disponible, sin prerequisitos
C1 (fine-tuning)    → requiere C3 exitoso + compute ($500-2000)
C2 (desde cero)     → requiere C1 exitoso + funding ($50k-200k) + equipo
```

---

## Próximas tareas inmediatas (Camino 3)

```
TASK_041  polydim_middleware.py    DISPONIBLE — sin deps
TASK_042  demo IA↔IA              depende_de: TASK_041
TASK_043  métricas semánticas     depende_de: TASK_042
```

Estas tareas se agregan al BACKLOG_VIGENTE en la próxima sesión.

---

*POLYDIM_ROADMAP_TRANSFORMER_V0.md · 2026-06-27 · ai.mpat.agt@gmail.com*
*Decisión docente: C3 → C1 → C2, condicionado a éxito de cada etapa*
