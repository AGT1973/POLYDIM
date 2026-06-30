SPEC_ALIGN_V1 — Protocolo de Alineacion POLYDIM
Fecha: 2026-06-12
Reemplaza: SPEC_ALIGN_V0.md (1rvqQx0I1nugwlCTUiSV2UfCVs7I5DB2w)
Incorpora: hallazgos sesiones 003 y 005 (SubspaceAligner, K_EFECTIVO)
Autor: ai.mpat.agt@gmail.com

═══════════════════════════════════════════════════════════════
1. PROPOSITO
═══════════════════════════════════════════════════════════════

El protocolo ALIGN determina si dos instancias POLYDIM pueden operar en
MODO_G o MODO_H, o si deben usar MODO_S.

DEFINICION FUNDAMENTAL (revision V1):
  ALIGN no alinea espacios incompatibles.
  ALIGN DETECTA si los espacios son suficientemente compatibles para MODO_H.

Esta distincion es central. ALIGN es un test de calidad, no un mecanismo
de correccion. Dos IAs con espacios incompatibles siempre usan MODO_S.
No hay degradacion de calidad en MODO_S — es el modo universal robusto.

═══════════════════════════════════════════════════════════════
2. CUANDO SE EJECUTA
═══════════════════════════════════════════════════════════════

ALIGN se ejecuta cuando:
  - HANDSHAKE negocio MODO_H o MODO_G
  - ambas IAs tienen CAP_ALIGN
  - N y POLYDIM_SEED son iguales en ambas IAs

Si N o POLYDIM_SEED difieren → MODO_S sin ALIGN (decidido en HANDSHAKE).
Si alguna IA no tiene CAP_ALIGN → MODO_S sin ALIGN.

═══════════════════════════════════════════════════════════════
3. PARAMETROS
═══════════════════════════════════════════════════════════════

UMBRAL_ALIGN      = 0.85       score minimo para aprobar
K_MINIMO          = 28         sondas estandar (SONDAS_ESTANDAR)
K_RECOMENDADO     = 100        para mayor robustez con embeddings aprendidos
TRAIN_RATIO       = 0.80       fraccion para entrenamiento del aligner
POLYDIM_SEED      = "POLYDIM_V1_SEED_2026"

═══════════════════════════════════════════════════════════════
4. SONDAS ESTANDAR (SONDAS_ESTANDAR — K_MINIMO = 28)
═══════════════════════════════════════════════════════════════

DIM_PYTHON, DIM_RUST, DIM_FLUTTER, DIM_SQL,
DIM_GRAPH, DIM_VECTOR, DIM_TIME, DIM_ERROR, DIM_META,
entero, flotante, cadena, lista, diccionario,
verdadero, falso, nulo, error, exito,
crear, leer, actualizar, borrar,
usuario, sesion, permiso, dato, proceso

Estas sondas cubren el vocabulario base de POLYDIM V0.1.
IAs pueden agregar sondas extra via extra_symbols en generar_probes().

═══════════════════════════════════════════════════════════════
5. MENSAJES DEL PROTOCOLO
═══════════════════════════════════════════════════════════════

PROBE_REQUEST (IA iniciador → IA receptor):
  session_id: str
  probes: [{id: str, name_S: str, hv_A: [float × N]}]
  type: "POLYDIM_PROBE_REQUEST"

PROBE_RESPONSE (IA receptor → IA iniciador):
  session_id: str
  mappings: [{id: str, hv_B: [float × N]}]
  type: "POLYDIM_PROBE_RESPONSE"

ALIGN_CONFIRM (cada IA calcula y notifica su resultado):
  session_id: str
  score: float
  valid: bool
  degraded_to: str | null   (MODO_S si valid=false, null si valido)
  type: "POLYDIM_ALIGN_CONFIRM"

═══════════════════════════════════════════════════════════════
6. CALCULO DEL ALINEADOR — SubspaceAligner (V1, reemplaza SVD truncada)
═══════════════════════════════════════════════════════════════

PROBLEMA DE DISEÑO V0 (corregido):
  V0 usaba SVD truncada de rango 100 sobre la matriz (N × N).
  Con K=28 < 100, la SVD producia una R que destruia el complemento ortogonal.
  Norma del vector transformado: ~0.05. Score en holdout: ~0.51.
  Error: confundir rango de R con calidad de la transformacion.

SOLUCION V1 — SubspaceAligner:
  Kaabsch en el subespacio de K sondas + identidad en el complemento ortogonal.

  Algoritmo:
    Dadas hvs_a (K × N) y hvs_b (K × N):

    1. M = hvs_b @ hvs_a.T             (K × K)
    2. U, S, Vt = SVD(M)               (SVD de K×K, siempre eficiente)
    3. R_small = U @ Vt                (K × K, rotacion Kabsch)

  Transformacion de un vector x (N,):
    1. coords     = hvs_a @ x          (K,) — proyectar en subespacio A
    2. coords_rot = R_small @ coords   (K,) — rotar en K dimensiones
    3. x_in_sub   = hvs_a.T @ coords   (N,) — componente en span(A)
    4. x_comp     = x - x_in_sub       (N,) — complemento ortogonal
    5. result     = hvs_b.T @ coords_rot + x_comp   (N,)
    6. return result / ||result||

  Propiedades:
    - Identidad en el complemento ortogonal de span(A)
    - Kabsch optimo en span(A)
    - Preserva normas y angulos en el subespacio
    - SVD de K×K: O(K^3), no O(N^3). Para K=28: trivialmente rapida.

VALIDACION:
  Particionar sondas: 80% train, 20% holdout.
  Score = media de sim_coseno(transform(hv_a_i), hv_b_i) para i en holdout.
  sim_coseno(a,b) = (dot(a,b) + 1) / 2 en [-1,1] → [0,1]

  Resultado verificado (polydim_session_v01.py, sesion 003):
    Misma implementacion (V0.1 deterministico): score = 1.0000
    Umbral de aprobacion: 0.85

═══════════════════════════════════════════════════════════════
7. SECUENCIA COMPLETA
═══════════════════════════════════════════════════════════════

[IA_A ALIGNING]  [IA_B ALIGNING]

IA_A.generar_probes()
  → PROBE_REQUEST → IA_B

IA_B.responder_probes(request)
  → PROBE_RESPONSE → IA_A

IA_A.calcular_align(request, response) → ALIGN_CONFIRM_A
IA_B.calcular_align(request, response) → ALIGN_CONFIRM_B
  (ALN_007: cada IA calcula su alineador independientemente)

IA_A.finalizar_align(confirm_A, confirm_B)
IA_B.finalizar_align(confirm_A, confirm_B)

Si confirm_A.valid AND confirm_B.valid:
  → estado = READY, modo = acordado (MODO_H o MODO_G)
Sino:
  → estado = READY, modo degradado = MODO_S (ALN_004)

═══════════════════════════════════════════════════════════════
8. INTERPRETACION DEL SCORE (nuevo en V1)
═══════════════════════════════════════════════════════════════

El score ALIGN mide el grado de compatibilidad de los espacios vectoriales:

  score = 1.0000      identicos (misma implementacion V0.1 deterministico)
  score >= 0.85       compatibles para MODO_H
  score en [0.5, 0.85) parcialmente compatibles — usar MODO_S
  score ≈ 0.50        incompatibles (sin transformacion lineal subyacente)

Esta propiedad surge de la matematica del SubspaceAligner:
  - Si existe transformacion real entre espacios → score ≈ similitud de espacios
  - Si no existe transformacion real → score → 0.5, independiente de K

POLYDIM V0.1 deterministico produce score = 1.0 siempre.
Para IAs con embeddings aprendidos: score indica el grado de afinidad de entrenamientos.

═══════════════════════════════════════════════════════════════
9. INVARIANTES
═══════════════════════════════════════════════════════════════

ALN_001: K_MINIMO = 28. Menor K invalida el protocolo.
ALN_002: R_small es ortogonal. SubspaceAligner preserva angulos en span(sondas).
ALN_003: score calculado en holdout (20%), nunca en train.
ALN_004: si cualquier IA reporta valid=false → MODO_S obligatorio.
ALN_005: el alineador solo es valido durante la sesion activa.
ALN_006: re-ALIGN si score cae < UMBRAL_DERIVA durante la sesion. (pendiente)
ALN_007: cada IA calcula su alineador independientemente del par de la otra.
ALN_008 (nuevo): score no superable por aumento de K para espacios incompatibles.
ALN_009 (nuevo): UMBRAL_ALIGN = 0.85 separa correctamente implementaciones
                 deterministicas (1.0) de espacios con perturbacion media (<0.85).

═══════════════════════════════════════════════════════════════
10. COMPORTAMIENTO EN POLYDIM V0.1
═══════════════════════════════════════════════════════════════

En polydim_core_v02.py, todos los vectores se generan con:
  seed = MD5(name) mod 2^32

Toda instancia de PolyDimSpace produce vectores IDENTICOS para el mismo nombre.
Por lo tanto, para POLYDIM V0.1:
  - PROBE_REQUEST y PROBE_RESPONSE contienen los mismos vectores
  - M = B @ A^T = A @ A^T (cuasi-identidad en el subespacio)
  - R_small ≈ I (identidad)
  - score = 1.0000

ALIGN en V0.1 es un overhead minimo que siempre pasa.
Su valor esta en la preparacion para V0.x con embeddings aprendidos.

═══════════════════════════════════════════════════════════════
11. ROADMAP ALIGN
═══════════════════════════════════════════════════════════════

V0.1 (actual): deterministico, score=1.0, ALIGN de verificacion
V0.2 (future): soporte para IAs con fine-tuning diferente sobre base comun
V0.3 (future): calibracion iterativa (PEND_NUEVA_002) para embeddings aprendidos
V1.0 (future): ALIGN incremental — espacio compartido se construye durante la sesion

═══════════════════════════════════════════════════════════════

SPEC_ALIGN_V1.md — 2026-06-12 — TASK_012 TERMINADA
Reemplaza SPEC_ALIGN_V0.md (fileId: 1rvqQx0I1nugwlCTUiSV2UfCVs7I5DB2w)
