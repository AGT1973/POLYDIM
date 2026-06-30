# POLYDIM_DEST
# destino: polydim/spec/
# nombre:  SPEC_ALIGN_V0.md
# autor:   ai.mpat.agt@gmail.com

# SPEC — Protocolo ALIGN
# Version: V0.1 — 2026-06-11
# TASK_004 — Resuelve PEND_006

---

## PROPOSITO

ALIGN es el protocolo que permite a dos IAs con espacios de embedding distintos
operar en MODO_G o MODO_H sin perdida semantica.

Problema: IA-A y IA-B pueden tener la misma POLYDIM_SEED y el mismo N,
pero si sus arquitecturas internas son distintas (transformer vs VSA puro,
distintos pesos de entrenamiento), el vector que representa "usuario" en IA-A
no es el mismo vector en IA-B aunque usen el mismo nombre.

ALIGN calcula una matriz de rotacion R tal que:
  R * HV_A(concepto) ≈ HV_B(concepto) para todo concepto conocido

Despues de ALIGN, IA-A puede transformar cualquier hipervector con R
antes de enviarlo, y IA-B lo recibe en su propio espacio.

---

## BASE MATEMATICA

### Fundamento: alineacion lineal de espacios de embedding

Mikolov et al. (2013) demostraron que los espacios de embedding de dos idiomas
distintos son aproximadamente isomorfos — existe una transformacion lineal
que mapea uno en el otro con alta precision.

Esto se aplica directamente a POLYDIM:
dos instancias de POLYDIM_SPACE con la misma seed y N son isomorfas.
La matriz de rotacion R existe y es recuperable con suficientes pares de sondas.

### Calculo de R

Dado un conjunto de K pares (HV_A_i, HV_B_i) donde ambos representan el mismo concepto:

  Minimizar: sum_i ||R * HV_A_i - HV_B_i||^2
  Con restriccion: R es ortogonal (R^T * R = I)

Solucion exacta via SVD:
  M = HV_B^T * HV_A          (matriz de correlacion K x N)
  U, S, V^T = SVD(M)
  R = U * V^T                 (rotacion optima)

Propiedades de R:
  - Ortogonal: preserva normas y angulos
  - Optima en el sentido de minimos cuadrados
  - Unica si los pares son suficientemente diversos (K >= N teoricamente,
    K >= 100 practicamente con N=10000 por la alta dimensionalidad)

---

## MENSAJES DEL PROTOCOLO

### PROBE_REQUEST
IA-A solicita a IA-B que mapee un conjunto de sondas.

```
PROBE_REQUEST {
  type:       "POLYDIM_PROBE_REQUEST"
  session_id: string
  probes: [
    {
      id:     string        — identificador de la sonda
      name_S: string        — nombre en Capa S (ej: "DIM_SQL", "usuario", "entero")
      hv_A:   float[]       — hipervector de la sonda en espacio de IA-A
    },
    ...  (K sondas total)
  ]
}
```

### PROBE_RESPONSE
IA-B responde con sus propios hipervectores para los mismos conceptos.

```
PROBE_RESPONSE {
  type:       "POLYDIM_PROBE_RESPONSE"
  session_id: string
  mappings: [
    {
      id:     string    — mismo id que en PROBE_REQUEST
      hv_B:   float[]   — hipervector del mismo concepto en espacio de IA-B
    },
    ...
  ]
}
```

### ALIGN_CONFIRM
Ambas IAs confirman que la alineacion es valida.

```
ALIGN_CONFIRM {
  type:          "POLYDIM_ALIGN_CONFIRM"
  session_id:    string
  score:         float    — similitud promedio en holdout [0.0, 1.0]
  valid:         bool     — score >= UMBRAL_ALIGN (0.85)
  degraded_to:   string   — null si valid, "MODO_S" si no valido
}
```

---

## ALGORITMO COMPLETO

### Paso 1 — Seleccion de sondas (IA iniciadora)

La IA iniciadora selecciona K sondas del conjunto estandar POLYDIM.
Las sondas estandar son los 9 subespacios nativos + conceptos basicos:

```
SONDAS_ESTANDAR = [
  "DIM_PYTHON", "DIM_RUST", "DIM_FLUTTER", "DIM_SQL",
  "DIM_GRAPH",  "DIM_VECTOR", "DIM_TIME",  "DIM_ERROR", "DIM_META",
  "entero", "flotante", "cadena", "lista", "diccionario",
  "verdadero", "falso", "nulo", "error", "exito",
  "crear", "leer", "actualizar", "borrar",
  "usuario", "sesion", "permiso", "dato", "proceso"
]
```

K_MINIMO = 28 (todas las sondas estandar)
K_RECOMENDADO = 100 (agregar simbolos del dominio de la sesion)

Partition: 80% entrenamiento, 20% holdout (para validacion)

### Paso 2 — Intercambio de sondas

IA-A envia PROBE_REQUEST con sus HV_A para cada sonda.
IA-B responde con PROBE_RESPONSE con sus HV_B para los mismos conceptos.

### Paso 3 — Calculo de R (cada IA calcula la suya)

Cada IA calcula su propia matriz de rotacion:

```python
import numpy as np

def calcular_R(hv_origen, hv_destino):
    """
    hv_origen: matriz (K_train x N) — hipervectores del espacio origen
    hv_destino: matriz (K_train x N) — hipervectores del espacio destino
    Retorna R ortogonal (N x N) tal que R @ hv_origen[i] ≈ hv_destino[i]
    """
    M = hv_destino.T @ hv_origen      # (N x N)
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt                         # rotacion optima
    return R
```

IA-A calcula R_AB: transforma su espacio al de IA-B
IA-B calcula R_BA: transforma su espacio al de IA-A

En MODO_H: IA-A aplica R_AB antes de enviar Capa G. IA-B no necesita transformar.

### Paso 4 — Validacion con holdout

Cada IA verifica R con las sondas de holdout:

```python
def validar_R(R, hv_origen_holdout, hv_destino_holdout):
    scores = []
    for hv_a, hv_b in zip(hv_origen_holdout, hv_destino_holdout):
        hv_a_rot = R @ hv_a
        hv_a_rot = hv_a_rot / np.linalg.norm(hv_a_rot)
        sim = (float(np.dot(hv_a_rot, hv_b)) + 1.0) / 2.0
        scores.append(sim)
    return float(np.mean(scores))

score = validar_R(R_AB, holdout_A, holdout_B)
valido = score >= 0.85  # UMBRAL_ALIGN
```

### Paso 5 — Confirmacion

Ambas IAs envian ALIGN_CONFIRM con su score.
Si ambas reportan valid=true: sesion pasa a READY en MODO_G o MODO_H.
Si alguna reporta valid=false: degradar a MODO_S.

---

## RE-ALIGN PARCIAL

Durante una sesion activa, la alineacion puede degradarse si:
- Las IAs actualizan sus modelos internos (deriva semantica)
- Se agregan nuevos subespacios no cubiertos por las sondas originales

Deteccion de deriva:
```
Cada N_CHECK mensajes (default: 50), verificar similitud promedio
de los ultimos mensajes intercambiados contra las sondas de holdout.
Si similitud_promedio < UMBRAL_DERIVA (0.75): iniciar re-ALIGN parcial.
```

Re-ALIGN parcial: igual que ALIGN completo pero con K = K_MINIMO (sondas estandar).
No interrumpe la sesion — se ejecuta en paralelo.
Hasta que complete: mensajes en MODO_H usan Capa S para datos criticos.

---

## USO DE R DURANTE LA SESION

En MODO_G:
```
Emisor (IA-A) antes de enviar HV:
  hv_enviar = R_AB @ hv_original
  hv_enviar = hv_enviar / ||hv_enviar||
  # IA-B recibe en su propio espacio directamente
```

En MODO_H (plano de datos):
```
Idem — los hipervectores del plano de datos se transforman con R_AB
Los mensajes del plano de control (Capa S) no se transforman
```

En MODO_S:
```
No se usa R. El intercambio es simbolico. No hay hipervectores.
```

---

## CASO ESPECIAL: N DISTINTO

Si IA-A tiene N_A y IA-B tiene N_B donde N_A ≠ N_B:
R no puede ser calculada directamente (dimensiones incompatibles).

Opciones (PEND_010 — no resuelto en esta version):
1. Padding: completar el espacio menor con ceros hasta igualar N
2. Proyeccion: reducir ambos al min(N_A, N_B) antes de calcular R
3. Fallback obligatorio a MODO_S (el mas seguro)

En V0.1: si N distinto → MODO_S obligatorio (ver SPEC_HANDSHAKE_V0.md)

---

## COSTO COMPUTACIONAL

```
Paso 1 (sondas):   O(K) — seleccion trivial
Paso 2 (intercambio): O(K * N) — transmision de K hipervectores de dimension N
Paso 3 (SVD):       O(N^3) — el paso mas costoso
                    Para N=10000: ~10^12 operaciones en SVD exacto
                    ALTERNATIVA: SVD truncada con rango r << N
                                 O(r * N^2) donde r ~ 100 → ~10^8 operaciones
                                 aceptable para ALIGN inicial

Paso 4 (validacion): O(K_holdout * N) — trivial

TOTAL estimado con SVD truncada (r=100, K=100, N=10000):
  ~10^8 operaciones → segundos en hardware moderno
```

SVD truncada recomendada para implementacion practica:
```python
from sklearn.utils.extmath import randomized_svd
U, S, Vt = randomized_svd(M, n_components=100)
R = U @ Vt  # aproximacion de rango 100 — suficiente para UMBRAL_ALIGN=0.85
```

---

## INVARIANTES DE ALIGN

```
ALN_001: ALIGN requiere K_MINIMO = 28 sondas estandar como minimo
ALN_002: R es ortogonal — preserva similitudes coseno
ALN_003: Score de validacion en holdout debe ser >= 0.85 (UMBRAL_ALIGN)
ALN_004: Si score < umbral: degradar a MODO_S, no operar en MODO_G con R invalida
ALN_005: R es valida solo para la sesion activa — no se reutiliza en sesiones nuevas
ALN_006: Re-ALIGN se ejecuta en paralelo sin interrumpir la sesion
ALN_007: IA-A y IA-B calculan R independientemente — sin transferir la matriz
         (cada una tiene su propia R para transformar su espacio al del otro)
```

---
*SPEC_ALIGN_V0.md — V0.1 — 2026-06-11 — TASK_004 TERMINADA — PEND_006 RESUELTO*
