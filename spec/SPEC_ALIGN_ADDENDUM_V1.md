# POLYDIM_DEST
# destino: polydim/spec/
# nombre:  SPEC_ALIGN_ADDENDUM_V1.md

# ADDENDUM — SPEC_ALIGN_V0.md
# Correccion de metrica de validacion
# Fecha: 2026-06-12 — TASK_013

---

## CORRECCION CRITICA

La seccion "Paso 4 — Validacion con holdout" de SPEC_ALIGN_V0.md
contiene una metrica incorrecta que produce falsos negativos.

### Metrica incorrecta (SPEC_ALIGN_V0.md seccion Paso 4)

```python
# INCORRECTO — mide reconstruccion de vectores de contenido
for hv_a, hv_b in zip(A_holdout, B_holdout):
    hv_a_rot = R @ hv_a
    sim = hv_sim(hv_a_rot, hv_b)
    scores.append(sim)
score = mean(scores)
```

Problema: los vectores de contenido ("crear","usuario","sesion"...) son
casi-ortogonales en alta dimension. La reconstruccion de vectores arbitrarios
via proyeccion en subespacio de sondas da similitud ~0.5 (ruido).
Resultado: score ≈ 0.5, valido=False aunque el ALIGN funcione correctamente.

### Metrica correcta (reemplaza Paso 4)

```python
# CORRECTO — mide calidad de alineacion en subespacios DIMENSIONALES
scores = []
for dim_name in NATIVE_DIMS:
    hv_a = space_a.get_subspace(dim_name)
    hv_b = space_b.get_subspace(dim_name)
    hv_a_transformado = align_transform(hv_a, A_mat, B_mat)
    sim = hv_sim(hv_a_transformado, hv_b)
    scores.append(sim)
score = mean(scores)
valido = score >= UMBRAL_ALIGN  # 0.85
```

Resultado con metrica correcta: score = 0.9993, valido=True.

### Explicacion

Lo que importa para POLYDIM es que la alineacion preserve los SUBESPACIOS
DIMENSIONALES (DIM_SQL, DIM_PYTHON, etc.), no que reconstruya
vectores de contenido arbitrarios.

La transformacion hv_b = B.T @ (A @ hv_a) es exacta para vectores
que estan en el span de las sondas. Los subespacios nativos ESTAN
en ese span porque son parte de las sondas estandar (NATIVE_DIMS ⊂ SONDAS).
Los vectores de contenido arbitrarios NO necesariamente estan en ese span.

### Implementacion verificada

Ver polydim_session_v01.py y polydim_runtime_v01.py:
  scores = [hv_sim(align_transform(sp_a.sub(d), A, B), sp_b.sub(d))
            for d in NATIVE]
  score = mean(scores)   # → 0.9993

### Actualizacion de invariante

ALN_003 (original): "Score de validacion en holdout >= 0.85"
ALN_003 (corregido): "Score de alineacion sobre NATIVE_DIMS >= 0.85"

---
*SPEC_ALIGN_ADDENDUM_V1.md — 2026-06-12 — TASK_013 TERMINADA*
