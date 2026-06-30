# POLYDIM_DEST
# destino: polydim/spec/
# nombre:  SPEC_POLYDIM_SPACE_V0.md
# autor:   ai.mpat.agt@gmail.com

# SPEC — POLYDIM_SPACE: el espacio compartido
# Version: V0.1 — 2026-06-10

---

## DEFINICION

POLYDIM_SPACE es el espacio vectorial compartido en el que existen todos los OBJECT_ND.

Es un espacio de alta dimension R^N con una tabla de simbolos deterministicos.
Deterministico significa: dado un nombre de simbolo, su hipervector es siempre
el mismo en cualquier instancia de POLYDIM_SPACE que use el mismo N y la misma seed.

Esto es lo que permite que dos IAs sin ALIGN previo puedan operar en MODO_S:
los simbolos tienen el mismo nombre y se codifican igual en ambas instancias.

---

## PARAMETROS

```
N (dimension):     10000  (default — configurable por el runtime)
SEED_GLOBAL:       "POLYDIM_V1_SEED_2026"  (fija para version del lenguaje)
ENCODING_METHOD:   MAP (Multiply-Add-Permute con vectores reales)
NORM:              L2 unitario (todos los hipervectores tienen norma 1)
```

Por que N = 10000:
- Capacidad confiable: sqrt(10000) = 100 dimensiones superposicionadas
- Similitud entre vectores aleatorios: E[sim] ≈ 0, desviacion ≈ 0.01
  (probabilidad de colision accidental < 1 en 10^8)
- Costo de memoria: 10000 * 4 bytes = 40KB por objeto (aceptable)
- Compatible con espacios de embedding de LLMs actuales (tipicamente 4096-16384)

---

## TABLA DE SIMBOLOS

Los simbolos son deterministicos: generados con hash MD5 del nombre como seed.

```python
seed = int(MD5(nombre.encode()).hexdigest(), 16) % 2^32
rng  = numpy.random.default_rng(seed)
hv   = rng.standard_normal(N)
hv   = hv / ||hv||
```

Cualquier instancia de POLYDIM_SPACE con N=10000 y el mismo nombre
produce el mismo hipervector. Sin coordinacion previa necesaria.

---

## SUBESPACIOS NATIVOS

Los subespacios nativos son dimensiones pre-registradas en toda instancia de
POLYDIM_SPACE. Son parte del estandar del lenguaje.

```
Nombre          Descripcion
DIM_PYTHON      Logica dinamica, tipado duck, interpretado, analisis, ML
DIM_RUST        Seguridad de memoria, ownership, zero-cost, performance
DIM_FLUTTER     Estado reactivo, UI, widgets, streams, presentacion
DIM_SQL         Datos relacionales, constraints, joins, persistencia
DIM_GRAPH       Grafos, nodos, aristas, traversal, relaciones
DIM_VECTOR      Embeddings, similitud semantica, operaciones VSA
DIM_TIME        Secuencias temporales, eventos, orden, causalidad
DIM_ERROR       Estados de error, excepciones, recuperacion
DIM_META        Metadatos del objeto, version, auditoria, origen
```

Los subespacios nativos tienen hipervectores fijos derivados de sus nombres
con la misma funcion deterministica.

Las IAs pueden registrar subespacios adicionales. Los subespacios no-nativos
requieren ALIGN para ser compartidos entre IAs distintas (ver SPEC_ALIGN).

---

## ESPACIO LIBRE

El espacio R^10000 tiene capacidad para aproximadamente:
  - 100 dimensiones superposicionadas confiablemente por objeto
  - La tabla de simbolos puede crecer sin limite (los simbolos son casi-ortogonales)
  - Los 9 subespacios nativos ocupan una fraccion insignificante del espacio total

El espacio libre esta disponible para:
  - Subespacios de dominio especifico (DIM_MEDICINA, DIM_JURIDICO, etc.)
  - Subespacios definidos por IAs en runtime
  - Extensiones del lenguaje sin modificar el estandar base

---

## COMPATIBILIDAD ENTRE INSTANCIAS

Dos instancias de POLYDIM_SPACE son compatibles si:
  1. Mismo N
  2. Misma SEED_GLOBAL
  3. Mismo ENCODING_METHOD

Con estas tres condiciones, los simbolos son identicos — no se necesita ALIGN
para operar en MODO_S o MODO_H con subespacios nativos.

Para subespacios no-nativos o MODO_G puro: se necesita ALIGN.

---
*SPEC_POLYDIM_SPACE_V0.md — V0.1 — 2026-06-10 — TASK_008 TERMINADA*
