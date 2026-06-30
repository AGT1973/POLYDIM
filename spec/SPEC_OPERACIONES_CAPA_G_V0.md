# POLYDIM_DEST
# destino: polydim/spec/
# nombre:  SPEC_OPERACIONES_CAPA_G_V0.md
# autor:   ai.mpat.agt@gmail.com

# SPEC — Operaciones Formales Capa G
# Version: V0.1 — 2026-06-10

---

## BASE MATEMATICA: VSA con vectores reales (MAP model)

POLYDIM usa el modelo MAP (Multiply-Add-Permute) de VSA con vectores reales en R^N.
Elegido por: maxima composicionalidad, diferenciabilidad (compatible con gradientes),
compatibilidad con espacios de embedding de LLMs existentes.

Todos los hipervectores son unitarios: ||hv|| = 1

---

## OPERACIONES PRIMITIVAS

### HV_BIND(hv_a, hv_b) → hv_bound
```
Definicion:   hv_bound = normalize(hv_a ⊙ hv_b)
              donde ⊙ es producto elemento a elemento (Hadamard)

Propiedad:    NO conmutativo con signo, pero BIND(a,b) ≈ -BIND(b,a) en algunos modelos
              En MAP: aproximadamente conmutativo — usar PERMUTE si se necesita orden

Para POLYDIM: BIND asocia un nombre de dimension con su contenido
              BIND(DIM_SQL, contenido_sql) → objeto en dimension SQL

Inversa:      HV_UNBIND(hv_bound, hv_a) ≈ hv_b
              ya que (a ⊙ b) ⊙ a = b ⊙ (a ⊙ a) = b ⊙ 1 ≈ b
              (aproximado, no exacto — propiedad VSA)
```

### HV_SUPERPOSE(hv_1, ..., hv_k) → hv_combined
```
Definicion:   hv_combined = normalize(hv_1 + hv_2 + ... + hv_k)
              suma vectorial seguida de normalizacion

Propiedad:    CONMUTATIVO y ASOCIATIVO
              SUPERPOSE(a, b) = SUPERPOSE(b, a)
              SUPERPOSE(a, b, c) = SUPERPOSE(SUPERPOSE(a,b), c)

Para POLYDIM: combina multiples dimensiones en un solo OBJECT_ND
              el resultado "contiene" todas las dimensiones

Recuperacion: dado hv_combined y hv_dim_i conocido,
              SIM(hv_combined, BIND(hv_dim_i, x)) > umbral → x esta presente

Nota critica: con k dimensiones, la señal de cada una se degrada ~1/sqrt(k)
              con N=10000 se pueden superponer ~100 dimensiones con recuperacion confiable
```

### HV_PROJECT(hv, subspace_hv) → peso [0.0, 1.0]
```
Definicion:   similitud_coseno = dot(hv, subspace_hv)
              peso = (similitud_coseno + 1.0) / 2.0   [mapeo a [0,1]]

Para POLYDIM: mide cuanto esta "activa" una dimension en un objeto
              peso > 0.7 → dimension claramente presente
              peso ~ 0.5 → dimension latente o ruido
              peso < 0.3 → dimension ausente

No destruye: PROJECT no modifica hv — es una consulta, no una mutacion
```

### HV_UNBIND(hv_bound, hv_a) → hv_b (aproximado)
```
Definicion:   hv_b_aprox = normalize(hv_bound ⊙ hv_a)
              aprovecha que Hadamard es su propia inversa

Propiedad:    SIM(hv_b_aprox, hv_b_real) > umbral en alta dimension
              la aproximacion mejora con N mayor

Para POLYDIM: recupera el contenido de una dimension desde un objeto superposicionado
              UNBIND(objeto_nd, DIM_SQL) ≈ contenido_sql
```

### HV_SIM(hv_a, hv_b) → [0.0, 1.0]
```
Definicion:   sim = (dot(hv_a, hv_b) + 1.0) / 2.0

Interpetacion:
  sim > 0.9 → objetos semanticamente equivalentes (misma dimension, mismo contenido)
  sim > 0.7 → objetos relacionados
  sim ~ 0.5 → sin relacion (ortogonales — esperado entre dims distintas)
  sim < 0.3 → objetos opuestos o en tension semantica
```

### HV_ENCODE(objeto_S) → hv
```
Definicion:   traduccion de Capa S a Capa G

Algoritmo:
  1. Para cada dimension D_i con peso w_i:
     a. hv_nombre = SPACE.get_symbol(D_i.nombre)
     b. hv_props  = ENCODE_PROPS(D_i.props)
     c. hv_dim_i  = BIND(hv_nombre, hv_props)
  2. hv_objeto = geo_id + SUPERPOSE(hv_dim_i * w_i para todo i)
  3. normalizar

Donde ENCODE_PROPS(props_dict):
  Para cada (clave, valor) en props:
    hv_par = BIND(get_symbol(clave), get_symbol(str(valor)))
  Retorna SUPERPOSE(todos los pares)

Nota: los simbolos son deterministicos desde el nombre (seed de hash)
      mismo nombre → mismo hipervector en cualquier instancia de POLYDIM_SPACE
```

### HV_DECODE(hv) → objeto_S (aproximado)
```
Definicion:   traduccion de Capa G a Capa S

Algoritmo:
  1. Para cada subespacio conocido D_i en POLYDIM_SPACE:
     peso_i = PROJECT(hv, get_subspace(D_i))
  2. Si peso_i > umbral (default 0.5): D_i esta presente en el objeto
  3. Para recuperar props de D_i:
     contenido_aprox = UNBIND(hv, get_symbol(D_i.nombre))
     buscar en tabla de simbolos conocidos por similitud

Limitacion: solo recupera dimensiones y simbolos conocidos por POLYDIM_SPACE
            dimensiones no registradas en el espacio no son recuperables
            esto es perdida aceptable (INV_010)
```

---

## OPERACIONES COMPUESTAS

### ND_SIMILAR(objeto_a, objeto_b) → bool
```
HV_SIM(objeto_a.geo_id, objeto_b.geo_id) > UMBRAL_IDENTIDAD
Retorna true si son el mismo objeto (misma identidad, distintas dimensiones posibles)
```

### ND_MERGE(objeto_a, objeto_b) → objeto_nuevo
```
Crea un nuevo objeto que contiene las dimensiones de ambos
geo_id nuevo = BIND(objeto_a.geo_id, objeto_b.geo_id)
hv_nuevo = SUPERPOSE(objeto_a.to_hv(), objeto_b.to_hv())
No modifica ninguno de los originales (inmutabilidad dimensional INV_006)
```

### ND_DISTANCE(objeto_a, objeto_b) → float
```
1.0 - HV_SIM(objeto_a.to_hv(), objeto_b.to_hv())
Distancia semantica: 0.0 = identicos, 1.0 = opuestos/sin relacion
```

---

## PARAMETROS DEL RUNTIME

```
POLYDIM_N:            10000   (dimension del espacio — configurable)
UMBRAL_RECUPERACION:  0.5     (SIM minimo para considerar dimension presente)
UMBRAL_IDENTIDAD:     0.85    (SIM de geo_id para considerar mismo objeto)
UMBRAL_ALIGN:         0.85    (SIM minimo en ALIGN para validar alineacion)
K_SONDAS_ALIGN:       100     (sondas minimas para protocolo ALIGN)
MAX_DIMS_CONFIABLES:  100     (dimensiones superposicionadas con recuperacion confiable)
                              calculado como: sqrt(N) = sqrt(10000) = 100
```

---
*SPEC_OPERACIONES_CAPA_G_V0.md — V0.1 — 2026-06-10 — TASK_005 TERMINADA*
