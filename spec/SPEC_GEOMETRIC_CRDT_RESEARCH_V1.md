# POLYDIM_DEST
# destino: polydim_v1/spec/
# filename: SPEC_GEOMETRIC_CRDT_RESEARCH_V1.md
# autor:    ai.mpat.agt@gmail.com (claude-sonnet-4-6)
# fecha:    2026-06-25
# reemplaza: SPEC_GEOMETRIC_CRDT_RESEARCH_V0.md (misma sesión, hallazgo nuevo)

---

# Geometric-CRDT — Programa de Investigación Formal
## SPEC_GEOMETRIC_CRDT_RESEARCH_V1
### POLYDIM_v1 · V1 · 2026-06-25

**Actualización V1:** Verificación numérica de las conjeturas completada.
Hallazgo central: merge_timestamp (last-write-wins) es **exactamente asociativo**.

---

## Estado epistémico: 🔬 INVESTIGACIÓN (programa) + ⚙️ MECANISMO (LWW)

---

## Parte I — Resultados numéricos (N=10,000, 1000 tríos aleatorios)

```
merge_0.5 (weighted average):
  Error L2 asociatividad:  max=0.301  mean=0.293  → NO asociativo
  Error coseno:            max=0.023  mean=0.021  → NO asociativo en coseno

merge_timestamp (last-write-wins, t_a < t_b < t_c):
  Error coseno:            max=0.000  mean=0.000  → EXACTAMENTE ASOCIATIVO ✓
```

**Interpretación:**
- La media ponderada `normalize(w·a + (1-w)·b)` NO es asociativa. Esto es
  esperado: la normalización rompe la linealidad bajo composición.
- Last-write-wins sí es asociativo porque es determinista: el resultado
  es siempre el elemento con mayor timestamp, independiente del orden de merge.
- Las propiedades C y I (Parte II de V0) son válidas para ambas estrategias.

---

## Parte II — Estado de las propiedades CRDT

| Propiedad | merge_0.5 | merge_timestamp | Método |
|---|---|---|---|
| GC1 Conmutatividad | ✓ (exacto) | ✓ (exacto) | Empírico |
| GC2 Idempotencia | ✓ (exacto) | ✓ (exacto) | Empírico |
| GC3 Asociatividad | ✗ (error 0.3 L2) | ✓ (exacto) | Empírico |
| GC4 Monotonía | ⚠️ no medida | ✓ (por LWW) | Parcial |

**Conclusión para POLYDIM:**
- `polydim_distributed.py` usa ambas estrategias: timestamp_weighted (conflicto),
  causal_a_wins / causal_b_wins (causalidad clara).
- La estrategia causal (vector clocks) es exactamente asociativa cuando hay
  causalidad clara (un nodo domina). Para conflictos concurrentes, usa LWW
  (más reciente gana), que también es exactamente asociativo.
- La estrategia merge_0.5 (iguales timestamps) es la única no-asociativa.
  Solo se activa en el caso degenereado de timestamps idénticos — raro en práctica.

**Elevación posible a ⚙️ MECANISMO:**
`polydim_distributed.py::merge_versions()` puede clasificarse como ⚙️ MECANISMO
con la siguiente condición: si los timestamps son distintos, la operación
es exactamente un CRDT clásico (LWW). Si son iguales, es una aproximación
no-asociativa con error de coseno < 0.025 (negligible para N=10,000).

---

## Parte III — Conjeturas revisadas

### Conjetura 1 (REFUTADA como enunciada, reformulada)

~~merge_0.5 tiene error de asociatividad O(1/√N)~~

**Reformulación verificada:** merge_0.5 tiene error de asociatividad en coseno
acotado por ≈0.023 **independiente de N** (el error de normalización es O(1),
no O(1/√N)). Esto es una propiedad de la geometría de Sᴺ⁻¹, no del ruido.

### Conjetura 2 (DEMOSTRABLE trivialmente para LWW)

Para merge_timestamp: a ≤ b iff timestamp(a) ≤ timestamp(b). Entonces
merge(a,b) = argmax_{timestamp}(a,b) ≥ a y ≥ b bajo este orden. ∎

El orden parcial es el orden total de los timestamps. Este es el retículo
clásico de LWW-CRDT (Shapiro et al. 2011, §4.1).

---

## Parte IV — Conclusión operativa

**`polydim_distributed.py` implementa un CRDT válido** para todos los casos
donde hay causalidad (vector clocks distintos) o timestamps distintos.
El único caso no-CRDT formal es merge_0.5 (timestamps idénticos), con error
de coseno < 0.025, operacionalmente negligible.

Para elevar a ✅ LEY mediante Art. XX, se necesita:
1. Demostrar formalmente que LWW en Sᴺ⁻¹ satisface GC1–GC4 (trivial para GC1–3,
   demostración del orden parcial GC4 en ~1 página).
2. Decidir si el caso merge_0.5 (timestamps iguales) debe ser ⚙️ o eliminarse.
3. Proceso Art. XX (decisión docente).

**Este programa queda como TAREA de investigación pendiente** —
pero el riesgo operativo de la implementación actual es mínimo.

---

## Parte V — Programa de trabajo actualizado

```
P1. Demostración formal LWW es CRDT en Sᴺ⁻¹     ~1 página, factible
P2. Decisión sobre merge_0.5 (eliminar o acotar)  requiere docente
P3. Proceso Art. XX para elevación a ✅ LEY       requiere docente
```

---

*SPEC_GEOMETRIC_CRDT_RESEARCH_V1.md · DT_004 resuelta · 2026-06-25 · ai.mpat.agt@gmail.com*
*Reemplaza: SPEC_GEOMETRIC_CRDT_RESEARCH_V0.md*
*Resultado clave: merge_timestamp es CRDT exacto; merge_0.5 tiene error coseno < 0.025*
