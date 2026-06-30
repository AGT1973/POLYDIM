# POLYDIM_DEST
# destino: polydim_v1/spec/
# filename: SPEC_LWW_CRDT_PROOF_V0.md
# autor:    ai.mpat.agt@gmail.com (claude-sonnet-4-6)
# fecha:    2026-06-26
# tarea:    TASK_037

---

# Last-Write-Wins es un CRDT en Sᴺ⁻¹
## Demostración Formal
### POLYDIM_v1 · V0 · 2026-06-26

**Estado epistémico:** ✅ LEY (demostración algebraica completa en la capa algebraica)

---

## Enunciado

**Teorema LWW (Last-Write-Wins CRDT en Sᴺ⁻¹).**

Sea Sᴺ⁻¹ = {v ∈ ℝᴺ : ‖v‖ = 1} la semiesfera unitaria con N ≥ 1.
Sea timestamp: Sᴺ⁻¹ → ℝ⁺ una función inyectiva (timestamps distintos).
Define:

```
merge_lww(a, b) = a   si timestamp(a) ≥ timestamp(b)
                 b   si timestamp(b) > timestamp(a)
```

Entonces (Sᴺ⁻¹, merge_lww) es un **CRDT** (Conflict-free Replicated Data Type),
es decir, satisface:

```
GC1. COMMUTATIVITY:   merge_lww(a, b) = merge_lww(b, a)
GC2. IDEMPOTENCY:     merge_lww(a, a) = a
GC3. ASSOCIATIVITY:   merge_lww(merge_lww(a,b), c) = merge_lww(a, merge_lww(b,c))
GC4. MONOTONICITY:    ∃ orden parcial ≤ en Sᴺ⁻¹ tal que merge_lww(a,b) ≥ a y ≥ b
```

---

## Demostración

### Preliminar: Notación

Para brevedad escribimos ts(x) := timestamp(x) y m(a,b) := merge_lww(a,b).
Sin pérdida de generalidad asumimos ts inyectiva (timestamps únicos).

---

### GC1 — Conmutatividad

**Claim:** m(a, b) = m(b, a).

**Caso 1:** ts(a) > ts(b).
```
m(a, b) = a       (por definición: ts(a) ≥ ts(b))
m(b, a) = a       (por definición: ts(a) > ts(b), ergo ts(a) ≥ ts(b) desde perspectiva de b)
```
Ambos = a. ∎

**Caso 2:** ts(b) > ts(a). Simétrico al caso 1. Ambos = b. ∎

**Caso 3:** ts(a) = ts(b) → a = b (por inyectividad de ts). Entonces m(a,a) = a = m(a,a). ∎

---

### GC2 — Idempotencia

**Claim:** m(a, a) = a.

Por definición: ts(a) ≥ ts(a) es verdadero, entonces m(a, a) = a. ∎

---

### GC3 — Asociatividad

**Claim:** m(m(a,b), c) = m(a, m(b,c)).

Sea t_max(S) = argmax_{x ∈ S} ts(x) para cualquier conjunto S.

**Observación clave:** merge_lww sobre cualquier conjunto S de elementos produce
el elemento con timestamp máximo:

```
m(a, b) = argmax_{x ∈ {a,b}} ts(x)     (para ts inyectiva)
```

**Lema:** Para cualquier multiconjunto {a, b, c}: m(m(a,b), c) = t_max({a,b,c}).

*Demostración del Lema:*
```
m(m(a,b), c) = m(argmax{a,b}, c)
             = argmax{argmax{a,b}, c}
             = argmax{a, b, c}          (argmax es asociativo sobre conjuntos)
```

Análogamente, m(a, m(b,c)) = argmax{a, b, c}.

Por tanto m(m(a,b), c) = argmax{a,b,c} = m(a, m(b,c)). ∎

**Nota:** La asociatividad se sostiene porque merge_lww es esencialmente la
operación `max` bajo el orden total de timestamps. `max` es asociativo
sobre cualquier conjunto ordenado totalmente (propiedad estándar de semirretículos).

---

### GC4 — Monotonicidad (orden parcial)

**Definición del orden:**
```
a ≤ b   :⟺   ts(a) ≤ ts(b)
```

Este es un **orden total** en Sᴺ⁻¹ (dado que ts es inyectiva, la comparación es total).
Un orden total es en particular un orden parcial.

**Claim:** m(a, b) ≥ a y m(a, b) ≥ b bajo ≤.

Sea x = m(a, b) = argmax_{ts}{a, b}.

- ts(x) = max(ts(a), ts(b)) ≥ ts(a), ergo x ≥ a. ✓
- ts(x) = max(ts(a), ts(b)) ≥ ts(b), ergo x ≥ b. ✓

∎

---

## Corolario para polydim_distributed.py

**Corolario 1.** `merge_versions()` en `polydim_distributed.py` es un CRDT
para todos los casos donde los timestamps son distintos:

- **causal_a_wins:** ts(a) > ts(b) → merge_lww(a, b) = a. ✓ CRDT exacto.
- **causal_b_wins:** ts(b) > ts(a) → merge_lww(a, b) = b. ✓ CRDT exacto.
- **timestamp_weighted (ts distintos):** la estrategia usa ts para ponderar,
  pero el winner efectivo es el de mayor ts. ✓ CRDT exacto.

**Corolario 2.** El único caso no-CRDT en `polydim_distributed.py` es
`merge_0.5` cuando timestamps son iguales. Este caso:
- Es degenerado (requiere timestamps exactamente iguales, poco probable en práctica)
- Tiene error de coseno < 0.025 (negligible para N=10,000, ver SPEC_GEOMETRIC_CRDT_RESEARCH_V1.md)
- Es conmutativo e idempotente (GC1, GC2 verificados empíricamente)
- Falla GC3 con error acotado (no exacto)

**Recomendación:** Para elevación de `polydim_distributed.py` a ✅ LEY
mediante Art. XX, la condición suficiente es restringir `merge` al caso
LWW (timestamps distintos garantizados por clock monotónico). El caso
merge_0.5 puede eliminarse o mantenerse como ⚙️ con la acotación documentada.

---

## Relación con CRDTs clásicos

Este resultado es una instancia directa de **LWW-Register CRDT** (Shapiro et al. 2011, §4.1).
La novedad de POLYDIM es que el "valor" del registro es un hipervector en Sᴺ⁻¹
(en lugar de un tipo escalar), pero las propiedades CRDT no dependen del tipo
del valor — solo de la estructura del merge y del orden de timestamps.

**Referencia:** Shapiro, M., Preguiça, N., Baquero, C., Zawirski, M. (2011).
"Conflict-free replicated data types." DISC 2011, pp. 386–400.

---

## Estado tras esta prueba

| Propiedad | merge_lww | Estatus |
|---|---|---|
| GC1 Conmutatividad | ✅ probado | |
| GC2 Idempotencia | ✅ probado | |
| GC3 Asociatividad | ✅ probado | argmax es asociativo |
| GC4 Monotonicidad | ✅ probado | orden total por timestamps |

`polydim_distributed.py` implementa un CRDT formalmente correcto para
todos los casos con timestamps distintos (que es el caso operativo normal).

---

*SPEC_LWW_CRDT_PROOF_V0.md · TASK_037 ✓ · 2026-06-26 · ai.mpat.agt@gmail.com*
*Eleva merge_lww de ⚙️ MECANISMO a ✅ LEY (capa algebraica)*
*Proceso Art. XX pendiente para actualización formal de la Constitución V6*
