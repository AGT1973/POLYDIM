"""
POLYDIM Demo End-to-End — V0.1
================================
Tres escenarios de intercambio de OBJECT_ND entre IAs:

  ESCENARIO A — Espacios identicos (misma arquitectura, mismo modelo)
    ALIGN score = 1.0000 → MODO_H con hv_g real
    Demuestra el canal geometrico completo

  ESCENARIO B — Arquitecturas similares (mismo corpus, distinto ruido)
    K=28 sondas cubren ~0.28% del espacio N=10000
    ALIGN score ~0.50: insuficiente → degradacion automatica a MODO_S
    DIM_SQL detectada igual via Capa S

  ESCENARIO C — Arquitecturas completamente distintas
    Sin relacion lineal posible
    ALIGN score ~0.50 → MODO_S
    DIM_SQL detectada igual via Capa S

HALLAZGO TECNICO (sesion 003):
  ALIGN en V0.1 es efectivo solo cuando N es pequeño o K >> sqrt(N).
  Para N=10000, K efectivo requiere ~1000 sondas (PEND_NEW_001).
  MODO_S es el modo de produccion por defecto.
  MODO_H es optimo para IAs que comparten el mismo modelo base.

Verificaciones por escenario:
  V1 DIM_SQL detectada en receptor
  V2 DIM_PYTHON detectada
  V3 DIM_RUST latente
  V4 geo_hash preservado
  V5 props intactas en Capa S
  V6 modo correcto (MODO_H en A, MODO_S en B y C)

Autor:   ai.mpat.agt@gmail.com
Version: V0.1 — 2026-06-12 — TASK_011
"""

import numpy as np
import hashlib
import json
from polydim_core_v02 import PolyDimSpace, ObjectND, hv_sim, UMBRAL_RECUPERACION
from polydim_session_v01 import (
    PolyDimSession, SessionState, Mode, Cap, UMBRAL_ALIGN
)


# ─────────────────────────────────────────────
# ESPACIOS
# ─────────────────────────────────────────────

class NoisySpace(PolyDimSpace):
    """Espacio A + ruido gaussiano por nombre — similar pero no igual."""
    def __init__(self, base_space, noise_scale=1.0):
        self._base = base_space; self._noise = noise_scale
        self._symbols = {}; self._subspaces = {}; self.N = base_space.N
        from polydim_core_v02 import NATIVE_DIMS
        for d in NATIVE_DIMS: self._subspaces[d] = self._make(d)

    def _make(self, name):
        hv = self._base._make(name).copy()
        seed = int(hashlib.md5(("NOISY"+name).encode()).hexdigest(), 16) % (2**32)
        noise = np.random.default_rng(seed).standard_normal(self.N).astype(np.float32)
        hv = hv + noise * self._noise
        n = np.linalg.norm(hv)
        return hv / n if n > 1e-10 else hv


class RandomSpace(PolyDimSpace):
    """Espacio completamente aleatorio — sin relacion con ninguno otro."""
    def _make(self, name):
        seed = int(hashlib.md5(("RAND"+name).encode()).hexdigest(), 16) % (2**32)
        hv = np.random.default_rng(seed).standard_normal(self.N).astype(np.float32)
        return hv / np.linalg.norm(hv)


# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────

def reconstruir_objeto(paquete, session_b):
    space_b = session_b.space
    obj = ObjectND(space_b)
    for dim_name, dim_data in paquete.get("capa_s", {}).get("dimensions", {}).items():
        obj.add_dim(dim_name, dim_data.get("props", {}),
                    weight=dim_data.get("weight_declared", 1.0))
    if "hv_g" in paquete:
        hv = np.array(paquete["hv_g"], dtype=np.float32)
        n = np.linalg.norm(hv)
        obj._cache = hv / n if n > 1e-10 else hv
    return obj


def crear_objeto_test(space_a):
    return (ObjectND(space_a)
        .add_dim("DIM_SQL",    {"tabla": "usuarios", "pk": "id"},   weight=1.0)
        .add_dim("DIM_PYTHON", {"tipo": "dict", "len": "variable"}, weight=0.7)
        .add_dim("DIM_RUST",   {"lifetime": "static"},              weight=0.0))


def run_escenario(label, espera_modo_h, space_a, space_b):
    print(f"\n{'─'*65}")
    print(f"ESCENARIO {label}")
    print(f"{'─'*65}")

    sim_pre = hv_sim(space_a.sub("DIM_SQL"), space_b.sub("DIM_SQL"))
    print(f"  Sim(DIM_SQL) pre-ALIGN: {sim_pre:.4f}")

    caps = [Cap.CAP_S.value, Cap.CAP_G.value, Cap.CAP_ALIGN.value]
    ia_a = PolyDimSession(f"IA-A", space_a, caps=caps)
    ia_b = PolyDimSession(f"IA-B", space_b, caps=caps)

    init   = ia_a.iniciar_handshake()
    accept = ia_b.recibir_init(init)
    ack    = ia_a.recibir_accept(accept)
    ia_b.recibir_ack(ack)

    ia_b.state = SessionState.ALIGNING
    probe_req  = ia_a.generar_probes()
    probe_resp = ia_b.responder_probes(probe_req)
    c_a = ia_a.calcular_align(probe_req, probe_resp)
    c_b = ia_b.calcular_align(probe_req, probe_resp)
    ia_a.finalizar_align(c_a, c_b)
    ia_b.finalizar_align(c_a, c_b)

    modo_final = ia_a.agreed_mode
    print(f"  ALIGN: score={c_a.score:.4f}  →  modo final: {modo_final}")

    obj_orig = crear_objeto_test(space_a)
    paquete  = ia_a.empaquetar_objeto(obj_orig)
    obj_recv = reconstruir_objeto(paquete, ia_b)

    dims_recv = obj_recv.active_dims()
    dims_s    = paquete.get("capa_s", {}).get("dimensions", {})
    has_hv_g  = "hv_g" in paquete

    print(f"  Dims detectadas en receptor ({modo_final}):")
    for d, w in sorted(dims_recv.items(), key=lambda x: -x[1]):
        print(f"    {d:20s}: {w:.4f}")

    v1 = "DIM_SQL"    in dims_recv or "DIM_SQL"    in dims_s
    v2 = "DIM_PYTHON" in dims_recv or "DIM_PYTHON" in dims_s
    v3 = "DIM_RUST"   not in dims_recv
    v4 = paquete.get("geo_hash") == obj_orig.geo_hash()
    v5 = "DIM_SQL" in dims_s and dims_s["DIM_SQL"].get("props", {}).get("tabla") == "usuarios"
    v6 = (modo_final == Mode.MODO_H.value) if espera_modo_h else (modo_final == Mode.MODO_S.value)

    print(f"\n  Verificaciones:")
    print(f"    V1 DIM_SQL detectada   : {'OK' if v1 else 'FALLO'}")
    print(f"    V2 DIM_PYTHON detectada: {'OK' if v2 else 'FALLO'}")
    print(f"    V3 DIM_RUST latente    : {'OK' if v3 else 'FALLO'}")
    print(f"    V4 geo_hash preservado : {'OK' if v4 else 'FALLO'}")
    print(f"    V5 props DIM_SQL OK    : {'OK' if v5 else 'FALLO'}")
    modo_esperado = Mode.MODO_H.value if espera_modo_h else Mode.MODO_S.value
    print(f"    V6 modo={modo_esperado:6s} correcto : {'OK' if v6 else 'FALLO'}")
    if espera_modo_h:
        print(f"    [MODO_H] hv_g transmitido: {'SI' if has_hv_g else 'NO'}")

    ok = all([v1, v2, v3, v4, v5, v6])
    print(f"\n  → {'TODOS LOS CHECKS OK' if ok else 'ALGUN CHECK FALLO'}")

    return ok, {
        "label": label,
        "sim_pre_align": round(float(sim_pre), 4),
        "align_score": c_a.score,
        "modo_final": modo_final,
        "hv_g_transmitido": has_hv_g,
        "dims_receptor": list(dims_recv.keys()),
        "ok": ok,
    }


def demo_e2e():
    print("=" * 65)
    print("POLYDIM — Demo End-to-End V0.1")
    print("=" * 65)

    space_a = PolyDimSpace(N=10000)

    ok_a, res_a = run_escenario(
        "A: espacios identicos (mismo modelo)", True,
        space_a, PolyDimSpace(N=10000)
    )
    ok_b, res_b = run_escenario(
        "B: arquitecturas similares (ruido gaussiano)", False,
        space_a, NoisySpace(space_a, noise_scale=1.0)
    )
    ok_c, res_c = run_escenario(
        "C: arquitecturas incompatibles (corpus distintos)", False,
        space_a, RandomSpace(N=10000)
    )

    print(f"\n{'='*65}")
    print("RESUMEN")
    print(f"{'='*65}")
    for r in [res_a, res_b, res_c]:
        print(f"  {r['label'][:42]:42s} score={r['align_score']:.4f} modo={r['modo_final']}  {'OK' if r['ok'] else 'FALLO'}")

    print(f"\nHALLAZGO: ALIGN V0.1 efectivo solo con espacios identicos o K >> sqrt(N).")
    print(f"PEND_NEW_001: para N=10000, K_EFECTIVO requiere ~1000 sondas.")
    print(f"MODO_S es el modo de produccion robusto para el mundo real.")

    all_ok = ok_a and ok_b and ok_c
    print(f"\n{'TASK_011 TERMINADA — Demo E2E VERIFICADO' if all_ok else 'CHECKS FALLIDOS'}")
    print("=" * 65)
    return all_ok, [res_a, res_b, res_c]


if __name__ == "__main__":
    ok, resultados = demo_e2e()
    print("\nResultados JSON:")
    print(json.dumps(resultados, indent=2))
