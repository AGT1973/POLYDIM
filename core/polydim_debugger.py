# POLYDIM_DEST
# destination: polydim/core/
# filename:    polydim_debugger.py
# author:      claude-sonnet-4-6 (curso05.mithril@gmail.com)

"""
POLYDIM PolydimDebugger — TASK_031
=====================================
Resuelve AP_004: separar to_symbolic() del runtime core.

AP_004 (KF_003):
    to_symbolic() en el core significa que POLYDIM depende de
    representación humana para funcionar. La Capa S (simbólica)
    debe ser un debugger externo, no parte del lenguaje.

    "Este es un telescopio apuntando a POLYDIM, no parte del lenguaje."

RESULTADOS VERIFICADOS (6/6 tests, ejecución real):
    inspect() → dict JSON con dims_activas, dims_registradas, geo_id
    explain() → string legible para humanos
    diff()    → similitud coseno + dims compartidas / únicas
    nearest_dims() → top-K dims ordenadas por proyección
    trace()   → historial de transfers por sesión

DEMO OUTPUT:
    ObjectND(geo_id=2ed8bd2e9831)
      Dims activas (2):
        DIM_SQL    → 0.8150
        DIM_PYTHON → 0.7178
      nearest_dims: [DIM_SQL, DIM_PYTHON, DIM_GRAPH, DIM_META]

Prerequisito: polydim_runtime_v07.py
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Any

from polydim_runtime_v07 import (
    Space, ObjectND, Session, N, UMBRAL, NATIVE,
    _sim, _proj
)


class PolydimDebugger:
    """
    Observador externo de objetos POLYDIM.
    Traduce geometría a JSON/texto para humanos.
    No es parte del lenguaje — es un telescopio apuntando al lenguaje.

    Uso:
        dbg = PolydimDebugger(space)
        print(dbg.explain(obj))
        d = dbg.inspect(obj)
    """

    def __init__(self, space: Space):
        self.space = space
        self._history: List[Dict[str, Any]] = []

    def inspect(self, obj: ObjectND) -> Dict[str, Any]:
        """Convierte ObjectND a dict JSON legible por humanos (Capa S externa)."""
        hv = obj._hv()
        dims_activas = {d: round(_proj(hv, self.space.sub(d)), 4)
                        for d in NATIVE if _proj(hv, self.space.sub(d)) > UMBRAL}
        return {
            "geo_id":           obj.geo_id,
            "dims_activas":     dims_activas,
            "dims_registradas": {d: {"w": obj._w.get(d, 1.0), "props": p}
                                 for d, p in obj._props.items()},
            "n_dims":           len(obj._props),
            "nota":             "inspeccion externa — no parte del lenguaje POLYDIM"
        }

    def explain(self, obj: ObjectND) -> str:
        """Explicación textual para humanos."""
        d     = self.inspect(obj)
        lines = [f"ObjectND(geo_id={d['geo_id']})"]
        lines.append(f"  Dims activas ({len(d['dims_activas'])}):")
        for dim, act in sorted(d["dims_activas"].items(), key=lambda x: -x[1]):
            lines.append(f"    {dim:20s} → {act:.4f}")
        if d["dims_registradas"]:
            lines.append(f"  Dims registradas ({d['n_dims']}):")
            for dim, info in d["dims_registradas"].items():
                w = info["w"]; p = info["props"]
                prop_str = str(p) if p and not p.get("__gp__") else "(geométrico)"
                lines.append(f"    {dim:20s} w={w:.2f} {prop_str}")
        return "\n".join(lines)

    def diff(self, obj1: ObjectND, obj2: ObjectND) -> Dict[str, Any]:
        """Compara dos ObjectNDs geométricamente (no por JSON)."""
        hv1 = obj1._hv(); hv2 = obj2._hv()
        sim  = _sim(hv1, hv2)
        dims1 = {d for d in NATIVE if _proj(hv1, self.space.sub(d)) > UMBRAL}
        dims2 = {d for d in NATIVE if _proj(hv2, self.space.sub(d)) > UMBRAL}
        return {
            "similitud_coseno": round(sim, 4),
            "son_similares":    sim > 0.85,
            "dims_solo_en_1":   sorted(dims1 - dims2),
            "dims_solo_en_2":   sorted(dims2 - dims1),
            "dims_compartidas": sorted(dims1 & dims2),
            "geo_ids":          [obj1.geo_id, obj2.geo_id],
        }

    def nearest_dims(self, obj: ObjectND, top_k: int = 5) -> List[tuple]:
        """Top-K dimensiones NATIVE más cercanas al obj."""
        hv = obj._hv()
        return sorted([(d, round(_proj(hv, self.space.sub(d)), 4)) for d in NATIVE],
                      key=lambda x: -x[1])[:top_k]

    def log_transfer(self, obj: ObjectND, session_name: str, dims_received: Dict[str, float]):
        """Registra un transfer en el historial."""
        self._history.append({
            "geo_id":        obj.geo_id,
            "session":       session_name,
            "dims_received": dims_received,
            "step":          len(self._history) + 1,
        })

    def trace(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def clear_trace(self):
        self._history.clear()

    def __repr__(self) -> str:
        return f"PolydimDebugger(space={self.space.ps!r}, history={len(self._history)})"


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_inspect_returns_dict():
    sp = Space("DBG_INSPECT"); dbg = PolydimDebugger(sp)
    obj = ObjectND(sp).add("DIM_SQL", {"tabla": "usuarios"}, w=1.0)
    d   = dbg.inspect(obj)
    return "geo_id" in d and "dims_activas" in d and "DIM_SQL" in d["dims_activas"]

def test_explain_is_string():
    sp = Space("DBG_EXPLAIN"); dbg = PolydimDebugger(sp)
    s  = dbg.explain(ObjectND(sp).add("DIM_PYTHON", {}, w=1.0))
    return isinstance(s, str) and "DIM_PYTHON" in s and "geo_id" in s

def test_diff_similar_objects():
    sp = Space("DBG_DIFF"); dbg = PolydimDebugger(sp)
    obj1 = ObjectND(sp).add("DIM_SQL", {}, w=1.0)
    obj2 = ObjectND(sp).add("DIM_SQL", {}, w=1.0)
    d    = dbg.diff(obj1, obj2)
    return d["similitud_coseno"] > 0.5 and "DIM_SQL" in d["dims_compartidas"]

def test_diff_distinct_objects():
    sp = Space("DBG_DIFF2"); dbg = PolydimDebugger(sp)
    d = dbg.diff(ObjectND(sp).add("DIM_SQL", {}, w=1.0),
                 ObjectND(sp).add("DIM_FLUTTER", {}, w=1.0))
    return "DIM_SQL" in d["dims_solo_en_1"] and "DIM_FLUTTER" in d["dims_solo_en_2"]

def test_nearest_dims():
    sp = Space("DBG_NEAR"); dbg = PolydimDebugger(sp)
    top = dbg.nearest_dims(ObjectND(sp).add("DIM_RUST", {}, w=1.0), top_k=3)
    return len(top) == 3 and top[0][0] == "DIM_RUST" and top[0][1] >= top[1][1] >= top[2][1]

def test_log_and_trace():
    sp = Space("DBG_TRACE"); dbg = PolydimDebugger(sp)
    obj = ObjectND(sp).add("DIM_SQL", {}, w=1.0)
    dbg.log_transfer(obj, "IA_A", {"DIM_SQL": 0.82})
    dbg.log_transfer(obj, "IA_B", {"DIM_SQL": 0.79})
    t = dbg.trace()
    return len(t) == 2 and t[0]["session"] == "IA_A" and t[0]["step"] == 1


if __name__ == "__main__":
    tests = [test_inspect_returns_dict, test_explain_is_string,
             test_diff_similar_objects, test_diff_distinct_objects,
             test_nearest_dims, test_log_and_trace]
    results = [t() for t in tests]
    for t, r in zip(tests, results):
        print(f"  {t.__name__:48s}: {'OK' if r else 'FALLO'}")
    sp  = Space("DEMO"); dbg = PolydimDebugger(sp)
    obj = ObjectND(sp).add("DIM_SQL", {"tabla": "pedidos"}, w=1.0).add("DIM_PYTHON", {"clase": "PedidoService"}, w=0.7)
    print(f"\n── Demo ─────────────────────────────────────────────────────\n{dbg.explain(obj)}")
    print(f"\n  nearest: {dbg.nearest_dims(obj, 4)}")
    print(f"\n  [Telescopio, no lenguaje.]")
    passed = sum(results)
    print(f"\n  {passed}/{len(results)} tests OK -- {'TASK_031 VERIFICADA' if passed == len(results) else 'CHECKS FALLIDOS'}")
