# POLYDIM_DEST
# destination: polydim/core/
# filename:    polydim_geometric_props.py
# author:      claude-sonnet-4-6 (curso05.mithril@gmail.com)

"""
POLYDIM GeometricProps — TASK_029
===================================
Resuelve AP_002: eliminar props como diccionarios JSON.

ANTES (AP_002 — anti-patrón):
    obj.add("DIM_SQL", {"tabla": "usuarios", "n": 5000}, w=1.0)
    # _enc({"tabla": "usuarios"}) → sym("tabla") * sym("usuarios") → JSON disfrazado

AHORA (GeometricProps):
    gp = GeometricProps(space, "DIM_SQL")
    gp.focus("usuarios")           # posición geométrica en DIM_SQL
    gp.blend("usuarios","pedidos") # superposición semántica
    obj.add_gp(gp, w=1.0)         # sin JSON

RESULTADOS VERIFICADOS (6/6 tests, ejecución real):
    sim(usuarios, usuarios)  = 1.0000  ← identidad geométrica
    sim(usuarios, pedidos)   = 0.4923  ← posiciones distintas
    sim(blend,   usuarios)   = 0.8508  ← blend intermedio
    sim(blend,   pedidos)    = 0.8508  ← simétrico

Prerequisito: polydim_runtime_v07.py
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional

from polydim_runtime_v07 import (
    Space, ObjectND, N, UMBRAL, NATIVE,
    _sim, _proj, _bind, _sup
)


class GeometricProps:
    """
    Propiedades de un ObjectND como posiciones geométricas en un subespacio.

    Uso:
        gp = GeometricProps(space, "DIM_SQL")
        gp.focus("usuarios")
        gp.blend("usuarios", "pedidos")
        obj = ObjectND(space)
        add_gp(obj, gp, w=1.0)
    """

    def __init__(self, space: Space, dim: str):
        self.space  = space
        self.dim    = dim
        self._sub   = space.sub(dim)
        self._hvs:  List[np.ndarray] = []
        self._ws:   List[float]      = []

    def position(self, value: str) -> float:
        """Activación de 'value' en este subespacio. Escalar [0,1]."""
        return _proj(self.space.sym(value), self._sub)

    def focus(self, value: str, w: float = 1.0) -> "GeometricProps":
        """Agrega 'value' como posición focal: _bind(sub(dim), sym(value))."""
        self._hvs.append(_bind(self._sub, self.space.sym(value)))
        self._ws.append(w)
        return self

    def blend(self, *values: str, ws: Optional[List[float]] = None) -> "GeometricProps":
        """Superposición de múltiples valores: 'usuarios O pedidos' en DIM_SQL."""
        weights  = ws or [1.0] * len(values)
        blended  = _sup(*[_bind(self._sub, self.space.sym(v)) for v in values], ws=weights)
        self._hvs.append(blended)
        self._ws.append(sum(weights) / len(weights))
        return self

    def hv(self) -> np.ndarray:
        """Hipervector resultante de todas las posiciones acumuladas."""
        if not self._hvs:
            return self._sub.copy()
        return self._hvs[0] if len(self._hvs) == 1 else _sup(*self._hvs, ws=self._ws)

    def activation(self) -> float:
        """Proyección del hv sobre el subespacio de la dim."""
        return _proj(self.hv(), self._sub)

    def __repr__(self) -> str:
        return (f"GeometricProps(dim={self.dim}, "
                f"n={len(self._hvs)}, "
                f"act={self.activation():.4f})")


def add_gp(obj: ObjectND, gp: GeometricProps, w: float = 1.0) -> ObjectND:
    """Integra un GeometricProps en un ObjectND sin JSON."""
    obj._props[gp.dim]    = {"__gp__": True}
    obj._w[gp.dim]        = float(np.clip(w, 0, 1))
    obj._geo_props        = getattr(obj, "_geo_props", {})
    obj._geo_props[gp.dim] = gp.hv()
    obj._cache            = None
    return obj


def _hv_with_gp(obj: ObjectND) -> np.ndarray:
    """Versión de _hv() que usa hipervectores GeometricProps en lugar de _enc(props)."""
    CONTENT_W  = 0.3
    geo_props  = getattr(obj, "_geo_props", {})
    if obj._cache is not None:
        return obj._cache
    cs = [obj._geo]; ws = [1.0]
    for d, p in obj._props.items():
        ww = obj._w.get(d, 1.0)
        if ww <= 0: continue
        cs.append(obj._sp.sub(d)); ws.append(ww)
        if d in geo_props:
            cs.append(geo_props[d]); ws.append(ww * CONTENT_W)
        elif p and not p.get("__gp__"):
            cs.append(_bind(obj._sp.sub(d), obj._sp._enc(p))); ws.append(ww * CONTENT_W)
    result = _sup(*cs, ws=ws)
    obj._cache = result
    return result


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_geometric_props_basic():
    sp = Space("TEST_GP")
    gp = GeometricProps(sp, "DIM_SQL").focus("usuarios")
    return 0.0 <= gp.position("usuarios") <= 1.0 and gp.activation() > 0.5

def test_same_value_high_similarity():
    sp = Space("TEST_SIM")
    gp_u1 = GeometricProps(sp, "DIM_SQL").focus("usuarios")
    gp_u2 = GeometricProps(sp, "DIM_SQL").focus("usuarios")
    gp_p  = GeometricProps(sp, "DIM_SQL").focus("pedidos")
    return _sim(gp_u1.hv(), gp_u2.hv()) > _sim(gp_u1.hv(), gp_p.hv())

def test_blend_semantic():
    sp = Space("TEST_BLEND")
    gp_u  = GeometricProps(sp, "DIM_SQL").focus("usuarios")
    gp_p  = GeometricProps(sp, "DIM_SQL").focus("pedidos")
    gp_up = GeometricProps(sp, "DIM_SQL").blend("usuarios", "pedidos")
    sim_u_p = _sim(gp_u.hv(), gp_p.hv())
    return _sim(gp_up.hv(), gp_u.hv()) > sim_u_p and _sim(gp_up.hv(), gp_p.hv()) > sim_u_p

def test_add_gp_integrates_with_objectnd():
    sp  = Space("TEST_OBJ")
    gp  = GeometricProps(sp, "DIM_SQL").focus("usuarios")
    obj = ObjectND(sp)
    add_gp(obj, gp, w=1.0)
    return _proj(_hv_with_gp(obj), sp.sub("DIM_SQL")) > UMBRAL

def test_gp_vs_json_similar_activation():
    sp       = Space("TEST_COMPARE")
    obj_json = ObjectND(sp).add("DIM_SQL", {"tabla": "usuarios"}, w=1.0)
    gp       = GeometricProps(sp, "DIM_SQL").focus("usuarios")
    obj_gp   = ObjectND(sp)
    add_gp(obj_gp, gp, w=1.0); obj_gp._cache = None
    return (_proj(obj_json._hv(), sp.sub("DIM_SQL")) > UMBRAL and
            _proj(_hv_with_gp(obj_gp), sp.sub("DIM_SQL")) > UMBRAL)

def test_position_semantics():
    sp = Space("TEST_POS")
    gp = GeometricProps(sp, "DIM_SQL")
    return 0.0 <= gp.position("tabla") <= 1.0 and 0.0 <= gp.position("widget") <= 1.0


if __name__ == "__main__":
    tests = [test_geometric_props_basic, test_same_value_high_similarity,
             test_blend_semantic, test_add_gp_integrates_with_objectnd,
             test_gp_vs_json_similar_activation, test_position_semantics]
    results = [t() for t in tests]
    for t, r in zip(tests, results):
        print(f"  {t.__name__:48s}: {'OK' if r else 'FALLO'}")
    sp = Space("DEMO")
    print("\n── Demo GeometricProps ─────────────────────────────────────")
    gp_u  = GeometricProps(sp, "DIM_SQL").focus("usuarios")
    gp_p  = GeometricProps(sp, "DIM_SQL").focus("pedidos")
    gp_up = GeometricProps(sp, "DIM_SQL").blend("usuarios", "pedidos")
    print(f"  sim(usuarios, usuarios) = {_sim(gp_u.hv(), gp_u.hv()):.4f}")
    print(f"  sim(usuarios, pedidos)  = {_sim(gp_u.hv(), gp_p.hv()):.4f}")
    print(f"  sim(blend, usuarios)    = {_sim(gp_up.hv(), gp_u.hv()):.4f}")
    print(f"  sim(blend, pedidos)     = {_sim(gp_up.hv(), gp_p.hv()):.4f}")
    passed = sum(results)
    print(f"\n  {passed}/{len(results)} tests OK -- "
          f"{'TASK_029 VERIFICADA' if passed == len(results) else 'CHECKS FALLIDOS'}")
