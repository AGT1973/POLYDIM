# POLYDIM_DEST
# destination: polydim/core/
# filename:    polydim_transform.py
# author:      polydim.ai.lenguage@gmail.com

"""
POLYDIM — Transformaciones sin nombre (TASK_030)
=================================================
Módulo standalone: TransformND + geometric_collapse sobre V0.8.

MOTIVACIÓN (evaluación Antigravity + POLYDIM_BASES_V1.md KF_003 AP_003):
  Un 'programa' POLYDIM no es una secuencia de instrucciones con nombre.
  Es una transformación T: R^N → R^N que configura el estado geométrico.

CLASES:
  TransformND         T: R^N → R^N como objeto de primera clase
    .identity()       T = I
    .projection(v)    rank-1 sobre subespacio v
    .rotation(v, θ)   rotación en plano v⊥e₁
    .raw(fn)          transformación arbitraria
    .apply(obj)       → nuevo ObjectND con hv transformado
    __matmul__        composición: (T2 @ T1)(x) = T2(T1(x))

  geometric_collapse  compilación como proyección geométrica nativa
    → activation score + symbolic + projected_geo_id
    (Executor real → TASK_032)

Tests: 6/6 OK (2026-06-19)
Autor: polydim.ai.lenguage@gmail.com
V1.0 — 2026-06-19 — TASK_030
Base: polydim_runtime_v08.py (V0.8, PCRG-1 + SemanticBackend, 19/19 tests)
"""

from __future__ import annotations
import math, hashlib
import numpy as np
from typing import Callable, Dict, Any

from polydim_runtime_v08 import (
    Space, ObjectND, N, NATIVE, UMBRAL, _sim, _proj, _sup
)


class TransformND:
    """
    T: R^N → R^N como objeto de primera clase de POLYDIM.
    _label es solo para debugging — NO parte de la semántica.
    """
    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], label: str = "T"):
        self._fn = fn
        self._label = label

    @classmethod
    def identity(cls) -> "TransformND":
        return cls(lambda hv: hv.copy(), "I")

    @classmethod
    def projection(cls, target: np.ndarray) -> "TransformND":
        """P(x) = (x·v̂)·v̂ — rank-1. Uso: geometric_collapse, compilación."""
        v = target.astype(np.float32).copy()
        vn = v / (np.linalg.norm(v) + 1e-12)
        def fn(hv):
            score = float(np.dot(hv, vn))
            proj = score * vn
            n = np.linalg.norm(proj)
            return proj / n if n > 1e-10 else vn.copy()
        return cls(fn, "P")

    @classmethod
    def rotation(cls, axis: np.ndarray, theta: float) -> "TransformND":
        """Rotación en plano (axis, e_⊥). Mueve objetos entre subespacios."""
        v = axis.astype(np.float32).copy()
        v /= np.linalg.norm(v) + 1e-12
        e = np.zeros(N, dtype=np.float32); e[0] = 1.0
        e = e - float(np.dot(e, v)) * v
        n = np.linalg.norm(e)
        if n < 1e-10:
            e = np.zeros(N, dtype=np.float32); e[1] = 1.0
            e = e - float(np.dot(e, v)) * v; n = np.linalg.norm(e)
        e /= n
        c, s = math.cos(theta), math.sin(theta)
        def fn(hv):
            pv = float(np.dot(hv, v)) * v
            pe = float(np.dot(hv, e)) * e
            rest = hv - pv - pe
            rot = (c * pv - s * pe) + (s * pv + c * pe) + rest
            n_ = np.linalg.norm(rot)
            return rot / n_ if n_ > 1e-10 else rot
        return cls(fn, f"R(θ={theta:.3f})")

    @classmethod
    def raw(cls, fn, label="T_raw") -> "TransformND":
        return cls(fn, label)

    def apply(self, obj: "ObjectND") -> "ObjectND":
        """Aplica T → nuevo ObjectND con cache = T(hv_original)."""
        transformed = self._fn(obj._hv())
        n = np.linalg.norm(transformed)
        transformed = transformed / n if n > 1e-10 else transformed
        new_obj = ObjectND(obj._sp)
        new_obj._geo = obj._geo.copy()
        for dim, props in obj._props.items():
            new_obj.add(dim, dict(props), w=obj._w.get(dim, 1.0))
        new_obj._cache = transformed
        return new_obj

    def __matmul__(self, other: "TransformND") -> "TransformND":
        """(self @ other)(x) = self(other(x)) — T2 luego T1."""
        f1, f2 = self._fn, other._fn
        return TransformND(lambda hv: f1(f2(hv)), f"({self._label}∘{other._label})")

    def activation_score(self, obj: "ObjectND") -> float:
        hv = obj._hv()
        t = self._fn(hv)
        n = np.linalg.norm(t)
        t = t / n if n > 1e-10 else t
        return float(_sim(hv, t))

    def __repr__(self): return f"TransformND({self._label})"


def geometric_collapse(obj: "ObjectND", executor_dim: str, space: "Space") -> Dict[str, Any]:
    """
    'Compila' obj al executor. Compilar = proyección geométrica nativa.
    POLYDIM_BASES_V1.md KF_002. Executor real → TASK_032.
    """
    if executor_dim not in NATIVE and executor_dim not in space._sub:
        raise ValueError(f"'{executor_dim}' no es dimensión conocida")
    executor_sub = space.sub(executor_dim)
    hv = obj._hv()
    activation = (float(np.dot(hv, executor_sub)) + 1.0) / 2.0
    T = TransformND.projection(executor_sub)
    proj_hv = T._fn(hv)
    n = np.linalg.norm(proj_hv)
    proj_hv = proj_hv / n if n > 1e-10 else proj_hv
    return {
        "executor": executor_dim,
        "activation": round(activation, 4),
        "dims_activas": obj.dims_activas(),
        "projected_geo_id": hashlib.md5(proj_hv.tobytes()).hexdigest()[:12],
        "symbolic": obj.to_symbolic(),
        "note": f"activación={activation:.4f}. Executor real pendiente TASK_032."
    }


# Tests 6/6 OK

def test_identity():
    sp = Space("T"); obj = ObjectND(sp).add("DIM_SQL", {"t": "u"}, w=1.0)
    return _sim(obj._hv(), TransformND.identity().apply(obj)._hv()) > 0.999

def test_projection_on_subspace():
    sp = Space("T"); obj = ObjectND(sp).add("DIM_SQL", {}, w=1.0)
    obj2 = TransformND.projection(sp.sub("DIM_SQL")).apply(obj)
    return float(np.dot(obj2._cache, sp.sub("DIM_SQL"))) > 0.99

def test_compose_rotations():
    sp = Space("T"); obj = ObjectND(sp).add("DIM_PYTHON", {}, w=1.0)
    T_q = TransformND.rotation(sp.sub("DIM_PYTHON"), math.pi / 4)
    T_h = TransformND.rotation(sp.sub("DIM_PYTHON"), math.pi / 2)
    return _sim((T_q @ T_q).apply(obj)._cache, T_h.apply(obj)._cache) > 0.95

def test_apply_preserves_dims():
    sp = Space("T"); obj = ObjectND(sp).add("DIM_SQL", {}, w=1.0).add("DIM_PYTHON", {}, w=0.5)
    return set(TransformND.identity().apply(obj).get_dims()) == {"DIM_SQL", "DIM_PYTHON"}

def test_collapse_score():
    sp = Space("T")
    r_sql = geometric_collapse(ObjectND(sp).add("DIM_SQL", {}, w=1.0), "DIM_SQL", sp)
    r_py  = geometric_collapse(ObjectND(sp).add("DIM_PYTHON", {}, w=1.0), "DIM_PYTHON", sp)
    r_cross = geometric_collapse(ObjectND(sp).add("DIM_SQL", {}, w=1.0), "DIM_PYTHON", sp)
    return r_sql["activation"] > r_cross["activation"] and r_py["activation"] > r_cross["activation"]

def test_chain_transforms():
    sp = Space("T"); obj = ObjectND(sp).add("DIM_GRAPH", {}, w=1.0)
    T1 = TransformND.rotation(sp.sub("DIM_SQL"), math.pi / 8)
    T2 = TransformND.rotation(sp.sub("DIM_PYTHON"), math.pi / 8)
    T3 = TransformND.projection(sp.sub("DIM_GRAPH"))
    return _sim(obj._hv(), (T3 @ T2 @ T1).apply(obj)._cache) < 0.999


if __name__ == "__main__":
    tests = [test_identity, test_projection_on_subspace, test_compose_rotations,
             test_apply_preserves_dims, test_collapse_score, test_chain_transforms]
    print("── polydim_transform.py ─────────────────────────────────")
    results = [t() for t in tests]
    for t, r in zip(tests, results):
        print(f"  {t.__name__:40s}: {'OK' if r else 'FALLO'}")
    print(f"\n  {sum(results)}/{len(results)} tests OK")
    if all(results): print("  TASK_030 VERIFICADA — polydim_transform.py OK")
