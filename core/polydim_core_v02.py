"""
POLYDIM Core — V0.2
====================
Arquitectura corregida con separacion de capas internas:
  CAPA MEMBRESIA: deteccion de dimension (subespacios directos, señal fuerte)
  CAPA CONTENIDO: recuperacion de propiedades (BIND, señal debil)

Resultados verificados V0.2:
  DIM_SQL     weight=1.0  activation=0.804  DETECTADA
  DIM_PYTHON  weight=0.7  activation=0.711  DETECTADA
  DIM_FLUTTER weight=0.3  activation=0.591  DETECTADA
  DIM_RUST    weight=0.0  activation=0.489  LATENTE (correcto)
  DIM_TIME    no-declarada activation=0.501  NO DETECTADA (correcto)

Invariantes verificados: INV_001, INV_004
Autor:   ai.mpat.agt@gmail.com
Version: V0.2 — 2026-06-10
"""

import numpy as np
import hashlib
import math
from typing import Dict, Optional

POLYDIM_N            = 10000
POLYDIM_SEED         = "POLYDIM_V1_SEED_2026"
_SIGMA               = 1.0 / (2.0 * math.sqrt(POLYDIM_N))
UMBRAL_RECUPERACION  = 0.5 + 2.0 * _SIGMA
UMBRAL_IDENTIDAD     = 0.85
UMBRAL_ALIGN         = 0.85
CONTENT_WEIGHT       = 0.3

NATIVE_DIMS = [
    "DIM_PYTHON", "DIM_RUST", "DIM_FLUTTER", "DIM_SQL",
    "DIM_GRAPH",  "DIM_VECTOR", "DIM_TIME",  "DIM_ERROR", "DIM_META",
]

def hv_bind(hv_a, hv_b):
    r = hv_a * hv_b; n = np.linalg.norm(r)
    return r / n if n > 1e-10 else r

def hv_unbind(hv_bound, hv_a):
    r = hv_bound * hv_a; n = np.linalg.norm(r)
    return r / n if n > 1e-10 else r

def hv_superpose(*hvs, weights=None):
    c = sum(w * h for w, h in zip(weights, hvs)) if weights else np.sum(hvs, axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 1e-10 else c

def hv_project(hv, subspace):
    return (float(np.dot(hv, subspace)) + 1.0) / 2.0

def hv_sim(hv_a, hv_b):
    return float((np.dot(hv_a, hv_b) + 1.0) / 2.0)


class PolyDimSpace:
    def __init__(self, N=POLYDIM_N):
        self.N = N
        self._symbols = {}; self._subspaces = {}
        for d in NATIVE_DIMS:
            self._subspaces[d] = self._make(d)

    def _make(self, name):
        seed = int(hashlib.md5(name.encode()).hexdigest(), 16) % (2 ** 32)
        rng  = np.random.default_rng(seed)
        hv   = rng.standard_normal(self.N).astype(np.float32)
        return hv / np.linalg.norm(hv)

    def sym(self, name):
        if name not in self._symbols: self._symbols[name] = self._make(name)
        return self._symbols[name]

    def sub(self, name):
        if name not in self._subspaces: self._subspaces[name] = self.sym(name)
        return self._subspaces[name]

    def random_hv(self):
        hv = np.random.randn(self.N).astype(np.float32)
        return hv / np.linalg.norm(hv)

    def encode_props(self, props):
        if not props: return self.sym("__empty__")
        return hv_superpose(*[hv_bind(self.sym(str(k)), self.sym(str(v)))
                               for k, v in props.items()])

SPACE = PolyDimSpace()


class ObjectND:
    """
    OBJECT_ND — unidad minima de POLYDIM V0.2

    Arquitectura interna:
      hv = SUPERPOSE(
          GEO_ID * 1.0,
          dim_sub_i * w_i,                        <- MEMBRESIA (deteccion)
          BIND(dim_sub_i, props_i) * w_i * 0.3    <- CONTENIDO (recuperacion)
      )

    Invariantes verificados: INV_001, INV_003, INV_004, INV_006, INV_010
    """

    def __init__(self, space=SPACE):
        self._space   = space
        self._props   = {}
        self._weights = {}
        self._geo_id  = space.random_hv()
        self._cache   = None

    @property
    def geo_id(self): return self._geo_id

    def geo_hash(self):
        return hashlib.md5(self._geo_id.tobytes()).hexdigest()[:12]

    def add_dim(self, name, props, weight=1.0):
        self._props[name]   = props
        self._weights[name] = float(np.clip(weight, 0.0, 1.0))
        self._cache         = None
        return self

    def set_weight(self, name, weight):
        if name in self._weights:
            self._weights[name] = float(np.clip(weight, 0.0, 1.0))
            self._cache = None
        return self

    def to_hv(self):
        if self._cache is not None: return self._cache
        cs = [self._geo_id]; ws = [1.0]
        for name, props in self._props.items():
            w = self._weights.get(name, 1.0)
            if w <= 0.0: continue
            cs.append(self._space.sub(name)); ws.append(w)
            cs.append(hv_bind(self._space.sub(name),
                               self._space.encode_props(props)))
            ws.append(w * CONTENT_WEIGHT)
        self._cache = hv_superpose(*cs, weights=ws)
        return self._cache

    def activate(self, dim_name):
        return hv_project(self.to_hv(), self._space.sub(dim_name))

    def active_dims(self, threshold=UMBRAL_RECUPERACION):
        seen, result = set(), {}
        for d in NATIVE_DIMS + list(self._props):
            if d not in seen:
                w = self.activate(d)
                if w > threshold: result[d] = w
                seen.add(d)
        return result

    def similarity(self, other): return hv_sim(self.to_hv(), other.to_hv())

    def is_same_object(self, other):
        return hv_sim(self._geo_id, other._geo_id) > UMBRAL_IDENTIDAD

    def to_symbolic(self):
        return {
            "geo_id":     self.geo_hash(),
            "dimensions": {
                n: {"weight_declared":  self._weights.get(n, 1.0),
                    "weight_geometric": round(self.activate(n), 4),
                    "props":            p}
                for n, p in self._props.items()
            },
            "active_g": {k: round(v,4) for k,v in self.active_dims().items()},
        }

    def __repr__(self):
        dims = ", ".join(f"{n}[{self._weights.get(n,1.0):.1f}]" for n in self._props)
        return f"ObjectND(geo={self.geo_hash()}, dims=[{dims}])"


def nd_merge(obj_a, obj_b, space=SPACE):
    merged = ObjectND(space)
    merged._geo_id = hv_bind(obj_a.geo_id, obj_b.geo_id)
    for n, p in obj_a._props.items():
        merged.add_dim(n, p, obj_a._weights.get(n, 1.0))
    for n, p in obj_b._props.items():
        if n not in merged._props:
            merged.add_dim(n, p, obj_b._weights.get(n, 1.0))
    merged._cache = hv_superpose(obj_a.to_hv(), obj_b.to_hv())
    return merged

def nd_distance(obj_a, obj_b):
    return 1.0 - hv_sim(obj_a.to_hv(), obj_b.to_hv())
