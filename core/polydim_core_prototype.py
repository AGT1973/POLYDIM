"""
POLYDIM Core — Prototype V0.1
==============================
Primer codigo ejecutable del lenguaje POLYDIM.
Implementa OBJECT_ND con operaciones VSA (Vector Symbolic Architecture)
usando numpy como backend de Capa G.

Basado en:
  SPEC_OBJETO_ND_REVISION_V1.md
  SPEC_OPERACIONES_CAPA_G_V0.md
  SPEC_POLYDIM_SPACE_V0.md
  SPEC_ARQUITECTURA_AMBOS_V1.md

Autor:   ai.mpat.agt@gmail.com
Version: V0.1 — 2026-06-10
Estado:  EJECUTADO — pasa todos los invariantes
"""

import numpy as np
import hashlib
from typing import Dict, List, Optional

POLYDIM_N            = 10000
POLYDIM_SEED         = "POLYDIM_V1_SEED_2026"
UMBRAL_RECUPERACION  = 0.5
UMBRAL_IDENTIDAD     = 0.85
UMBRAL_ALIGN         = 0.85

NATIVE_DIMS = [
    "DIM_PYTHON", "DIM_RUST", "DIM_FLUTTER", "DIM_SQL",
    "DIM_GRAPH",  "DIM_VECTOR", "DIM_TIME",  "DIM_ERROR", "DIM_META",
]


def hv_bind(hv_a, hv_b):
    result = hv_a * hv_b
    norm = np.linalg.norm(result)
    return result / norm if norm > 1e-10 else result

def hv_unbind(hv_bound, hv_a):
    result = hv_bound * hv_a
    norm = np.linalg.norm(result)
    return result / norm if norm > 1e-10 else result

def hv_superpose(*hvs, weights=None):
    if weights is not None:
        combined = sum(w * hv for w, hv in zip(weights, hvs))
    else:
        combined = np.sum(hvs, axis=0)
    norm = np.linalg.norm(combined)
    return combined / norm if norm > 1e-10 else combined

def hv_project(hv, subspace):
    return (float(np.dot(hv, subspace)) + 1.0) / 2.0

def hv_sim(hv_a, hv_b):
    return float((np.dot(hv_a, hv_b) + 1.0) / 2.0)


class PolyDimSpace:
    def __init__(self, N=POLYDIM_N, seed=POLYDIM_SEED):
        self.N = N
        self.seed = seed
        self._symbols   = {}
        self._subspaces = {}
        for dim in NATIVE_DIMS:
            self._subspaces[dim] = self._make_symbol(dim)

    def _make_symbol(self, name):
        seed_int = int(hashlib.md5(name.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed_int)
        hv  = rng.standard_normal(self.N).astype(np.float32)
        return hv / np.linalg.norm(hv)

    def get_symbol(self, name):
        if name not in self._symbols:
            self._symbols[name] = self._make_symbol(name)
        return self._symbols[name]

    def get_subspace(self, dim_name):
        if dim_name not in self._subspaces:
            self._subspaces[dim_name] = self.get_symbol(dim_name)
        return self._subspaces[dim_name]

    def random_hv(self):
        hv = np.random.randn(self.N).astype(np.float32)
        return hv / np.linalg.norm(hv)

    def encode_props(self, props):
        if not props:
            return self.get_symbol("__empty__")
        pairs = [hv_bind(self.get_symbol(str(k)), self.get_symbol(str(v)))
                 for k, v in props.items()]
        return hv_superpose(*pairs)


SPACE = PolyDimSpace()


class Dimension:
    def __init__(self, name, props, space=SPACE):
        self.name  = name
        self.props = props
        self._space = space
        self._hv   = None

    def to_hv(self):
        if self._hv is None:
            self._hv = hv_bind(
                self._space.get_subspace(self.name),
                self._space.encode_props(self.props)
            )
        return self._hv


class ObjectND:
    def __init__(self, space=SPACE):
        self._space    = space
        self._dims     = {}
        self._weights  = {}
        self._geo_id   = space.random_hv()
        self._hv_cache = None

    @property
    def geo_id(self):
        return self._geo_id

    def geo_hash(self):
        return hashlib.md5(self._geo_id.tobytes()).hexdigest()[:12]

    def add_dim(self, name, props, weight=1.0):
        self._dims[name]    = Dimension(name, props, self._space)
        self._weights[name] = float(np.clip(weight, 0.0, 1.0))
        self._hv_cache      = None
        return self

    def set_weight(self, name, weight):
        if name in self._weights:
            self._weights[name] = float(np.clip(weight, 0.0, 1.0))
            self._hv_cache      = None
        return self

    def to_hv(self):
        if self._hv_cache is not None:
            return self._hv_cache
        components = [self._geo_id]
        ws         = [1.0]
        for name, dim in self._dims.items():
            w = self._weights.get(name, 1.0)
            if w > 0.0:
                components.append(dim.to_hv())
                ws.append(w)
        self._hv_cache = hv_superpose(*components, weights=ws)
        return self._hv_cache

    def activate(self, dim_name):
        return hv_project(self.to_hv(), self._space.get_subspace(dim_name))

    def active_dims(self, threshold=UMBRAL_RECUPERACION):
        result = {}
        for d in NATIVE_DIMS:
            w = self.activate(d)
            if w > threshold:
                result[d] = w
        for d in self._dims:
            if d not in result:
                w = self.activate(d)
                if w > threshold:
                    result[d] = w
        return result

    def similarity(self, other):
        return hv_sim(self.to_hv(), other.to_hv())

    def is_same_object(self, other):
        return hv_sim(self._geo_id, other._geo_id) > UMBRAL_IDENTIDAD

    def to_symbolic(self):
        return {
            "geo_id":     self.geo_hash(),
            "dimensions": {
                n: {"weight_declared":  self._weights.get(n, 1.0),
                    "weight_geometric": round(self.activate(n), 4),
                    "props":            d.props}
                for n, d in self._dims.items()
            },
            "active_g": {k: round(v, 4) for k, v in self.active_dims().items()},
        }

    def __repr__(self):
        dims = ", ".join(f"{n}[{self._weights.get(n,1.0):.1f}]" for n in self._dims)
        return f"ObjectND(geo={self.geo_hash()}, dims=[{dims}])"


def nd_merge(obj_a, obj_b, space=SPACE):
    merged = ObjectND(space)
    merged._geo_id = hv_bind(obj_a.geo_id, obj_b.geo_id)
    for n, d in obj_a._dims.items():
        merged.add_dim(n, d.props, obj_a._weights.get(n, 1.0))
    for n, d in obj_b._dims.items():
        if n not in merged._dims:
            merged.add_dim(n, d.props, obj_b._weights.get(n, 1.0))
    merged._hv_cache = hv_superpose(obj_a.to_hv(), obj_b.to_hv())
    return merged

def nd_distance(obj_a, obj_b):
    return 1.0 - hv_sim(obj_a.to_hv(), obj_b.to_hv())
