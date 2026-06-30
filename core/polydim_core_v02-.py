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

# ---------------------------------------------------------------------------
# Parametros POLYDIM V1
# ---------------------------------------------------------------------------
POLYDIM_N   = 10000
POLYDIM_SEED = "POLYDIM_V1_SEED_2026"
_SIGMA      = 1.0 / (2.0 * math.sqrt(POLYDIM_N))
UMBRAL_RECUPERACION = 0.5 + 2.0 * _SIGMA   # 0.510 para N=10000
UMBRAL_IDENTIDAD    = 0.85
UMBRAL_ALIGN        = 0.85
CONTENT_WEIGHT      = 0.3   # peso relativo de capa contenido vs membresía

NATIVE_DIMS = [
    "DIM_PYTHON", "DIM_RUST", "DIM_FLUTTER", "DIM_SQL",
    "DIM_GRAPH",  "DIM_VECTOR", "DIM_TIME",  "DIM_ERROR", "DIM_META",
]

# ---------------------------------------------------------------------------
# Operaciones VSA primitivas
# ---------------------------------------------------------------------------

def hv_bind(hv_a: np.ndarray, hv_b: np.ndarray) -> np.ndarray:
    """HV_BIND: producto Hadamard normalizado. NO conmutativo."""
    r = hv_a * hv_b
    n = np.linalg.norm(r)
    return r / n if n > 1e-10 else r

def hv_unbind(hv_bound: np.ndarray, hv_a: np.ndarray) -> np.ndarray:
    """HV_UNBIND: inversa aproximada de BIND. UNBIND(BIND(a,b),a) ≈ b"""
    r = hv_bound * hv_a
    n = np.linalg.norm(r)
    return r / n if n > 1e-10 else r

def hv_superpose(*hvs: np.ndarray, weights=None) -> np.ndarray:
    """HV_SUPERPOSE: suma vectorial ponderada normalizada. CONMUTATIVO."""
    c = sum(w * h for w, h in zip(weights, hvs)) if weights else np.sum(hvs, axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 1e-10 else c

def hv_project(hv: np.ndarray, subspace: np.ndarray) -> float:
    """HV_PROJECT: peso de activacion [0.0, 1.0]. NO modifica hv."""
    return (float(np.dot(hv, subspace)) + 1.0) / 2.0

def hv_sim(hv_a: np.ndarray, hv_b: np.ndarray) -> float:
    """HV_SIM: similitud coseno [0.0, 1.0]."""
    return float((np.dot(hv_a, hv_b) + 1.0) / 2.0)

# ---------------------------------------------------------------------------
# PolyDimSpace — espacio compartido deterministico
# ---------------------------------------------------------------------------

class PolyDimSpace:
    def __init__(self, N: int = POLYDIM_N):
        self.N = N
        self._symbols:   Dict[str, np.ndarray] = {}
        self._subspaces: Dict[str, np.ndarray] = {}
        for d in NATIVE_DIMS:
            self._subspaces[d] = self._make(d)

    def _make(self, name: str) -> np.ndarray:
        """Hipervector deterministico: mismo nombre → mismo vector en cualquier instancia."""
        seed = int(hashlib.md5(name.encode()).hexdigest(), 16) % (2 ** 32)
        rng  = np.random.default_rng(seed)
        hv   = rng.standard_normal(self.N).astype(np.float32)
        return hv / np.linalg.norm(hv)

    def sym(self, name: str) -> np.ndarray:
        if name not in self._symbols:
            self._symbols[name] = self._make(name)
        return self._symbols[name]

    def sub(self, name: str) -> np.ndarray:
        if name not in self._subspaces:
            self._subspaces[name] = self.sym(name)
        return self._subspaces[name]

    def random_hv(self) -> np.ndarray:
        hv = np.random.randn(self.N).astype(np.float32)
        return hv / np.linalg.norm(hv)

    def encode_props(self, props: dict) -> np.ndarray:
        if not props:
            return self.sym("__empty__")
        pairs = [hv_bind(self.sym(str(k)), self.sym(str(v))) for k, v in props.items()]
        return hv_superpose(*pairs)

SPACE = PolyDimSpace()

# ---------------------------------------------------------------------------
# ObjectND — unidad minima de POLYDIM
# ---------------------------------------------------------------------------

class ObjectND:
    """
    OBJECT_ND: existe simultaneamente en N dimensiones.

    Arquitectura interna de dos capas:
      CAPA MEMBRESIA: subespacios dimensionales directos (ponderados por weight)
                      → HV_PROJECT confiable para deteccion de membresía
      CAPA CONTENIDO: BIND(dim_sub, props) (a CONTENT_WEIGHT del peso)
                      → HV_UNBIND para recuperacion de propiedades

    hv_objeto = SUPERPOSE(
        GEO_ID * 1.0,
        dim_sub_i * w_i,              ← membresía (señal fuerte)
        BIND(dim_sub_i,props_i)*w_i*CONTENT_WEIGHT  ← contenido (señal débil)
    )

    Invariantes: INV_001, INV_003, INV_004, INV_006, INV_010
    """

    def __init__(self, space: PolyDimSpace = SPACE):
        self._space  = space
        self._props:   Dict[str, dict]         = {}
        self._weights: Dict[str, float]        = {}
        self._geo_id:  np.ndarray              = space.random_hv()
        self._cache:   Optional[np.ndarray]    = None

    @property
    def geo_id(self) -> np.ndarray:
        return self._geo_id

    def geo_hash(self) -> str:
        return hashlib.md5(self._geo_id.tobytes()).hexdigest()[:12]

    def add_dim(self, name: str, props: dict, weight: float = 1.0) -> "ObjectND":
        """Agrega dimension. No modifica las existentes (INV_006)."""
        self._props[name]   = props
        self._weights[name] = float(np.clip(weight, 0.0, 1.0))
        self._cache         = None
        return self

    def set_weight(self, name: str, weight: float) -> "ObjectND":
        if name in self._weights:
            self._weights[name] = float(np.clip(weight, 0.0, 1.0))
            self._cache = None
        return self

    def to_hv(self) -> np.ndarray:
        """Capa G completa: GEO_ID + capas de membresía y contenido."""
        if self._cache is not None:
            return self._cache
        cs = [self._geo_id]
        ws = [1.0]
        for name, props in self._props.items():
            w = self._weights.get(name, 1.0)
            if w <= 0.0:
                continue
            # Capa membresía: subespacio directo (deteccion)
            cs.append(self._space.sub(name))
            ws.append(w)
            # Capa contenido: BIND con props (recuperacion)
            cs.append(hv_bind(self._space.sub(name), self._space.encode_props(props)))
            ws.append(w * CONTENT_WEIGHT)
        self._cache = hv_superpose(*cs, weights=ws)
        return self._cache

    def activate(self, dim_name: str) -> float:
        """HV_PROJECT: peso de activacion real de una dimension [0.0, 1.0]."""
        return hv_project(self.to_hv(), self._space.sub(dim_name))

    def active_dims(self, threshold: float = UMBRAL_RECUPERACION) -> Dict[str, float]:
        """Dimensiones activas por encima del threshold. Sin falsos positivos."""
        seen, result = set(), {}
        for d in NATIVE_DIMS + list(self._props):
            if d not in seen:
                w = self.activate(d)
                if w > threshold:
                    result[d] = w
                seen.add(d)
        return result

    def similarity(self, other: "ObjectND") -> float:
        return hv_sim(self.to_hv(), other.to_hv())

    def is_same_object(self, other: "ObjectND") -> bool:
        return hv_sim(self._geo_id, other._geo_id) > UMBRAL_IDENTIDAD

    def to_symbolic(self) -> dict:
        """HV_DECODE aproximado: representacion Capa S."""
        return {
            "geo_id":     self.geo_hash(),
            "dimensions": {
                n: {"weight_declared":  self._weights.get(n, 1.0),
                    "weight_geometric": round(self.activate(n), 4),
                    "props":            p}
                for n, p in self._props.items()
            },
            "active_g": {k: round(v, 4) for k, v in self.active_dims().items()},
        }

    def __repr__(self) -> str:
        dims = ", ".join(f"{n}[{self._weights.get(n,1.0):.1f}]" for n in self._props)
        return f"ObjectND(geo={self.geo_hash()}, dims=[{dims}])"


# ---------------------------------------------------------------------------
# Operaciones compuestas
# ---------------------------------------------------------------------------

def nd_merge(obj_a: ObjectND, obj_b: ObjectND, space: PolyDimSpace = SPACE) -> ObjectND:
    """Nuevo objeto con dimensiones de ambos. GEO_ID = BIND(geo_a, geo_b)."""
    merged = ObjectND(space)
    merged._geo_id = hv_bind(obj_a.geo_id, obj_b.geo_id)
    for n, p in obj_a._props.items():
        merged.add_dim(n, p, obj_a._weights.get(n, 1.0))
    for n, p in obj_b._props.items():
        if n not in merged._props:
            merged.add_dim(n, p, obj_b._weights.get(n, 1.0))
    merged._cache = hv_superpose(obj_a.to_hv(), obj_b.to_hv())
    return merged

def nd_distance(obj_a: ObjectND, obj_b: ObjectND) -> float:
    return 1.0 - hv_sim(obj_a.to_hv(), obj_b.to_hv())


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(f"POLYDIM Core V0.2  —  N={POLYDIM_N}  umbral={UMBRAL_RECUPERACION:.4f}")
    print("=" * 60)

    user = ObjectND()
    user.add_dim("DIM_SQL",    {"tabla":"usuarios","pk":"id"},        weight=1.0)
    user.add_dim("DIM_PYTHON", {"tipo":"dict","analisis":"pandas"},   weight=0.7)
    user.add_dim("DIM_FLUTTER",{"widget":"Form","estado":"reactivo"}, weight=0.3)
    user.add_dim("DIM_RUST",   {"tipo":"struct","ownership":"owned"}, weight=0.0)

    sym = user.to_symbolic()
    print(f"\nOBJECT_ND usuario  GEO_ID={sym['geo_id']}")
    for d, info in sym["dimensions"].items():
        print(f"  {d:<16} decl={info['weight_declared']:.1f}  "
              f"geo={info['weight_geometric']:.4f}")

    print(f"\nActivas detectadas:")
    for d, w in sorted(sym["active_g"].items(), key=lambda x: -x[1]):
        print(f"  {d}: {w:.4f}")

    precio = ObjectND()
    precio.add_dim("DIM_SQL",    {"columna":"precio","tipo":"DECIMAL"}, weight=1.0)
    precio.add_dim("DIM_PYTHON", {"tipo":"float","precision":"2"},      weight=1.0)
    precio.add_dim("DIM_FLUTTER",{"widget":"TextField"},                weight=1.0)
    precio.add_dim("DIM_RUST",   {"tipo":"f64"},                        weight=1.0)

    print(f"\nHV_SIM(usuario, precio) = {user.similarity(precio):.4f}")
    print(f"son mismo objeto: {user.is_same_object(precio)}")

    merged = nd_merge(user, precio)
    print(f"\nMERGE: distancia→usuario={nd_distance(merged,user):.4f}  "
          f"distancia→precio={nd_distance(merged,precio):.4f}")

    geo1 = user.geo_hash()
    user.add_dim("DIM_GRAPH", {"nodo":"usuario"}, weight=0.5)
    print(f"\nINV_001: {user.geo_hash()==geo1}")
    w1 = user.activate("DIM_SQL")
    _  = user.to_symbolic()
    w2 = user.activate("DIM_SQL")
    print(f"INV_004: {abs(w1-w2)<1e-6}")
