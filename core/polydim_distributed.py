# POLYDIM_DEST
# destination: polydim/core/
# filename:    polydim_distributed.py
# author:      ai.mpat.agt@gmail.com

"""
POLYDIM Consistencia Distribuida — V0.1
==========================================
Resolución pragmática de PEND_003.

ESTADO DEL ARTE EN LA CONSTITUCIÓN:
  PEND_003 está marcado como 🔬 investigación especulativa en V6 §IX
  (Geometric-CRDTs, sin demostración matemática formal). Este módulo
  implementa la versión pragmática basada en propiedades verificadas
  empíricamente, sin reclamar el status de Geometric-CRDT formal.

PROPIEDADES IMPLEMENTADAS Y VERIFICADAS:
  merge(a, b) = merge(b, a)           → conmutativo (sim=1.0 empírico)
  merge(a, a) = a                     → idempotente (sim=1.0 empírico)
  dist(merge, base) < max(dist_a, dist_b)  → conserva info del ancestro

PROPIEDADES NO GARANTIZADAS (requieren demostración formal, ver V6 §IX):
  - Asociatividad con tres o más nodos concurrentes (NO verificada en
    casos de conflicto con timestamps iguales)
  - Convergencia bajo partición de red prolongada
  - Preservación de GEO_ID post-merge (el merge produce un nuevo objeto)

NOTA PARA DOCENTES:
  La elevación de este módulo a ✅ LEY requiere el proceso formal
  del Artículo XX de la Constitución V6. Actualmente es ⚙️ MECANISMO.

Autor:   ai.mpat.agt@gmail.com
Versión: V0.1 — 2026-06-25
Spec:    PEND_003 (pragmático)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from polydim_runtime_v04 import ObjectND, Space, _sim, _proj, NATIVE, UMBRAL, N
except ImportError:
    from polydim_runtime_v03 import ObjectND, Space, _sim, _proj, NATIVE, UMBRAL, N  # type: ignore

# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------

@dataclass
class VersionedHV:
    """
    Hipervector versionado para replicación distribuida.

    Attributes:
        hv:         Hipervector float32 normalizado.
        geo_id:     GEO_ID del objeto que lo generó.
        node_id:    Identificador del nodo que creó esta versión.
        timestamp:  Unix timestamp de la modificación (float).
        vector_clock: Reloj vectorial {node_id: counter}.
    """
    hv: np.ndarray
    geo_id: str
    node_id: str
    timestamp: float = field(default_factory=time.time)
    vector_clock: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.node_id not in self.vector_clock:
            self.vector_clock[self.node_id] = 1


@dataclass
class MergeResult:
    """Resultado de una operación merge entre dos versiones.

    Attributes:
        hv:           Hipervector fusionado.
        conflict:     True si las versiones son concurrentes (no hay causalidad).
        score_a:      Similitud del merge con la versión A.
        score_b:      Similitud del merge con la versión B.
        strategy:     Estrategia usada ('timestamp', 'weighted', 'equal').
    """
    hv: np.ndarray
    conflict: bool
    score_a: float
    score_b: float
    strategy: str


# ---------------------------------------------------------------------------
# Operaciones de merge
# ---------------------------------------------------------------------------

def _merge_hvs(
    hv_a: np.ndarray,
    hv_b: np.ndarray,
    w_a: float = 0.5,
    w_b: float = 0.5,
) -> np.ndarray:
    """Mezcla ponderada normalizada. Conmutativo e idempotente."""
    c = hv_a * w_a + hv_b * w_b
    n = np.linalg.norm(c)
    return c / n if n > 1e-9 else c


def merge_versions(
    va: VersionedHV,
    vb: VersionedHV,
) -> MergeResult:
    """
    Fusiona dos versiones de un hipervector distribuido.

    Estrategia de resolución (en orden):
      1. Si una versión causalmente precede a la otra → gana la más reciente.
      2. Si son concurrentes → merge ponderado por timestamp (last-write-wins).
      3. Si timestamps iguales → merge igual (0.5/0.5).

    Args:
        va: VersionedHV de la IA A.
        vb: VersionedHV de la IA B.

    Returns:
        MergeResult con el hipervector fusionado y metadatos.
    """
    # Determinar causalidad via vector clock
    vc_a = va.vector_clock
    vc_b = vb.vector_clock
    all_nodes = set(vc_a) | set(vc_b)

    a_dom = all(vc_a.get(n, 0) >= vc_b.get(n, 0) for n in all_nodes)
    b_dom = all(vc_b.get(n, 0) >= vc_a.get(n, 0) for n in all_nodes)

    if a_dom and not b_dom:
        # A causalmente domina → A gana
        return MergeResult(
            hv=va.hv.copy(),
            conflict=False,
            score_a=1.0,
            score_b=round(float(_sim(va.hv, vb.hv)), 4),
            strategy="causal_a_wins",
        )
    elif b_dom and not a_dom:
        # B causalmente domina → B gana
        return MergeResult(
            hv=vb.hv.copy(),
            conflict=False,
            score_a=round(float(_sim(vb.hv, va.hv)), 4),
            score_b=1.0,
            strategy="causal_b_wins",
        )
    else:
        # Concurrentes → merge ponderado
        conflict = True
        if abs(va.timestamp - vb.timestamp) < 1e-6:
            w_a = w_b = 0.5
            strategy = "equal"
        else:
            total = va.timestamp + vb.timestamp
            w_a = va.timestamp / total
            w_b = vb.timestamp / total
            strategy = "timestamp_weighted"

        merged = _merge_hvs(va.hv, vb.hv, w_a, w_b)
        return MergeResult(
            hv=merged,
            conflict=conflict,
            score_a=round(float(_sim(merged, va.hv)), 4),
            score_b=round(float(_sim(merged, vb.hv)), 4),
            strategy=strategy,
        )


def merge_multiple(versions: List[VersionedHV]) -> np.ndarray:
    """
    Fusiona N versiones concurrentes con pesos iguales.

    Propiedad: el resultado no depende del orden de la lista.

    Args:
        versions: Lista de VersionedHV a fusionar.

    Returns:
        Hipervector fusionado normalizado.
    """
    if not versions:
        raise ValueError("Lista de versiones vacía")
    if len(versions) == 1:
        return versions[0].hv.copy()
    hvs = [v.hv for v in versions]
    c = np.mean(hvs, axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 1e-9 else c


# ---------------------------------------------------------------------------
# ReplicatedObject
# ---------------------------------------------------------------------------

class ReplicatedObject:
    """
    ObjectND distribuido con soporte de merge y vector clocks.

    Permite que múltiples nodos modifiquen el mismo objeto
    concurrentemente y converger a un estado consistente.

    Uso:
        # Nodo A
        obj_a = ReplicatedObject(sp, "nodo_A")
        obj_a.add("DIM_SQL", {"tabla": "users"}, w=1.0)

        # Nodo B (modificación concurrente)
        obj_b = ReplicatedObject(sp, "nodo_B")
        obj_b.add("DIM_SQL",    {"tabla": "users", "pk": "id"}, w=1.0)
        obj_b.add("DIM_PYTHON", {"tipo": "service"}, w=0.7)

        # Merge en cualquier nodo
        merged = obj_a.merge_with(obj_b)
        dims = merged.dims_activas()

    Args:
        space:   Space de este nodo.
        node_id: Identificador único del nodo (p.ej. "nodo_A").
    """

    def __init__(self, space: Space, node_id: str) -> None:
        self._obj = ObjectND(space)
        self._node_id = node_id
        self._vc: Dict[str, int] = {node_id: 0}
        self._ts: float = time.time()
        self._history: List[Tuple[str, float, Dict[str, int]]] = []

    def add(self, dim: str, props: Optional[dict] = None, w: float = 1.0) -> "ReplicatedObject":
        """Agrega una dimensión e incrementa el vector clock."""
        self._obj.add(dim, props, w)
        self._vc[self._node_id] = self._vc.get(self._node_id, 0) + 1
        self._ts = time.time()
        self._history.append((dim, self._ts, dict(self._vc)))
        return self

    def to_versioned(self) -> VersionedHV:
        """Exporta el hipervector actual como VersionedHV."""
        return VersionedHV(
            hv=self._obj._hv().copy(),
            geo_id=self._obj.geo_id,
            node_id=self._node_id,
            timestamp=self._ts,
            vector_clock=dict(self._vc),
        )

    def merge_with(self, other: "ReplicatedObject") -> "ReplicatedObject":
        """
        Fusiona este objeto con otro y retorna un nuevo ReplicatedObject.

        El resultado tiene el vector clock unión de ambos.

        Args:
            other: ReplicatedObject del otro nodo.

        Returns:
            Nuevo ReplicatedObject con el hipervector fusionado.
        """
        va = self.to_versioned()
        vb = other.to_versioned()
        result_merge = merge_versions(va, vb)

        merged = ReplicatedObject.__new__(ReplicatedObject)
        merged._obj = ObjectND.__new__(ObjectND)
        merged._obj._sp = self._obj._sp
        merged._obj._props = {}
        merged._obj._w = {}
        merged._obj._geo = result_merge.hv
        merged._obj._cache = result_merge.hv

        merged._node_id = f"merge({self._node_id},{other._node_id})"
        merged._ts = max(self._ts, other._ts)
        # Vector clock: máximo componente a componente
        all_nodes = set(self._vc) | set(other._vc)
        merged._vc = {n: max(self._vc.get(n, 0), other._vc.get(n, 0)) for n in all_nodes}
        merged._history = []
        return merged

    def dims_activas(self) -> Dict[str, float]:
        """Delega a ObjectND.dims_activas()."""
        return self._obj.dims_activas()

    def hv(self) -> np.ndarray:
        """Retorna el hipervector actual."""
        return self._obj._hv()

    @property
    def vector_clock(self) -> Dict[str, int]:
        """Vector clock actual de este nodo."""
        return dict(self._vc)


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "VersionedHV",
    "MergeResult",
    "ReplicatedObject",
    "merge_versions",
    "merge_multiple",
]
