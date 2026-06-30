# POLYDIM_DEST
# destination: polydim/core/
# filename:    polydim_distributed_consistency.py
# author:      ai.mpat.agt@gmail.com

"""
POLYDIM Distributed Consistency — V0.1
========================================
Resuelve PEND_003: dado un ObjectND transmitido a N IAs distintas,
verificar que las dimensiones activas detectadas son consistentes
entre sí (no que sean idénticas, sino que coincidan en el núcleo semántico).

CONCEPTO:
  Cada IA recibe el mismo objeto via Session.send/receive.
  Por diferencias en el Space de cada IA (distintas semillas, backends),
  las dims detectadas pueden diferir levemente.
  ConsistencyChecker mide el grado de acuerdo y alerta si cae por debajo
  del umbral configurado.

MÉTRICAS:
  - Jaccard(A, B):   |A ∩ B| / |A ∪ B|  en [0, 1]
  - Consenso(k, N):  dims presentes en al menos k de N receptores
  - ConsistencyScore: Jaccard promedio entre todos los pares de receptores

UMBRALES:
  UMBRAL_CONSISTENCIA = 0.5  (Jaccard ≥ 0.5 = consistente)
  CONSENSO_K_DEFAULT  = 0.6  (dim válida si ≥ 60% de IAs la detectan)

Autor:   ai.mpat.agt@gmail.com
Versión: V0.1 — 2026-06-24
Spec:    PEND_003
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Import del runtime
# ---------------------------------------------------------------------------
try:
    from polydim_runtime_v04 import ObjectND, Session, Space, Packet, NATIVE
except ImportError:
    from polydim_runtime_v03 import ObjectND, Session, Space, Packet, NATIVE  # type: ignore

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

UMBRAL_CONSISTENCIA: float = 0.5
CONSENSO_K_DEFAULT: float = 0.6


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------

@dataclass
class ReceptionResult:
    """Resultado de recepción de un objeto por una IA."""
    ia_name: str
    dims_activas: Dict[str, float]   # {dim: activacion}
    session_id: Optional[str] = None

    @property
    def dims_set(self) -> Set[str]:
        return set(self.dims_activas.keys())


@dataclass
class ConsistencyReport:
    """Reporte completo de consistencia para una transmisión multi-IA."""
    objeto_geo_id: str
    resultados: List[ReceptionResult]
    jaccard_matrix: Dict[Tuple[str, str], float] = field(default_factory=dict)
    consistency_score: float = 0.0
    dims_consenso: Dict[str, float] = field(default_factory=dict)  # {dim: fraccion_IAs}
    dims_nucleo: Set[str] = field(default_factory=set)   # dims en >= CONSENSO_K_DEFAULT IAs
    es_consistente: bool = False

    def summary(self) -> str:
        lines = [
            f"ConsistencyReport  geo_id={self.objeto_geo_id}",
            f"IAs: {[r.ia_name for r in self.resultados]}",
            f"ConsistencyScore: {self.consistency_score:.4f}  "
            f"Consistente: {self.es_consistente}",
            f"Núcleo semántico ({int(CONSENSO_K_DEFAULT*100)}%+): {sorted(self.dims_nucleo)}",
            "Jaccard por par:",
        ]
        for (a, b), j in sorted(self.jaccard_matrix.items()):
            lines.append(f"  {a} ↔ {b}: {j:.4f}")
        lines.append("Consenso por dim:")
        for d, f_ in sorted(self.dims_consenso.items(), key=lambda x: -x[1]):
            lines.append(f"  {d:<18} {f_:.2f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Funciones principales
# ---------------------------------------------------------------------------

def jaccard(a: Set[str], b: Set[str]) -> float:
    """Similitud Jaccard entre dos conjuntos de dimensiones."""
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u > 0 else 0.0


def consistency_score(resultados: List[ReceptionResult]) -> float:
    """
    Jaccard promedio entre todos los pares de receptores.

    Con N receptores calcula C(N,2) pares.
    Para N=1 retorna 1.0 (trivialmente consistente).
    """
    if len(resultados) < 2:
        return 1.0
    scores = [
        jaccard(a.dims_set, b.dims_set)
        for a, b in itertools.combinations(resultados, 2)
    ]
    return round(sum(scores) / len(scores), 4)


def dims_consenso(
    resultados: List[ReceptionResult],
    k: float = CONSENSO_K_DEFAULT,
) -> Tuple[Dict[str, float], Set[str]]:
    """
    Calcula qué fracción de IAs detectó cada dimensión.

    Returns:
        (fraccion_por_dim, dims_nucleo)
        fraccion_por_dim: {dim: fraccion} para todas las dims detectadas por alguien
        dims_nucleo:      dims detectadas por >= k fracción de IAs
    """
    if not resultados:
        return {}, set()
    n = len(resultados)
    counter: Dict[str, int] = {}
    for r in resultados:
        for d in r.dims_set:
            counter[d] = counter.get(d, 0) + 1
    fracciones = {d: round(c / n, 4) for d, c in counter.items()}
    nucleo = {d for d, f in fracciones.items() if f >= k}
    return fracciones, nucleo


def check_consistency(
    obj: ObjectND,
    sesiones: List[Session],
    emisor: Session,
    k: float = CONSENSO_K_DEFAULT,
    umbral: float = UMBRAL_CONSISTENCIA,
) -> ConsistencyReport:
    """
    Transmite obj desde emisor a todas las sesiones y evalúa consistencia.

    El emisor llama a .send(obj) una vez por sesión receptora.
    Cada sesión receptora llama a .receive(pkt).

    Args:
        obj:      ObjectND a transmitir (debe usar el mismo Space que emisor).
        sesiones: Lista de Sessions receptoras, ya conectadas con emisor.
        emisor:   Session emisora.
        k:        Fracción mínima de IAs para incluir dim en el núcleo (default 0.6).
        umbral:   Umbral de Jaccard para declarar consistencia (default 0.5).

    Returns:
        ConsistencyReport completo.
    """
    resultados: List[ReceptionResult] = []
    for ses in sesiones:
        pkt = emisor.send(obj)
        dims = ses.receive(pkt)
        resultados.append(ReceptionResult(
            ia_name=ses.name,
            dims_activas=dims,
            session_id=ses.session_id,
        ))

    # Matriz Jaccard
    jac_matrix: Dict[Tuple[str, str], float] = {}
    for a, b in itertools.combinations(resultados, 2):
        jac_matrix[(a.ia_name, b.ia_name)] = jaccard(a.dims_set, b.dims_set)

    score = consistency_score(resultados)
    fracciones, nucleo = dims_consenso(resultados, k=k)

    return ConsistencyReport(
        objeto_geo_id=obj.geo_id,
        resultados=resultados,
        jaccard_matrix=jac_matrix,
        consistency_score=score,
        dims_consenso=fracciones,
        dims_nucleo=nucleo,
        es_consistente=(score >= umbral),
    )


def broadcast_and_check(
    obj: ObjectND,
    ia_spaces: List[Tuple[str, Space]],
    emisor_space: Space,
    emisor_name: str = "IA_EMISOR",
    k: float = CONSENSO_K_DEFAULT,
) -> ConsistencyReport:
    """
    API de alto nivel: crea las sesiones, conecta, transmite y verifica.

    Args:
        obj:           ObjectND a distribuir (mismo Space que emisor_space).
        ia_spaces:     Lista de (nombre, Space) para las IAs receptoras.
        emisor_space:  Space del emisor.
        emisor_name:   Nombre del emisor.
        k:             Umbral de consenso.

    Returns:
        ConsistencyReport.

    Ejemplo:
        sp_a = Space("IA_A")
        obj  = ObjectND(sp_a).add("DIM_SQL", {"tabla": "users"}, w=1.0)
        report = broadcast_and_check(
            obj, [("IA_B", Space("IA_B")), ("IA_C", Space("IA_C"))], sp_a
        )
        print(report.summary())
    """
    emisor = Session(emisor_space, emisor_name)
    receptores: List[Session] = []
    for name, sp in ia_spaces:
        r = Session(sp, name)
        # Clonar emisor para cada receptor (Session es stateful)
        e = Session(emisor_space, emisor_name)
        e.connect(r)
        receptores.append(r)

    # Para broadcast usamos un emisor fresco por receptor
    resultados: List[ReceptionResult] = []
    for ses in receptores:
        e = Session(emisor_space, emisor_name)
        e.connect(ses)
        pkt = e.send(obj)
        dims = ses.receive(pkt)
        resultados.append(ReceptionResult(
            ia_name=ses.name,
            dims_activas=dims,
            session_id=ses.session_id,
        ))

    jac_matrix: Dict[Tuple[str, str], float] = {
        (a.ia_name, b.ia_name): jaccard(a.dims_set, b.dims_set)
        for a, b in itertools.combinations(resultados, 2)
    }
    score = consistency_score(resultados)
    fracciones, nucleo = dims_consenso(resultados, k=k)

    return ConsistencyReport(
        objeto_geo_id=obj.geo_id,
        resultados=resultados,
        jaccard_matrix=jac_matrix,
        consistency_score=score,
        dims_consenso=fracciones,
        dims_nucleo=nucleo,
        es_consistente=(score >= UMBRAL_CONSISTENCIA),
    )


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "ReceptionResult",
    "ConsistencyReport",
    "jaccard",
    "consistency_score",
    "dims_consenso",
    "check_consistency",
    "broadcast_and_check",
    "UMBRAL_CONSISTENCIA",
    "CONSENSO_K_DEFAULT",
]
