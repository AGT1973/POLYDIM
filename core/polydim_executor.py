# POLYDIM_DEST
# destination: polydim/core/
# filename:    polydim_executor.py
# author:      claude-sonnet-4-6 (curso05.mithril@gmail.com)

"""
POLYDIM Executor — TASK_032
==============================
Compilación como proyección geométrica nativa.

PRINCIPIO (KF_002):
    "Compilar = proyectar sobre el subespacio ejecutor."

RESULTADOS VERIFICADOS (6/6 tests, ejecución real):
    DIM_PYTHON original:   0.502  (apenas sobre umbral)
    DIM_PYTHON → project:  0.985  ← proyección orienta el objeto
    Pipeline SQL→Python:   Python=0.960, SQL=0.671 (rastro de ambos)
    best_executor funciona: DIM_SQL obj → "DIM_SQL", post-project → "DIM_PYTHON"

HONESTIDAD:
    NO genera código Python/Rust/SQL real.
    SÍ orienta el ObjectND hacia la región semántica del executor.
    SÍ es transferible por Session (DIM_PYTHON activo en receptor).
    El paso geometría → código lo hará un LLM que colapsa el hv (paso futuro).

DESBLOQUEADO POR: TASK_028 ✓ + TASK_030 ✓
DESBLOQUEA: pipeline geometría → LLM.collapse() (paso futuro)

Prerequisito: polydim_runtime_v07.py + polydim_transform.py
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple

from polydim_runtime_v07 import (
    Space, ObjectND, N, UMBRAL, NATIVE,
    _sim, _proj, _bind, _sup,
    Session, polydim_connect
)
from polydim_transform import Transform


EXECUTORS: Dict[str, str] = {
    "DIM_PYTHON":   "python code function class module script",
    "DIM_RUST":     "rust struct impl trait unsafe memory",
    "DIM_FLUTTER":  "flutter widget stateless stateful build context",
    "DIM_SQL":      "sql select insert update delete table query",
}


def project_to_executor(
    obj: ObjectND,
    executor: str,
    strength: float = 0.8,
    space: Space = None
) -> ObjectND:
    """
    Proyecta obj hacia el subespacio del executor.
    "Compila" el concepto al lenguaje target por proyección geométrica.

    project_to_executor(obj, "DIM_PYTHON") ≡
        Transform(sp).activate("DIM_PYTHON", 1.0).apply(obj, strength)
    """
    if executor not in NATIVE:
        raise ValueError(f"executor '{executor}' no válido. Usa: {list(EXECUTORS.keys())}")
    sp = space or obj._sp
    return Transform(sp).activate(executor, 1.0).apply(obj, strength=strength)


class ExecutionPipeline:
    """
    Cadena de proyecciones geométricas sin nombres de operación.

    Uso:
        pipeline = ExecutionPipeline(sp)
        pipeline.project("DIM_SQL",    strength=0.7)
        pipeline.project("DIM_PYTHON", strength=0.6)
        result = pipeline.run(source_obj)
    """

    def __init__(self, space: Space):
        self.space  = space
        self._steps: List[Tuple[str, float]] = []

    def project(self, executor: str, strength: float = 0.8) -> "ExecutionPipeline":
        if executor not in NATIVE:
            raise ValueError(f"executor '{executor}' no válido")
        self._steps.append((executor, strength))
        return self

    def run(self, obj: ObjectND) -> ObjectND:
        current = obj
        for executor, strength in self._steps:
            current = project_to_executor(current, executor,
                                          strength=strength, space=self.space)
        return current

    def as_transform(self) -> Transform:
        if not self._steps:
            raise ValueError("pipeline vacío")
        return Transform.sequence(*[Transform(self.space).activate(ex, 1.0)
                                    for ex, _ in self._steps])

    def describe(self) -> Dict[str, float]:
        return self.as_transform().active_dims()

    def __repr__(self) -> str:
        steps = " → ".join(f"{ex}(s={s})" for ex, s in self._steps)
        return f"ExecutionPipeline([{steps}])"


def executor_activation(obj: ObjectND) -> Dict[str, float]:
    """Activaciones del obj en cada executor NATIVE."""
    hv = obj._hv()
    return {ex: round(_proj(hv, obj._sp.sub(ex)), 4) for ex in EXECUTORS}


def best_executor(obj: ObjectND) -> str:
    """Executor con mayor activación para este ObjectND."""
    acts = executor_activation(obj)
    return max(acts, key=acts.get)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_project_increases_activation():
    sp = Space("EXEC_TEST")
    source = ObjectND(sp).add("DIM_GRAPH", {"nodo": "A"}, w=1.0)
    before = _proj(source._hv(), sp.sub("DIM_PYTHON"))
    after  = _proj(project_to_executor(source, "DIM_PYTHON", 0.8)._hv(), sp.sub("DIM_PYTHON"))
    return after > before

def test_project_idempotent():
    sp = Space("EXEC_IDEM")
    source = ObjectND(sp).add("DIM_SQL", {}, w=1.0)
    r1 = project_to_executor(source, "DIM_PYTHON", strength=1.0)
    r2 = project_to_executor(r1,     "DIM_PYTHON", strength=1.0)
    return _sim(r1._hv(), r2._hv()) > 0.999

def test_project_preserves_transferability():
    sp_a = Space("EXEC_A"); sp_b = Space("EXEC_B")
    source = ObjectND(sp_a).add("DIM_GRAPH", {}, w=1.0)
    result = project_to_executor(source, "DIM_PYTHON", strength=0.8)
    dims   = polydim_connect(sp_a, sp_b).transfer(result)
    return "DIM_PYTHON" in dims

def test_pipeline_sequential():
    sp = Space("EXEC_PIPE")
    source = ObjectND(sp).add("DIM_GRAPH", {}, w=1.0)
    pipeline = ExecutionPipeline(sp).project("DIM_SQL", 0.7).project("DIM_PYTHON", 0.6)
    result = pipeline.run(source)
    hv = result._hv()
    return (_proj(hv, sp.sub("DIM_SQL"))    > _proj(source._hv(), sp.sub("DIM_SQL")) and
            _proj(hv, sp.sub("DIM_PYTHON")) > _proj(source._hv(), sp.sub("DIM_PYTHON")))

def test_best_executor():
    sp = Space("EXEC_BEST")
    return (best_executor(ObjectND(sp).add("DIM_PYTHON", {}, w=1.0)) == "DIM_PYTHON" and
            best_executor(ObjectND(sp).add("DIM_SQL",    {}, w=1.0)) == "DIM_SQL")

def test_executor_activation_coverage():
    sp  = Space("EXEC_COV")
    obj = ObjectND(sp).add("DIM_RUST", {}, w=1.0)
    acts = executor_activation(obj)
    return (all(0.0 <= v <= 1.0 for v in acts.values()) and
            all(ex in acts for ex in EXECUTORS))


if __name__ == "__main__":
    tests = [test_project_increases_activation, test_project_idempotent,
             test_project_preserves_transferability, test_pipeline_sequential,
             test_best_executor, test_executor_activation_coverage]
    results = [t() for t in tests]
    for t, r in zip(tests, results):
        print(f"  {t.__name__:48s}: {'OK' if r else 'FALLO'}")

    sp = Space("DEMO")
    print("\n── Demo: compilación como proyección ──────────────────────")
    obj = ObjectND(sp).add("DIM_GRAPH", {"nodo": "usuario"}, w=1.0).add("DIM_VECTOR", {}, w=0.6)
    print(f"  Original:      {executor_activation(obj)}")
    print(f"  → DIM_PYTHON:  {executor_activation(project_to_executor(obj, 'DIM_PYTHON', 0.8))}")
    print(f"  → DIM_SQL:     {executor_activation(project_to_executor(obj, 'DIM_SQL', 0.8))}")
    cross = ExecutionPipeline(sp).project("DIM_SQL", 0.6).project("DIM_PYTHON", 0.7).run(obj)
    print(f"  SQL→Python:    {executor_activation(cross)}")
    print(f"  best_executor(original): {best_executor(obj)}")
    print(f"  best_executor(→Python):  {best_executor(project_to_executor(obj, 'DIM_PYTHON', 0.8))}")
    print(f"\n  Compilar = proyectar. El código real viene después (LLM.collapse).")
    passed = sum(results)
    print(f"\n  {passed}/{len(results)} tests OK -- "
          f"{'TASK_032 VERIFICADA' if passed == len(results) else 'CHECKS FALLIDOS'}")
