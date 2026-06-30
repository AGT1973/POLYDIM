# POLYDIM_DEST
# destination: polydim/core/
# filename:    polydim_debug.py
# author:      polydim.ai.lenguage@gmail.com

"""
POLYDIM — Debugger / Capa S explícita (TASK_031)
=================================================
POLYDIM_BASES_V1.md AP_001: "MODO_S is the debugger, NOT the language."

FUNCIONES:
  activation_map(obj, space)          → Dict[str,float] espectro completo
  activation_bar(obj, space, width)   → str ASCII con barras
  explain(obj, space, verbose)        → str descripción textual
  compare(obj_a, obj_b, space)        → str similitud global + por dim
  describe_transform(T, space)        → str efecto geométrico de TransformND
  session_report(session)             → str estado de Session
  object_report(obj, space)           → dict JSON-friendly

Tests: 7/7 OK (2026-06-20)
Autor: polydim.ai.lenguage@gmail.com
V1.0 — 2026-06-20 — TASK_031
Base: polydim_runtime_v08.py
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import numpy as np
from polydim_runtime_v08 import Space, ObjectND, Session, N, NATIVE, UMBRAL, _proj, _sim


def activation_map(obj: ObjectND, space: Space) -> Dict[str, float]:
    hv = obj._hv()
    dims = list(dict.fromkeys(NATIVE + list(obj._props.keys())))
    scores = {d: round(float(_proj(hv, space.sub(d))), 4) for d in dims}
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


def activation_bar(obj: ObjectND, space: Space, width: int = 36, show_umbral: bool = True) -> str:
    am = activation_map(obj, space)
    lines = [f"ObjectND({obj.geo_id}) — activation map:"]
    for dim, score in am.items():
        filled = int(score * width)
        bar = "█" * filled + "░" * (width - filled)
        mark = " ✓" if score > UMBRAL else ""
        label = dim.replace("DIM_", "").ljust(10)
        lines.append(f"  {label} [{bar}] {score:.4f}{mark}")
    if show_umbral:
        lines.append(f"  {'─'*10}  umbral={UMBRAL:.4f}")
    return "\n".join(lines)


def explain(obj: ObjectND, space: Space, verbose: bool = False) -> str:
    lines = ["[POLYDIM ObjectND]",
             f"  geo_id : {obj.geo_id}",
             f"  N      : {N}"]
    if obj._props:
        lines.append(f"  dims declaradas ({len(obj._props)}):")
        for dim, props in obj._props.items():
            lines.append(f"    {dim} [w={obj._w.get(dim,1.0):.2f}] {_fmt(props)}")
    else:
        lines.append("  dims declaradas: ninguna")
    am = activation_map(obj, space)
    activas = {d: s for d, s in am.items() if s > UMBRAL}
    lines.append(f"  dims activas ({len(activas)}, umbral={UMBRAL:.3f}):")
    for dim, score in activas.items():
        mark = "★" if dim in obj._props else "·"
        lines.append(f"    {mark} {dim}: {score:.4f}")
    latentes = [d for d in obj._props if d not in activas]
    if latentes:
        lines.append(f"  latentes ({len(latentes)} bajo umbral):")
        for d in latentes:
            lines.append(f"    · {d}: {am.get(d, 0.0):.4f}")
    if verbose:
        lines.append("  espectro completo:")
        for dim, score in am.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            lines.append(f"    {dim.replace('DIM_',''):10s} [{bar}] {score:.4f}")
    return "\n".join(lines)


def compare(obj_a: ObjectND, obj_b: ObjectND,
            space: Space, label_a: str = "A", label_b: str = "B") -> str:
    hv_a, hv_b = obj_a._hv(), obj_b._hv()
    global_sim = float(_sim(hv_a, hv_b))
    lines = [f"[POLYDIM Compare: {label_a} vs {label_b}]",
             f"  geo_id {label_a}: {obj_a.geo_id}  {label_b}: {obj_b.geo_id}",
             f"  similitud global: {global_sim:.4f}",
             "  por dimensión nativa:"]
    dim_scores = []
    for dim in NATIVE:
        sa = float(_proj(hv_a, space.sub(dim)))
        sb = float(_proj(hv_b, space.sub(dim)))
        dim_scores.append((dim, sa, sb, sb - sa))
    dim_scores.sort(key=lambda x: -abs(x[3]))
    for dim, sa, sb, delta in dim_scores:
        arrow = "↑" if delta > 0.02 else ("↓" if delta < -0.02 else "≈")
        lines.append(f"    {dim.replace('DIM_',''):10s}  {label_a}:{sa:.3f}  {label_b}:{sb:.3f}  {arrow}{abs(delta):.3f}")
    return "\n".join(lines)


def describe_transform(T: Any, space: Space, probes: Optional[List[str]] = None) -> str:
    dims = probes or NATIVE
    lines = [f"[POLYDIM describe_transform: {T}]",
             "  Efecto sobre subespacios nativos:"]
    for dim in dims:
        probe = ObjectND(space)
        probe._geo = space.sub(dim).copy()
        probe._cache = space.sub(dim).copy()
        try:
            t_hv = T.apply(probe)._cache
            change = float(_proj(t_hv, space.sub(dim))) - float(_proj(space.sub(dim), space.sub(dim)))
            best = max(NATIVE, key=lambda d: float(_proj(t_hv, space.sub(d))))
            best_score = float(_proj(t_hv, space.sub(best)))
            arrow = "↑" if change > 0.01 else ("↓" if change < -0.01 else "≈")
            lines.append(f"    {dim.replace('DIM_',''):10s}  {arrow}  → activates {best.replace('DIM_','')} [{best_score:.4f}]")
        except Exception as e:
            lines.append(f"    {dim.replace('DIM_',''):10s}  ERROR: {e}")
    return "\n".join(lines)


def session_report(session: Session) -> str:
    info = session.info
    lines = [f"[POLYDIM Session: {session.name}]",
             f"  session_id : {info.get('session_id','N/A')}",
             f"  estado     : {info.get('state','N/A')}",
             f"  modo       : {info.get('mode','N/A')}"]
    score = info.get('align_score')
    if score is not None:
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        lines.append(f"  align      : [{bar}] {score:.4f} {'✓ OK' if score >= 0.85 else '✗ bajo'}")
    else:
        lines.append("  align      : no alineado")
    perms = info.get('meta_perms', [])
    lines.append(f"  meta_perms : {', '.join(perms) if perms else 'ninguno'}")
    return "\n".join(lines)


def object_report(obj: ObjectND, space: Space) -> dict:
    am = activation_map(obj, space)
    return {
        "geo_id": obj.geo_id, "N": N, "umbral": round(UMBRAL, 4),
        "dims_declaradas": {d: {"w": obj._w.get(d, 1.0), "props": p} for d, p in obj._props.items()},
        "activation_map": am,
        "dims_activas": {d: s for d, s in am.items() if s > UMBRAL},
        "dims_latentes": [d for d in obj._props if am.get(d, 0.0) <= UMBRAL],
    }


def _fmt(props: dict) -> str:
    if not props: return "{}"
    items = [f"{k}={repr(v)}" for k, v in list(props.items())[:3]]
    return "{" + ", ".join(items) + (", ..." if len(props) > 3 else "") + "}"


# Tests 7/7 OK

def test_activation_map_complete():
    sp = Space("T"); obj = ObjectND(sp).add("DIM_SQL", {"t":"u"}, w=1.0)
    am = activation_map(obj, sp)
    return set(NATIVE).issubset(set(am.keys())) and all(0.0 <= v <= 1.0 for v in am.values())

def test_activation_map_sorted():
    sp = Space("T"); obj = ObjectND(sp).add("DIM_SQL", {}, w=1.0)
    scores = list(activation_map(obj, sp).values())
    return scores == sorted(scores, reverse=True)

def test_explain_contains_key_info():
    sp = Space("T")
    obj = ObjectND(sp).add("DIM_SQL", {"tabla":"usuarios"}, w=1.0).add("DIM_PYTHON", {"fn":"q"}, w=0.7)
    txt = explain(obj, sp)
    return obj.geo_id in txt and "DIM_SQL" in txt and "DIM_PYTHON" in txt

def test_activation_bar_format():
    sp = Space("T"); obj = ObjectND(sp).add("DIM_GRAPH", {}, w=1.0)
    bar = activation_bar(obj, sp)
    return "█" in bar and "░" in bar and "GRAPH" in bar

def test_compare_identical_objects():
    sp = Space("T"); obj = ObjectND(sp).add("DIM_SQL", {"t":"u"}, w=1.0)
    txt = compare(obj, obj, sp, "X", "X")
    return "1.0000" in txt or "0.999" in txt

def test_compare_different_objects():
    sp = Space("T")
    txt = compare(ObjectND(sp).add("DIM_SQL",{},w=1.0), ObjectND(sp).add("DIM_PYTHON",{},w=1.0), sp)
    return "↑" in txt or "↓" in txt

def test_session_report():
    from polydim_runtime_v08 import Session as S
    sa = S(Space("A"), "IA"); sb = S(Space("B"), "P"); sa.connect(sb)
    txt = session_report(sa)
    return "READY" in txt or "MODO_H" in txt or "align" in txt


if __name__ == "__main__":
    tests = [test_activation_map_complete, test_activation_map_sorted,
             test_explain_contains_key_info, test_activation_bar_format,
             test_compare_identical_objects, test_compare_different_objects,
             test_session_report]
    print("── polydim_debug.py ─────────────────────────────────────")
    results = [t() for t in tests]
    for t, r in zip(tests, results):
        print(f"  {t.__name__:45s}: {'OK' if r else 'FALLO'}")
    print(f"\n  {sum(results)}/{len(results)} tests OK")
    if all(results): print("  TASK_031 VERIFICADA — polydim_debug.py OK")
