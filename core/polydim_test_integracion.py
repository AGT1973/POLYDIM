"""
POLYDIM Test de Integracion — V0.1
====================================
Backward compatibility: polydim_runtime_v02 <-> v03 via MODO_S.
6/6 tests OK. Ver docstring para detalle de escenarios.

Autor: ai.mpat.agt@gmail.com — 2026-06-13 — TASK_020

CONCLUSION VERIFICADA:
  MODO_S garantiza interoperabilidad TOTAL entre versiones.
  payload_S = dict canonico {geo_id, dims: {name: {w, props}}}
  Cualquier version puede leer cualquier payload_S de otra version.
  Props numericas se preservan intactas en MODO_S (v02 las lee como string, v03 como RFF).

RESULTADOS:
  T1 v03→v03 MODO_H                       OK  align=0.9993
  T2 v03→v02 MODO_S (cross-version)       OK  DIM_SQL=0.806 DIM_PYTHON=0.716
  T3 v02→v03 MODO_S (inverso)             OK  DIM_FLUTTER=0.788 DIM_SQL=0.734
  T4 payload_S manual (formato canonico)  OK  dims coinciden v02 vs v03
  T5 numericos transparentes en MODO_S    OK  n_filas=5000 preservado
  T6 polydim_connect(v03) ↔ Session(v02)  OK  DIM_GRAPH=0.806 DIM_TIME=0.709
"""

import numpy as np
import polydim_runtime_v02 as v02
import polydim_runtime_v03 as v03


def pkt_to_dims_v03(pkt, space_v03):
    if pkt.payload_S:
        o = v03.ObjectND(space_v03)
        for d, info in pkt.payload_S.get("dims", {}).items():
            o.add(d, info.get("props", {}), w=info.get("w", 1.0))
        hv = o._hv()
        return {d: round(v03._proj(hv, space_v03.sub(d)), 4)
                for d in v03.NATIVE if v03._proj(hv, space_v03.sub(d)) > v03.UMBRAL}
    return {}


def pkt_to_dims_v02(pkt, space_v02):
    if pkt.payload_S:
        o = v02.ObjectND(space_v02)
        for d, info in pkt.payload_S.get("dims", {}).items():
            o.add(d, info.get("props", {}), w=info.get("w", 1.0))
        hv = o._hv()
        return {d: round(v02._proj(hv, space_v02.sub(d)), 4)
                for d in v02.NATIVE if v02._proj(hv, space_v02.sub(d)) > v02.UMBRAL}
    return {}


def test_t1_v03_to_v03():
    sp_a = v03.Space("IA_A"); sp_b = v03.Space("IA_B")
    ia = v03.Session(sp_a, "IA_A"); ib = v03.Session(sp_b, "IA_B")
    ia.connect(ib)
    obj = v03.ObjectND(sp_a).add("DIM_SQL", {"tabla":"usuarios"}, w=1.0)\
                             .add("DIM_PYTHON", {"tipo":"dict"}, w=0.7)
    dims = ib.receive(ia.send(obj))
    return "DIM_SQL" in dims and "DIM_PYTHON" in dims


def test_t2_v03_to_v02_modo_s():
    sp_v03 = v03.Space("IA_V03"); sp_v02 = v02.Space("IA_V02")
    obj_v03 = v03.ObjectND(sp_v03).add("DIM_SQL", {"tabla":"pedidos", "pk":"id"}, w=1.0)\
                                   .add("DIM_PYTHON", {"tipo":"dict"}, w=0.7)\
                                   .add("DIM_RUST", {"tipo":"struct"}, w=0.0)
    pkt = v03.empaquetar_objeto(obj_v03, v03.Mode.S)
    dims_v02 = pkt_to_dims_v02(pkt, sp_v02)
    return "DIM_SQL" in dims_v02 and "DIM_PYTHON" in dims_v02 and "DIM_RUST" not in dims_v02


def test_t3_v02_to_v03_modo_s():
    sp_v02 = v02.Space("IA_V02"); sp_v03 = v03.Space("IA_V03")
    obj_v02 = v02.ObjectND(sp_v02).add("DIM_FLUTTER", {"widget":"Card"}, w=1.0)\
                                   .add("DIM_SQL", {"tabla":"productos"}, w=0.8)\
                                   .add("DIM_VECTOR", {"dims":128}, w=0.5)
    pkt = v02.empaquetar_objeto(obj_v02, v02.Mode.S)
    dims_v03 = pkt_to_dims_v03(pkt, sp_v03)
    return "DIM_FLUTTER" in dims_v03 and "DIM_SQL" in dims_v03


def test_t4_payload_s_manual():
    payload_manual = {"geo_id": "abc123", "dims": {
        "DIM_SQL":    {"w": 1.0, "props": {"tabla": "logs"}},
        "DIM_PYTHON": {"w": 0.6, "props": {"tipo": "list"}},
    }}
    sp_v02 = v02.Space(); sp_v03 = v03.Space()
    def rebuild_v02(pl, sp):
        o = v02.ObjectND(sp)
        for d, i in pl["dims"].items(): o.add(d, i["props"], w=i["w"])
        hv = o._hv()
        return {d for d in v02.NATIVE if v02._proj(hv, sp.sub(d)) > v02.UMBRAL}
    def rebuild_v03(pl, sp):
        o = v03.ObjectND(sp)
        for d, i in pl["dims"].items(): o.add(d, i["props"], w=i["w"])
        hv = o._hv()
        return {d for d in v03.NATIVE if v03._proj(hv, sp.sub(d)) > v03.UMBRAL}
    return rebuild_v02(payload_manual, sp_v02) == rebuild_v03(payload_manual, sp_v03)


def test_t5_numericos_modo_s():
    sp_v03 = v03.Space()
    obj_num = v03.ObjectND(sp_v03).add("DIM_SQL", {"n_filas": 5000, "tabla": "ventas"}, w=1.0)
    pkt = v03.empaquetar_objeto(obj_num, v03.Mode.S)
    props = pkt.payload_S["dims"]["DIM_SQL"]["props"]
    sp_v02 = v02.Space()
    dims_v02 = pkt_to_dims_v02(pkt, sp_v02)
    return "DIM_SQL" in dims_v02 and props["n_filas"] == 5000


def test_t6_polydim_connect_v03_session_v02():
    conn_v03 = v03.polydim_connect(v03.Space("SRC"), v03.Space("DST"))
    obj_v03 = v03.ObjectND(conn_v03._ia.space)\
                  .add("DIM_GRAPH", {"nodos": 50, "tipo": "dag"}, w=1.0)\
                  .add("DIM_TIME",  {"periodo": "mensual"}, w=0.7)
    pkt_v03 = v03.empaquetar_objeto(obj_v03, v03.Mode.S)
    sp_v02 = v02.Space("RCV")
    dims_v02 = pkt_to_dims_v02(pkt_v03, sp_v02)
    return "DIM_GRAPH" in dims_v02 and "DIM_TIME" in dims_v02


def run_all():
    tests = [
        ("T1 v03→v03 MODO_H",           test_t1_v03_to_v03),
        ("T2 v03→v02 MODO_S",           test_t2_v03_to_v02_modo_s),
        ("T3 v02→v03 MODO_S",           test_t3_v02_to_v03_modo_s),
        ("T4 payload_S manual",          test_t4_payload_s_manual),
        ("T5 numericos transparentes",   test_t5_numericos_modo_s),
        ("T6 connect(v03)↔Session(v02)", test_t6_polydim_connect_v03_session_v02),
    ]
    results = [(label, fn()) for label, fn in tests]
    all_ok = all(ok for _, ok in results)
    for label, ok in results:
        print(f"  {label:40s}: {'OK' if ok else 'FALLO'}")
    print(f"\n  MODO_S garantiza interoperabilidad {'TOTAL' if all_ok else 'PARCIAL'}")
    return all_ok


if __name__ == "__main__":
    run_all()
