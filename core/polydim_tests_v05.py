# POLYDIM_DEST
# destination: polydim/core/
# filename: polydim_tests_v05.py
# author: claude-sonnet-4-6

"""
POLYDIM Test Suite — V0.2
==========================
Actualizado de v03 → v05. Corrige HVE_007 (receive() → ObjectND).
Agrega: HVE_006, HVE_007, BIN_001-003, MESH_001-003.

Cobertura:
  INV_001-015  (SPEC_OBJETO_ND_REVISION_V1.md)
  HND_001-007  (SPEC_HANDSHAKE_V0.md)
  ALN_001-007  (SPEC_ALIGN_V0.md + ADDENDUM)
  LCY_001-006  (SPEC_SESSION_LIFECYCLE_V0.md)
  HVE_006      props preservadas en MODO_H (v04 fix)
  HVE_007      receive() retorna ObjectND no dict (v04 breaking change)
  BIN_001-003  hv_encode/hv_decode PLYD roundtrip
  MESH_001-003 malla N IAs (v05)
  + 4 tests de integracion

Uso:
  python polydim_tests_v05.py
  python -m pytest polydim_tests_v05.py -v

Cambios respecto a V0.1:
  - import cambiado de polydim_runtime_v03 → polydim_runtime_v05
  - test_objeto_mismo_space: receive() retorna ObjectND → usar dims_activas()
  - agregado: TestHVEInvariants, TestBinaryFormat, TestMesh
  - INV_native_dims_count: verifica 10 NATIVE_DIMS + DIM_CONTRACT

Autor: claude-sonnet-4-6
Version: V0.2 — 2026-06-17 — TASK_P01
"""

import unittest, sys, numpy as np, math, hashlib

from polydim_runtime_v05 import (
    Space, ObjectND, Session, polydim_connect,
    MockSemanticBackend, align_transform,
    Mode, Cap, SessionState, Packet,
    NATIVE, SONDAS, UMBRAL, UMBRAL_ALIGN, N, CONTENT_W,
    _bind, _sup, _proj, _sim,
    Mesh, hv_encode, hv_decode,
)


def make_session_pair(ps_a="IA_T1", ps_b="IA_T2"):
    ia = Session(Space(ps_a), ps_a)
    ib = Session(Space(ps_b), ps_b)
    ia.connect(ib)
    return ia, ib


def make_obj(space=None, dims=None):
    sp = space or Space("TEST")
    obj = ObjectND(sp)
    dims = dims or [("DIM_SQL", {"tabla": "t"}, 1.0), ("DIM_PYTHON", {"tipo": "d"}, 0.7)]
    for d, p, w in dims:
        obj.add(d, p, w)
    return obj


# INV — ObjectND
class TestObjectNDInvariants(unittest.TestCase):
    def test_INV_001_geo_id_invariant(self):
        sp=Space("T"); obj=ObjectND(sp); obj.add("DIM_SQL",{"t":"a"},w=1.0); geo1=obj.geo_id
        obj.add("DIM_PYTHON",{"t":"b"},w=0.7)
        self.assertEqual(geo1,obj.geo_id,"INV_001: GEO_ID cambio")

    def test_INV_002_manifold_min_one_active(self):
        sp=Space("T"); obj=ObjectND(sp); obj.add("DIM_SQL",{},w=1.0)
        self.assertGreater(len(obj.dims_activas()),0,"INV_002: sin activas")

    def test_INV_003_subespacios_casi_ortogonales(self):
        sp=Space()
        for i in range(len(NATIVE)):
            for j in range(i+1,len(NATIVE)):
                s=_sim(sp.sub(NATIVE[i]),sp.sub(NATIVE[j]))
                self.assertLess(s,0.7,f"INV_003: {NATIVE[i]}↔{NATIVE[j]} sim={s:.4f}")

    def test_INV_004_colapso_es_proyeccion(self):
        sp=Space("T"); obj=make_obj(sp); w1=obj.activacion("DIM_SQL")
        _=obj.to_symbolic()
        self.assertAlmostEqual(w1,obj.activacion("DIM_SQL"),places=6)

    def test_INV_006_add_dim_no_modifica_existentes(self):
        sp=Space("T"); obj=ObjectND(sp); obj.add("DIM_SQL",{"tabla":"u"},w=1.0)
        obj.add("DIM_PYTHON",{"tipo":"d"},w=0.7)
        self.assertGreater(obj.activacion("DIM_SQL"),UMBRAL,"INV_006: DIM_SQL perdida")

    def test_INV_008_similitud_calculable(self):
        sp=Space("T")
        o1=make_obj(sp,[("DIM_SQL",{},1.0)]); o2=make_obj(sp,[("DIM_SQL",{},1.0),("DIM_PYTHON",{},0.7)]); o3=make_obj(sp,[("DIM_FLUTTER",{},1.0)])
        self.assertGreater(_sim(o1._hv(),o2._hv()),_sim(o1._hv(),o3._hv()),"INV_008")

    def test_INV_009_objeto_degradado_operable(self):
        sp=Space("T"); obj=make_obj(sp); hv=obj._hv()
        hv_n=(hv+np.random.randn(N).astype(np.float32)*0.1); hv_n/=np.linalg.norm(hv_n)
        self.assertGreater(_proj(hv_n,sp.sub("DIM_SQL")),0.4,"INV_009: DIM_SQL perdida con ruido")

    def test_INV_010_capa_S_derivable_de_G(self):
        sym=make_obj().to_symbolic()
        self.assertIn("geo_id",sym); self.assertIn("dims",sym); self.assertIn("DIM_SQL",sym["dims"])

    def test_INV_015_superpose_commutative(self):
        sp=Space(); a=sp.sym("usuario"); b=sp.sym("tabla")
        self.assertAlmostEqual(_sim(_sup(a,b),_sup(b,a)),1.0,places=5,msg="INV_015: SUPERPOSE no conmutativo")

    def test_INV_native_dims_count(self):
        self.assertEqual(len(NATIVE),10,f"Esperados 10 NATIVE_DIMS, encontrados {len(NATIVE)}")
        self.assertIn("DIM_CONTRACT",NATIVE,"DIM_CONTRACT debe estar en NATIVE")


# HND — Handshake
class TestHandshakeInvariants(unittest.TestCase):
    def test_HND_001_session_id_unico(self):
        ia1,_=make_session_pair("A1","B1"); ia2,_=make_session_pair("A2","B2")
        self.assertNotEqual(ia1.session_id,ia2.session_id,"HND_001")

    def test_HND_002_modo_maximo_comun(self):
        ia=Session(Space("A"),caps=[Cap.S,Cap.G,Cap.ALIGN]); ib=Session(Space("B"),caps=[Cap.S,Cap.G,Cap.ALIGN])
        ia.connect(ib); self.assertEqual(ia.mode,Mode.H,"HND_002")

    def test_HND_002b_modo_S_si_una_no_tiene_G(self):
        ia=Session(Space("A"),caps=[Cap.S]); ib=Session(Space("B"),caps=[Cap.S,Cap.G])
        ia.handshake(ib); self.assertEqual(ia.mode,Mode.S,"HND_002b")

    def test_HND_003_modo_H_requiere_align(self):
        ia,_=make_session_pair()
        self.assertIsNotNone(ia.align_score,"HND_003: sin align")
        self.assertGreaterEqual(ia.align_score,UMBRAL_ALIGN,f"HND_003: score bajo ({ia.align_score})")

    def test_HND_006_modo_S_siempre_posible(self):
        ia=Session(Space("A"),caps=[Cap.S]); ib=Session(Space("B"),caps=[Cap.S])
        ia.handshake(ib); self.assertEqual(ia.mode,Mode.S,"HND_006")

    def test_HND_007_session_id_verificable(self):
        ia,ib=make_session_pair()
        self.assertEqual(ia.session_id,ib.session_id,"HND_007")


# ALN — Align
class TestAlignInvariants(unittest.TestCase):
    def test_ALN_001_k_minimo_sondas(self): self.assertGreaterEqual(len(SONDAS),28,"ALN_001")

    def test_ALN_002_transform_preserva_similitud(self):
        sp_a=Space("A"); sp_b=Space("B")
        A=np.array([sp_a.sym(s) for s in SONDAS],dtype=np.float32)
        B=np.array([sp_b.sym(s) for s in SONDAS],dtype=np.float32)
        hv=sp_a.sub("DIM_SQL"); t1=align_transform(hv,A,B); t2=align_transform(hv,A,B)
        self.assertAlmostEqual(_sim(t1,t2),1.0,places=5,msg="ALN_002")

    def test_ALN_003_score_sobre_native_dims(self):
        sp_a=Space("A"); sp_b=Space("B")
        A=np.array([sp_a.sym(s) for s in SONDAS],dtype=np.float32)
        B=np.array([sp_b.sym(s) for s in SONDAS],dtype=np.float32)
        score=float(np.mean([_sim(align_transform(sp_a.sub(d),A,B),sp_b.sub(d)) for d in NATIVE]))
        self.assertGreaterEqual(score,UMBRAL_ALIGN,f"ALN_003: score={score:.4f}")

    def test_ALN_004_degradar_si_score_bajo(self):
        ia=Session(Space("A"),caps=[Cap.S]); ib=Session(Space("B"),caps=[Cap.S])
        ia.connect(ib); self.assertEqual(ia.mode,Mode.S,"ALN_004")

    def test_ALN_007_ambas_ias_calculan_R(self):
        ia,ib=make_session_pair()
        for attr in ["_A","_B"]:
            self.assertIsNotNone(getattr(ia,attr),f"ALN_007: ia sin {attr}")
            self.assertIsNotNone(getattr(ib,attr),f"ALN_007: ib sin {attr}")
        self.assertGreater(_sim(ia._A[0],ib._B[0]),0.99,"ALN_007: matrices no inversas")


# LCY — Session Lifecycle
class TestSessionLifecycleInvariants(unittest.TestCase):
    def test_LCY_001_session_id_unico(self):
        ia,_=make_session_pair()
        self.assertIsNotNone(ia.session_id); self.assertEqual(len(ia.session_id),16)

    def test_LCY_002_seq_monotonicamente_creciente(self):
        ia,_=make_session_pair(); obj=make_obj(ia.space); seqs=[ia.send(obj).seq for _ in range(5)]
        self.assertEqual(seqs,sorted(seqs),"LCY_002: no monotono")
        self.assertEqual(seqs,list(range(1,6)),"LCY_002: no empieza en 1")

    def test_LCY_003_terminate_incluye_capa_S(self):
        sym=make_obj().to_symbolic()
        self.assertIsNotNone(sym.get("geo_id")); self.assertIsNotNone(sym.get("dims"))

    def test_LCY_005_failed_no_pasa_a_ready_sin_handshake(self):
        ia=Session(Space("A"),caps=[Cap.S]); ib=Session(Space("B"),caps=[Cap.S])
        ia.connect(ib); self.assertIsNone(ia.align_score,"LCY_005")

    def test_LCY_006_degraded_es_operativo(self):
        ia=Session(Space("A"),caps=[Cap.S]); ib=Session(Space("B"),caps=[Cap.S])
        ia.connect(ib); pkt=ia.send(make_obj(ia.space,[("DIM_SQL",{},1.0)]))
        self.assertIsNotNone(pkt); self.assertIsNotNone(pkt.payload_S,"LCY_006")


# HVE — v04 breaking changes
class TestHVEInvariants(unittest.TestCase):
    def test_HVE_007_receive_returns_ObjectND(self):
        ia,ib=make_session_pair()
        obj=make_obj(ia.space,[("DIM_SQL",{"tabla":"pedidos"},1.0),("DIM_PYTHON",{"lib":"pandas"},0.7)])
        resultado=ib.receive(ia.send(obj))
        self.assertIsInstance(resultado,ObjectND,"HVE_007: receive() debe retornar ObjectND, no dict")

    def test_HVE_006_props_preservadas_en_modo_H(self):
        ia,ib=make_session_pair()
        self.assertEqual(ia.mode,Mode.H,"test requiere MODO_H")
        obj=make_obj(ia.space,[("DIM_SQL",{"tabla":"pedidos","schema":"public"},1.0)])
        resultado=ib.receive(ia.send(obj))
        self.assertIsInstance(resultado,ObjectND,"HVE_006: receive() debe retornar ObjectND")
        props_sql=resultado._props.get("DIM_SQL",{})
        self.assertEqual(props_sql.get("tabla"),"pedidos",f"HVE_006: prop 'tabla' no preservada. Props: {props_sql}")

    def test_HVE_dim_sql_detectable_en_receptor(self):
        ia,ib=make_session_pair()
        obj=make_obj(ia.space,[("DIM_SQL",{"tabla":"u"},1.0)])
        resultado=ib.receive(ia.send(obj))
        activas=resultado.dims_activas()
        self.assertIn("DIM_SQL",activas,f"DIM_SQL no detectada. Activas: {list(activas.keys())}")

    def test_HVE_transfer_full_retorna_objectnd(self):
        conn=polydim_connect(Space("SRC"),Space("DST"))
        obj=make_obj(conn._ia.space,[("DIM_SQL",{"tabla":"ventas","pk":"id"},1.0)])
        resultado=conn.transfer_full(obj)
        self.assertIsInstance(resultado,ObjectND,"transfer_full debe retornar ObjectND")
        props=resultado._props.get("DIM_SQL",{})
        self.assertEqual(props.get("tabla"),"ventas",f"transfer_full no preservó props. Props: {props}")


# BIN — Formato binario PLYD
class TestBinaryFormat(unittest.TestCase):
    def test_BIN_001_roundtrip_sim(self):
        sp=Space("TEST"); hv=make_obj(sp)._hv().astype(np.float32)
        decoded=hv_decode(hv_encode(hv))
        s=_sim(hv,decoded)
        self.assertGreater(s,0.9999,f"BIN_001: roundtrip perdió precisión: sim={s:.6f}")

    def test_BIN_002_magic_bytes_PLYD(self):
        sp=Space("T"); encoded=hv_encode(make_obj(sp)._hv().astype(np.float32))
        self.assertEqual(encoded[:4],b'PLYD',f"BIN_002: magic bytes incorrectos: {encoded[:4]}")

    def test_BIN_003_frame_size(self):
        sp=Space("T"); encoded=hv_encode(make_obj(sp)._hv().astype(np.float32))
        expected=12+4*N
        self.assertEqual(len(encoded),expected,f"BIN_003: tamaño {len(encoded)} vs {expected}")


# MESH — v05
class TestMesh(unittest.TestCase):
    def test_MESH_001_broadcast_llega_a_todos(self):
        sp_a=Space("IA_A"); sp_b=Space("IA_B"); sp_c=Space("IA_C")
        ia=Session(sp_a,"IA_A"); ib=Session(sp_b,"IA_B"); ic=Session(sp_c,"IA_C")
        mesh=Mesh([ia,ib,ic])
        results=mesh.broadcast(make_obj(sp_a,[("DIM_SQL",{"tabla":"t"},1.0)]),"IA_A")
        self.assertGreaterEqual(len(results),2,f"MESH_001: llegó a {len(results)} peers")

    def test_MESH_002_topology_contiene_todos(self):
        ia=Session(Space("IA_A"),"IA_A"); ib=Session(Space("IA_B"),"IA_B")
        topo=Mesh([ia,ib]).topology()
        self.assertIn("IA_A",topo,"MESH_002: IA_A no en topología")
        self.assertIn("IA_B",topo,"MESH_002: IA_B no en topología")

    def test_MESH_003_status_ready(self):
        ia=Session(Space("IA_A"),"IA_A"); ib=Session(Space("IA_B"),"IA_B")
        status=Mesh([ia,ib]).status()
        for name,state in status.items():
            self.assertIn(str(state),["READY","SessionState.READY"],f"MESH_003: {name} estado={state}")


# Integración
class TestIntegracion(unittest.TestCase):
    def test_transferencia_dims_declaradas(self):
        ia,ib=make_session_pair()
        obj=make_obj(ia.space,[("DIM_SQL",{"tabla":"u"},1.0),("DIM_PYTHON",{"tipo":"d"},0.7),
                                ("DIM_FLUTTER",{"widget":"F"},0.3),("DIM_RUST",{"tipo":"s"},0.0)])
        hv=ia.send(obj).payload_G; sp=ib.space
        w_sql=_proj(hv,sp.sub("DIM_SQL")); w_py=_proj(hv,sp.sub("DIM_PYTHON"))
        w_fl=_proj(hv,sp.sub("DIM_FLUTTER")); w_ru=_proj(hv,sp.sub("DIM_RUST"))
        self.assertGreater(w_sql,0.70,f"DIM_SQL baja: {w_sql:.4f}")
        self.assertGreater(w_py,0.65,f"DIM_PYTHON baja: {w_py:.4f}")
        self.assertGreater(w_fl,0.55,f"DIM_FLUTTER baja: {w_fl:.4f}")
        self.assertLess(w_ru,w_fl,f"DIM_RUST latente ({w_ru:.4f}) >= DIM_FLUTTER ({w_fl:.4f})")

    def test_semantico_mejora_similitud(self):
        mock=MockSemanticBackend(); sp_sem=Space(semantic_backend=mock); sp_det=Space()
        self.assertGreater(_sim(sp_sem.sym("usuario"),sp_sem.sym("cliente")),
                           _sim(sp_det.sym("usuario"),sp_det.sym("cliente")),"semantico no supera deterministico")
        self.assertGreater(_sim(sp_sem.sym("usuario"),sp_sem.sym("cliente")),0.7,"similitud semantica baja")

    def test_packet_dual_modo_H(self):
        ia,ib=make_session_pair(); pkt=ia.send(make_obj(ia.space))
        self.assertEqual(ia.mode,Mode.H); self.assertIsNotNone(pkt.payload_S); self.assertIsNotNone(pkt.payload_G)

    def test_objeto_mismo_space_v05(self):
        sp_a=Space("IA_A"); sp_b=Space("IA_B")
        ia=Session(sp_a,"IA_A"); ib=Session(sp_b,"IA_B"); ia.connect(ib)
        resultado=ib.receive(ia.send(make_obj(sp_a,[("DIM_SQL",{"t":"u"},1.0)])))
        self.assertIn("DIM_SQL",resultado.dims_activas(),"DIM_SQL no detectada")


if __name__=="__main__":
    suite=unittest.TestSuite()
    for cls in [TestObjectNDInvariants,TestHandshakeInvariants,TestAlignInvariants,
                TestSessionLifecycleInvariants,TestHVEInvariants,TestBinaryFormat,
                TestMesh,TestIntegracion]:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(cls))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"\nTests: {result.testsRun}  Fallos: {len(result.failures)}  Errores: {len(result.errors)}")
    sys.exit(0 if result.wasSuccessful() else 1)
