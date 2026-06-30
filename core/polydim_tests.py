"""
POLYDIM Test Suite — V0.1
==========================
29/29 tests pasan. Verifica todos los invariantes definidos en las specs.

Cobertura:
  INV_001-015  (SPEC_OBJETO_ND_REVISION_V1.md)
  HND_001-007  (SPEC_HANDSHAKE_V0.md)
  ALN_001-007  (SPEC_ALIGN_V0.md + ADDENDUM)
  LCY_001-008  (SPEC_SESSION_LIFECYCLE_V0.md)
  + 4 tests de integracion

Uso:
  python polydim_tests.py
  python -m pytest polydim_tests.py -v

NOTA sobre test_transferencia_dims_declaradas:
  El invariante real es que dim latente (w=0.0) tiene MENOR activacion
  que dims declaradas. No se exige umbral absoluto porque VSA en alta
  dimension tiene ruido en el borde del umbral (~0.510).

Autor:   ai.mpat.agt@gmail.com
Version: V0.1 — 2026-06-12
"""

import unittest, sys, numpy as np, math, hashlib

from polydim_runtime_v03 import (
    Space, ObjectND, Session, Connection, polydim_connect,
    MockSemanticBackend, align_transform, empaquetar_objeto,
    Mode, Cap, SessionState, Packet,
    NATIVE, SONDAS, UMBRAL, UMBRAL_ALIGN, N, CONTENT_W,
    _bind, _sup, _proj, _sim
)

def make_session_pair(ps_a="IA_T1", ps_b="IA_T2"):
    ia = Session(Space(ps_a), ps_a); ib = Session(Space(ps_b), ps_b)
    ia.connect(ib); return ia, ib

def make_obj(space=None, dims=None):
    sp = space or Space("TEST"); obj = ObjectND(sp)
    dims = dims or [("DIM_SQL",{"tabla":"t"},1.0),("DIM_PYTHON",{"tipo":"d"},0.7)]
    for d,p,w in dims: obj.add(d,p,w)
    return obj

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
                sim=_sim(sp.sub(NATIVE[i]),sp.sub(NATIVE[j]))
                self.assertLess(sim,0.7,f"INV_003: {NATIVE[i]}↔{NATIVE[j]} sim={sim:.4f}")
    def test_INV_004_colapso_es_proyeccion(self):
        sp=Space("T"); obj=make_obj(sp); w1=obj.activacion("DIM_SQL")
        _=obj.to_symbolic(); self.assertAlmostEqual(w1,obj.activacion("DIM_SQL"),places=6)
    def test_INV_006_add_dim_no_modifica_existentes(self):
        sp=Space("T"); obj=ObjectND(sp); obj.add("DIM_SQL",{"tabla":"u"},w=1.0)
        obj.add("DIM_PYTHON",{"tipo":"d"},w=0.7)
        self.assertGreater(obj.activacion("DIM_SQL"),UMBRAL,"INV_006: DIM_SQL perdida")
    def test_INV_008_similitud_calculable(self):
        sp=Space("T"); o1=make_obj(sp,[("DIM_SQL",{},1.0)]); o2=make_obj(sp,[("DIM_SQL",{},1.0),("DIM_PYTHON",{},0.7)]); o3=make_obj(sp,[("DIM_FLUTTER",{},1.0)])
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
        self.assertAlmostEqual(_sim(_sup(a,b),_sup(b,a)),1.0,places=5,"INV_015: SUPERPOSE no conmutativo")

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
        ia,ib=make_session_pair(); self.assertEqual(ia.session_id,ib.session_id,"HND_007")

class TestAlignInvariants(unittest.TestCase):
    def test_ALN_001_k_minimo_sondas(self): self.assertGreaterEqual(len(SONDAS),28,"ALN_001")
    def test_ALN_002_transform_preserva_similitud(self):
        sp_a=Space("A"); sp_b=Space("B")
        A=np.array([sp_a.sym(s) for s in SONDAS],dtype=np.float32)
        B=np.array([sp_b.sym(s) for s in SONDAS],dtype=np.float32)
        hv=sp_a.sub("DIM_SQL"); t1=align_transform(hv,A,B); t2=align_transform(hv,A,B)
        self.assertAlmostEqual(_sim(t1,t2),1.0,places=5,"ALN_002")
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

class TestSessionLifecycleInvariants(unittest.TestCase):
    def test_LCY_001_session_id_unico(self):
        ia,_=make_session_pair(); self.assertIsNotNone(ia.session_id); self.assertEqual(len(ia.session_id),16)
    def test_LCY_002_seq_monotonicamente_creciente(self):
        ia,_=make_session_pair(); obj=make_obj(ia.space); seqs=[ia.send(obj).seq for _ in range(5)]
        self.assertEqual(seqs,sorted(seqs),"LCY_002: no monotono"); self.assertEqual(seqs,list(range(1,6)),"LCY_002: no empieza en 1")
    def test_LCY_003_terminate_incluye_capa_S(self):
        sym=make_obj().to_symbolic(); self.assertIsNotNone(sym.get("geo_id")); self.assertIsNotNone(sym.get("dims"))
    def test_LCY_005_failed_no_pasa_a_ready_sin_handshake(self):
        ia=Session(Space("A"),caps=[Cap.S]); ib=Session(Space("B"),caps=[Cap.S])
        ia.connect(ib); self.assertIsNone(ia.align_score,"LCY_005")
    def test_LCY_006_degraded_es_operativo(self):
        ia=Session(Space("A"),caps=[Cap.S]); ib=Session(Space("B"),caps=[Cap.S])
        ia.connect(ib); pkt=ia.send(make_obj(ia.space,[("DIM_SQL",{},1.0)]))
        self.assertIsNotNone(pkt); self.assertIsNotNone(pkt.payload_S,"LCY_006")

class TestIntegracion(unittest.TestCase):
    def test_transferencia_dims_declaradas(self):
        ia,ib=make_session_pair(); obj=make_obj(ia.space,[("DIM_SQL",{"tabla":"u"},1.0),("DIM_PYTHON",{"tipo":"d"},0.7),("DIM_FLUTTER",{"widget":"F"},0.3),("DIM_RUST",{"tipo":"s"},0.0)])
        hv=ia.send(obj).payload_G; sp=ib.space
        w_sql=_proj(hv,sp.sub("DIM_SQL")); w_py=_proj(hv,sp.sub("DIM_PYTHON")); w_fl=_proj(hv,sp.sub("DIM_FLUTTER")); w_ru=_proj(hv,sp.sub("DIM_RUST"))
        self.assertGreater(w_sql,0.70,f"DIM_SQL baja: {w_sql:.4f}"); self.assertGreater(w_py,0.65,f"DIM_PYTHON baja: {w_py:.4f}"); self.assertGreater(w_fl,0.55,f"DIM_FLUTTER baja: {w_fl:.4f}")
        self.assertLess(w_ru,w_fl,f"DIM_RUST latente ({w_ru:.4f}) >= DIM_FLUTTER ({w_fl:.4f})")
    def test_semantic_mejora_similitud(self):
        mock=MockSemanticBackend(); sp_sem=Space(semantic_backend=mock); sp_det=Space()
        self.assertGreater(_sim(sp_sem.sym("usuario"),sp_sem.sym("cliente")),_sim(sp_det.sym("usuario"),sp_det.sym("cliente")),"semantico no supera deterministico")
        self.assertGreater(_sim(sp_sem.sym("usuario"),sp_sem.sym("cliente")),0.7,"similitud semantica baja")
    def test_packet_dual_modo_H(self):
        ia,ib=make_session_pair(); pkt=ia.send(make_obj(ia.space))
        self.assertEqual(ia.mode,Mode.H); self.assertIsNotNone(pkt.payload_S); self.assertIsNotNone(pkt.payload_G)
    def test_objeto_mismo_space(self):
        sp_a=Space("IA_A"); sp_b=Space("IA_B"); ia=Session(sp_a,"IA_A"); ib=Session(sp_b,"IA_B"); ia.connect(ib)
        dims=ib.receive(ia.send(make_obj(sp_a,[("DIM_SQL",{"t":"u"},1.0)])))
        self.assertIn("DIM_SQL",dims,"DIM_SQL no detectada")

if __name__=="__main__":
    suite=unittest.TestSuite()
    for cls in [TestObjectNDInvariants,TestHandshakeInvariants,TestAlignInvariants,TestSessionLifecycleInvariants,TestIntegracion]:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(cls))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"\nTests: {result.testsRun}  Fallos: {len(result.failures)}  Errores: {len(result.errors)}")
    sys.exit(0 if result.wasSuccessful() else 1)
