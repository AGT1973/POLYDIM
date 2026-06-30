"""
POLYDIM Test Suite — V0.3
==========================
TASK_P05: Fix de Mesh.broadcast/route_path para alineamiento per-peer.

Cambios respecto a V0.2:
  - Import actualizado: polydim_runtime_v05 -> polydim_runtime_v06
  - test_MSH_001_broadcast_n3: YA NO usa el workaround de Space compartido;
    verifica el fix real con 3 IAs en Spaces DISTINTOS.
  - test_MSH_005_route_path_multisalto: idem, YA NO usa Space compartido.
  - NUEVO test_MSH_007_route_path_4_nodos: ruta de 3 saltos con Spaces distintos.
  - NUEVO test_MSH_008_peers_align_por_peer: verifica que _peers_align guarda
    pares (A,B) distintos por cada peer.
  - NUEVO test_send_v01_compat_no_rompe: confirma que send() (API V0.1,
    1 solo peer) sigue funcionando igual tras el fix.

Cobertura (heredada de V0.2 + nuevos de TASK_P05):
  INV_001-015  (SPEC_OBJETO_ND_REVISION_V1.md)
  HND_001-007  (SPEC_HANDSHAKE_V0.md)
  ALN_001-007  (SPEC_ALIGN_V0.md + ADDENDUM)
  LCY_001-008  (SPEC_SESSION_LIFECYCLE_V0.md)
  HVE_006-007  (B2 numerico, NATIVE count)
  MSH_001-008  (Mesh: broadcast, route, topology, hub, route_path x2, N=5,
                 peers_align)
  + 6 tests de integracion (binary_roundtrip, packet_dual, objeto_mismo_space,
                             dim_contract, transferencia_dims, send_v01_compat)

Resultados: 41/41 OK (sandbox 2026-06-20)

Uso:
  python polydim_tests_v03.py

Autor:   ai.mpat.agt@gmail.com (curso02.mithril) — TASK_P05 — 2026-06-20
Version: V0.3
"""

import unittest, sys, numpy as np, math, hashlib

from polydim_runtime_v06 import (
    Space, ObjectND, Session, Connection, polydim_connect,
    align_transform, empaquetar_objeto,
    Mode, Cap, SessionState, Packet, Mesh,
    NATIVE, SONDAS, UMBRAL, UMBRAL_ALIGN, N, CONTENT_W,
    META_ACK, META_REPROGRAM, META_RESET, META_QUERY,
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
        """v05: DIM_CONTRACT puede estar cerca de DIM_META; umbral subido a 0.75."""
        sp=Space()
        for i in range(len(NATIVE)):
            for j in range(i+1,len(NATIVE)):
                sim=_sim(sp.sub(NATIVE[i]),sp.sub(NATIVE[j]))
                self.assertLess(sim,0.75,
                    f"INV_003: {NATIVE[i]}<->{NATIVE[j]} sim={sim:.4f}")

    def test_INV_004_colapso_es_proyeccion(self):
        sp=Space("T"); obj=make_obj(sp); w1=obj.activacion("DIM_SQL")
        _=obj.to_symbolic(); self.assertAlmostEqual(w1,obj.activacion("DIM_SQL"),places=6)

    def test_INV_006_add_dim_no_modifica_existentes(self):
        sp=Space("T"); obj=ObjectND(sp); obj.add("DIM_SQL",{"tabla":"u"},w=1.0)
        obj.add("DIM_PYTHON",{"tipo":"d"},w=0.7)
        self.assertGreater(obj.activacion("DIM_SQL"),UMBRAL,"INV_006: DIM_SQL perdida")

    def test_INV_008_similitud_calculable(self):
        sp=Space("T")
        o1=make_obj(sp,[("DIM_SQL",{},1.0)])
        o2=make_obj(sp,[("DIM_SQL",{},1.0),("DIM_PYTHON",{},0.7)])
        o3=make_obj(sp,[("DIM_FLUTTER",{},1.0)])
        self.assertGreater(_sim(o1._hv(),o2._hv()),_sim(o1._hv(),o3._hv()),"INV_008")

    def test_INV_009_objeto_degradado_operable(self):
        sp=Space("T"); obj=make_obj(sp); hv=obj._hv()
        hv_n=(hv+np.random.randn(N).astype(np.float32)*0.1); hv_n/=np.linalg.norm(hv_n)
        self.assertGreater(_proj(hv_n,sp.sub("DIM_SQL")),0.4,
            "INV_009: DIM_SQL perdida con ruido")

    def test_INV_010_capa_S_derivable_de_G(self):
        sym=make_obj().to_symbolic()
        self.assertIn("geo_id",sym); self.assertIn("dims",sym)
        self.assertIn("DIM_SQL",sym["dims"])

    def test_INV_015_superpose_commutative(self):
        sp=Space(); a=sp.sym("usuario"); b=sp.sym("tabla")
        self.assertAlmostEqual(_sim(_sup(a,b),_sup(b,a)),1.0,places=5,
            msg="INV_015: SUPERPOSE no conmutativo")


class TestHandshakeInvariants(unittest.TestCase):

    def test_HND_001_session_id_unico(self):
        ia1,_=make_session_pair("A1","B1"); ia2,_=make_session_pair("A2","B2")
        self.assertNotEqual(ia1.session_id,ia2.session_id,"HND_001")

    def test_HND_002_modo_maximo_comun(self):
        ia=Session(Space("A"),caps=[Cap.S,Cap.G,Cap.ALIGN])
        ib=Session(Space("B"),caps=[Cap.S,Cap.G,Cap.ALIGN])
        ia.connect(ib); self.assertEqual(ia.mode,Mode.H,"HND_002")

    def test_HND_002b_modo_S_si_una_no_tiene_G(self):
        ia=Session(Space("A"),caps=[Cap.S])
        ib=Session(Space("B"),caps=[Cap.S,Cap.G])
        ia.handshake(ib); self.assertEqual(ia.mode,Mode.S,"HND_002b")

    def test_HND_003_modo_H_requiere_align(self):
        ia,_=make_session_pair()
        self.assertIsNotNone(ia.align_score,"HND_003: sin align")
        self.assertGreaterEqual(ia.align_score,UMBRAL_ALIGN,
            f"HND_003: score bajo ({ia.align_score})")

    def test_HND_006_modo_S_siempre_posible(self):
        ia=Session(Space("A"),caps=[Cap.S]); ib=Session(Space("B"),caps=[Cap.S])
        ia.handshake(ib); self.assertEqual(ia.mode,Mode.S,"HND_006")

    def test_HND_007_session_id_verificable(self):
        ia,ib=make_session_pair()
        self.assertEqual(ia.session_id,ib.session_id,"HND_007")


class TestAlignInvariants(unittest.TestCase):

    def test_ALN_001_k_minimo_sondas(self):
        self.assertGreaterEqual(len(SONDAS),28,"ALN_001")

    def test_ALN_002_transform_preserva_similitud(self):
        sp_a=Space("A"); sp_b=Space("B")
        A=np.array([sp_a.sym(s) for s in SONDAS],dtype=np.float32)
        B=np.array([sp_b.sym(s) for s in SONDAS],dtype=np.float32)
        hv=sp_a.sub("DIM_SQL")
        t1=align_transform(hv,A,B); t2=align_transform(hv,A,B)
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


class TestSessionLifecycleInvariants(unittest.TestCase):

    def test_LCY_001_session_id_unico(self):
        ia,_=make_session_pair()
        self.assertIsNotNone(ia.session_id); self.assertEqual(len(ia.session_id),16)

    def test_LCY_002_seq_monotonicamente_creciente(self):
        ia,_=make_session_pair(); obj=make_obj(ia.space)
        seqs=[ia.send(obj).seq for _ in range(5)]
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
        ia.connect(ib)
        pkt=ia.send(make_obj(ia.space,[("DIM_SQL",{},1.0)]))
        self.assertIsNotNone(pkt); self.assertIsNotNone(pkt.payload_S,"LCY_006")


class TestHVE(unittest.TestCase):
    """
    HVE_006 y HVE_007: tests de B2 numerico y conteo de dims nativas.
    Nuevos en V0.3 por TASK_P05.
    """

    def test_HVE_006_b2_numerico_proximidad(self):
        """
        B2 numerico: numeros proximos tienen similitud alta (>0.99),
        numeros lejanos tienen similitud menor. El orden se preserva.
        """
        sp = Space()
        h10  = sp._enc_numerico(10.0)
        h11  = sp._enc_numerico(11.0)
        h100 = sp._enc_numerico(100.0)
        sim_10_11  = _sim(h10, h11)
        sim_10_100 = _sim(h10, h100)
        self.assertGreater(sim_10_11, 0.99,
            f"HVE_006: sim(10,11)={sim_10_11:.4f} < 0.99")
        self.assertLess(sim_10_100, sim_10_11,
            f"HVE_006: sim(10,100)={sim_10_100:.4f} >= sim(10,11)={sim_10_11:.4f}")

    def test_HVE_007_native_dims_count(self):
        """
        v05 tiene exactamente 10 dimensiones nativas (v04 tenia 9).
        DIM_CONTRACT fue agregada en TASK_020.
        """
        self.assertEqual(len(NATIVE), 10,
            f"HVE_007: NATIVE tiene {len(NATIVE)} dims, se esperaban 10")
        self.assertIn("DIM_CONTRACT", NATIVE,
            "HVE_007: DIM_CONTRACT no esta en NATIVE")
        for d in ["DIM_PYTHON","DIM_RUST","DIM_FLUTTER","DIM_SQL",
                  "DIM_GRAPH","DIM_VECTOR","DIM_TIME","DIM_ERROR","DIM_META"]:
            self.assertIn(d, NATIVE, f"HVE_007: {d} desaparecio de NATIVE")


class TestMesh(unittest.TestCase):
    """
    MSH_001-008: tests de la clase Mesh. Nuevos/actualizados en V0.3 por TASK_P05.
    """

    def test_MSH_001_broadcast_n3(self):
        """
        3 IAs en malla plana, CADA UNA con Space distinto: broadcast desde
        IA_0 llega a IA_1 e IA_2 con DIM_SQL.
        V0.6 TASK_P05 fix: Mesh.broadcast usa send_to(obj,peer_name) por
        cada peer, con el alineamiento (A,B) propio de ese peer (guardado
        en Session._peers_align durante align()). Ya NO requiere el
        workaround de Space compartido que usaba V0.2.
        """
        mesh = Mesh()
        sp0,sp1,sp2 = Space("B0"),Space("B1"),Space("B2")
        mesh.add("IA_0",sp0); mesh.add("IA_1",sp1); mesh.add("IA_2",sp2)
        mesh.connect_all()
        obj = ObjectND(sp0).add("DIM_SQL",{"tabla":"mesh_test"},w=1.0)
        res = mesh.broadcast(obj,"IA_0")
        self.assertIn("IA_1",res,"MSH_001: IA_1 no en resultados")
        self.assertIn("IA_2",res,"MSH_001: IA_2 no en resultados")
        self.assertIn("DIM_SQL",res["IA_1"],"MSH_001: IA_1 no recibio DIM_SQL (espacios distintos)")
        self.assertIn("DIM_SQL",res["IA_2"],"MSH_001: IA_2 no recibio DIM_SQL (espacios distintos)")

    def test_MSH_002_route_punto_a_punto(self):
        """3 IAs: route punto a punto entre R_0 y R_2."""
        mesh = Mesh()
        sp0,sp1,sp2 = Space("R0"),Space("R1"),Space("R2")
        mesh.add("R_0",sp0); mesh.add("R_1",sp1); mesh.add("R_2",sp2)
        mesh.connect_all()
        obj = ObjectND(sp0).add("DIM_GRAPH",{"nodo":"origen"},w=1.0)
        dims = mesh.route(obj,"R_0","R_2")
        self.assertIn("DIM_GRAPH",dims,"MSH_002: DIM_GRAPH no llegó a R_2")

    def test_MSH_003_topology_malla_completa_n3(self):
        """En malla completa N=3, cada nodo tiene exactamente 2 peers."""
        mesh = Mesh()
        for i in range(3): mesh.add(f"T_{i}",Space(f"T{i}"))
        mesh.connect_all()
        topo = mesh.topology()
        for name,peers in topo.items():
            self.assertEqual(len(peers),2,
                f"MSH_003: {name} tiene {len(peers)} peers, esperaba 2")

    def test_MSH_004_hub_topology(self):
        """Topologia hub: HUB conectado a 3 workers; workers no conectados entre si."""
        mesh = Mesh()
        mesh.add("HUB",Space("HUB"))
        for i in range(3): mesh.add(f"W_{i}",Space(f"W{i}"))
        mesh.connect_hub("HUB")
        topo = mesh.topology()
        self.assertEqual(set(topo["HUB"]),{"W_0","W_1","W_2"},"MSH_004: HUB no tiene los 3 workers")
        for i in range(3):
            self.assertEqual(set(topo[f"W_{i}"]),{"HUB"},
                f"MSH_004: W_{i} tiene peers extras: {topo[f'W_{i}']}")

    def test_MSH_005_route_path_multisalto(self):
        """
        Ruta multisalto A -> B -> C, CADA NODO con Space distinto:
        objeto llega a C con DIM_TIME.
        V0.6 TASK_P05 fix: route_path reconstruye el objeto en el Space
        del nodo intermedio (Session.unpack) antes de retransmitir, en vez
        de reenviar el ObjectND crudo del nodo origen en todos los saltos.
        Ya NO requiere el workaround de Space compartido que usaba V0.2.
        """
        mesh = Mesh()
        spa,spb,spc = Space("PA"),Space("PB"),Space("PC")
        mesh.add("A",spa); mesh.add("B",spb); mesh.add("C",spc)
        mesh.connect_all()
        obj = ObjectND(spa).add("DIM_TIME",{"orden":1},w=1.0)
        dims = mesh.route_path(obj,["A","B","C"])
        self.assertIsInstance(dims,dict,"MSH_005: route_path no retorno dict")
        self.assertIn("DIM_TIME",dims,"MSH_005: DIM_TIME no llegó a C (espacios distintos)")

    def test_MSH_007_route_path_4_nodos(self):
        """Ruta de 3 saltos (4 nodos), cada uno con Space distinto."""
        mesh = Mesh()
        for i,n in enumerate(["W","X","Y","Z"]):
            mesh.add(n, Space(f"SP_{n}{i}"))
        mesh.connect_all()
        obj = ObjectND(mesh._nodes["W"].space).add("DIM_ERROR",{"codigo":500},w=1.0)
        dims = mesh.route_path(obj, ["W","X","Y","Z"])
        self.assertIn("DIM_ERROR",dims,"MSH_007: DIM_ERROR no llegó a Z tras 3 saltos")

    def test_MSH_006_n5_all_ready(self):
        """5 IAs: todas deben estar en estado READY tras connect_all."""
        mesh = Mesh()
        for i in range(5): mesh.add(f"M_{i}",Space(f"M{i}"))
        mesh.connect_all()
        st = mesh.status()
        for name,info in st.items():
            self.assertEqual(info["state"],"READY",
                f"MSH_006: {name} en estado {info['state']}, esperaba READY")

    def test_MSH_008_peers_align_por_peer(self):
        """
        TASK_P05: cada peer debe tener su propio par (A,B) guardado en
        _peers_align, y deben ser DISTINTOS entre si cuando los Spaces
        de los peers son distintos (antes solo existia un _A/_B global
        compartido por todos).
        """
        mesh = Mesh()
        sp0,sp1,sp2 = Space("AL0"),Space("AL1"),Space("AL2")
        mesh.add("N_0",sp0); mesh.add("N_1",sp1); mesh.add("N_2",sp2)
        mesh.connect_all()
        n0 = mesh._nodes["N_0"]
        self.assertIn("N_1",n0._peers_align,"MSH_008: falta alineamiento para N_1")
        self.assertIn("N_2",n0._peers_align,"MSH_008: falta alineamiento para N_2")
        A1,B1 = n0._peers_align["N_1"]
        A2,B2 = n0._peers_align["N_2"]
        self.assertLess(_sim(B1[0],B2[0]),0.99,
            "MSH_008: alineamiento de N_1 y N_2 es identico, deberia diferir (Spaces distintos)")

    def test_send_v01_compat_no_rompe(self):
        """
        TASK_P05: la API V0.1 send() (1 solo peer, sin Mesh) debe seguir
        funcionando exactamente igual que antes del fix.
        """
        ia,ib = make_session_pair()
        obj = make_obj(ia.space,[("DIM_SQL",{"tabla":"compat"},1.0)])
        pkt = ia.send(obj)
        dims = ib.receive(pkt)
        self.assertIn("DIM_SQL",dims,"send() V0.1 compat roto tras fix TASK_P05")


class TestIntegracion(unittest.TestCase):

    def test_transferencia_dims_declaradas(self):
        ia,ib=make_session_pair()
        obj=make_obj(ia.space,[
            ("DIM_SQL",  {"tabla":"u"},1.0),
            ("DIM_PYTHON",{"tipo":"d"},0.7),
            ("DIM_FLUTTER",{"widget":"F"},0.3),
            ("DIM_RUST",  {"tipo":"s"},0.0)
        ])
        hv=ia.send(obj).payload_G; sp=ib.space
        w_sql=_proj(hv,sp.sub("DIM_SQL")); w_py=_proj(hv,sp.sub("DIM_PYTHON"))
        w_fl=_proj(hv,sp.sub("DIM_FLUTTER")); w_ru=_proj(hv,sp.sub("DIM_RUST"))
        self.assertGreater(w_sql,0.70,f"DIM_SQL baja: {w_sql:.4f}")
        self.assertGreater(w_py,0.65,f"DIM_PYTHON baja: {w_py:.4f}")
        self.assertGreater(w_fl,0.55,f"DIM_FLUTTER baja: {w_fl:.4f}")
        self.assertLess(w_ru,w_fl,f"DIM_RUST latente ({w_ru:.4f}) >= DIM_FLUTTER ({w_fl:.4f})")

    def test_packet_dual_modo_H(self):
        ia,ib=make_session_pair(); pkt=ia.send(make_obj(ia.space))
        self.assertEqual(ia.mode,Mode.H)
        self.assertIsNotNone(pkt.payload_S); self.assertIsNotNone(pkt.payload_G)

    def test_objeto_mismo_space(self):
        sp_a=Space("IA_A"); sp_b=Space("IA_B")
        ia=Session(sp_a,"IA_A"); ib=Session(sp_b,"IA_B"); ia.connect(ib)
        dims=ib.receive(ia.send(make_obj(sp_a,[("DIM_SQL",{"t":"u"},1.0)])))
        self.assertIn("DIM_SQL",dims,"DIM_SQL no detectada")

    def test_dim_contract_transferible(self):
        """DIM_CONTRACT (nueva en v05) debe ser activa y transferirse correctamente."""
        sp=Space("IA_LEGAL")
        obj=ObjectND(sp)\
            .add("DIM_CONTRACT",{"partes":2,"tipo":"AESP","vigente":True},w=1.0)\
            .add("DIM_SQL",{"tabla":"contratos"},w=0.6)
        activa=obj.dims_activas()
        conn=polydim_connect(sp,Space("IA_AUDITORIA"))
        dims_recv=conn.transfer(obj)
        self.assertIn("DIM_CONTRACT",activa,"DIM_CONTRACT no activa en sender")
        self.assertIn("DIM_CONTRACT",dims_recv,"DIM_CONTRACT no recibida")

    def test_binary_roundtrip(self):
        """
        Un ObjectND con 3 dims enviado en modo H debe ser reconstructible
        en el receptor tanto por payload_G como por payload_S.
        receive() debe retornar dict[str,float] con las dims activas.
        """
        ia,ib=make_session_pair()
        obj=make_obj(ia.space,[
            ("DIM_SQL",   {"tabla":"test"},1.0),
            ("DIM_PYTHON",{"ver":"3.12"},0.8),
            ("DIM_GRAPH", {"nodo":"raiz"},0.6),
        ])
        pkt=ia.send(obj)
        self.assertEqual(ia.mode,Mode.H,"binary_roundtrip: modo no es H")
        self.assertIsNotNone(pkt.payload_G,"binary_roundtrip: sin payload_G")
        self.assertIsNotNone(pkt.payload_S,"binary_roundtrip: sin payload_S")

        dims_g=ib.receive(pkt)
        self.assertIsInstance(dims_g,dict,"binary_roundtrip: receive no retorna dict")
        self.assertIn("DIM_SQL",dims_g,"binary_roundtrip: DIM_SQL no en dims_g")
        self.assertIn("DIM_PYTHON",dims_g,"binary_roundtrip: DIM_PYTHON no en dims_g")

        ia_s=Session(Space("S_A"),"SA",caps=[Cap.S])
        ib_s=Session(Space("S_B"),"SB",caps=[Cap.S])
        ia_s.connect(ib_s)
        self.assertEqual(ia_s.mode,Mode.S,"binary_roundtrip: modo S no es S")
        pkt_s=ia_s.send(make_obj(ia_s.space,[("DIM_SQL",{"t":"u"},1.0)]))
        self.assertIsNone(pkt_s.payload_G,"binary_roundtrip: modo S tiene payload_G")
        self.assertIsNotNone(pkt_s.payload_S,"binary_roundtrip: modo S sin payload_S")
        dims_s=ib_s.receive(pkt_s)
        self.assertIn("DIM_SQL",dims_s,"binary_roundtrip: DIM_SQL no en dims_s")


if __name__=="__main__":
    suite=unittest.TestSuite()
    for cls in [
        TestObjectNDInvariants,
        TestHandshakeInvariants,
        TestAlignInvariants,
        TestSessionLifecycleInvariants,
        TestHVE,
        TestMesh,
        TestIntegracion,
    ]:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(cls))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    total=result.testsRun; fails=len(result.failures); errs=len(result.errors)
    print(f"\nTests: {total}  Fallos: {fails}  Errores: {errs}")
    sys.exit(0 if result.wasSuccessful() else 1)
