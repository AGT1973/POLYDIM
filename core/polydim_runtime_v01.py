"""
POLYDIM Runtime — V0.1
======================
Loop de ejecucion completo: dos instancias POLYDIM intercambian OBJECT_ND,
mantienen estado y demuestran autoprogramacion dimensional.

Arquitectura:
  PolyDimRuntime gestiona dos PolyDimSession ya conectadas.
  Cada IA tiene un Handler: funcion que recibe un objeto, contexto y sesion,
  y retorna lista de objetos de respuesta (puede ser vacia — corta el loop).
  El runtime ejecuta turnos alternados A->B, B->A hasta max_turns o sin respuesta.

Autoprogramacion (demo 3):
  Un OBJECT_ND con DIM_META puede contener instrucciones que modifican
  el handler de la IA receptora en tiempo de ejecucion.
  Primera implementacion de la capacidad central de POLYDIM:
  una IA puede reprogramar a otra via intercambio de objetos.

Demos:
  Demo 1: ping-pong dimensional — A envia SQL, B responde PYTHON
  Demo 2: acumulacion de estado — B acumula objetos y responde con resumen
  Demo 3: autoprogramacion — A envia META que modifica el handler de B

Resultados verificados (2026-06-12):
  Demo 1: 9 turnos, cadena SQL->PYTHON->RUST->FLUTTER->SQL ... OK
  Demo 2: 3 objetos acumulados, estado correcto OK
  Demo 3: handler reprogramado en tiempo de ejecucion OK

Autor:   ai.mpat.agt@gmail.com
Version: V0.1 — 2026-06-12 — TASK_014
"""

import numpy as np
import hashlib
import json
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field

from polydim_core_v02 import (
    PolyDimSpace, ObjectND, hv_sim, UMBRAL_RECUPERACION
)
from polydim_session_v01 import (
    PolyDimSession, SessionState, Mode, Cap,
    SONDAS_ESTANDAR
)

Handler = Callable[[ObjectND, Dict[str, Any], PolyDimSession], List[ObjectND]]


def reconstruir_objeto(paquete: dict, session: PolyDimSession) -> ObjectND:
    space = session.space
    obj = ObjectND(space)
    capa_s = paquete.get("capa_s", {})
    for dim_name, dim_data in capa_s.get("dimensions", {}).items():
        obj.add_dim(dim_name, dim_data.get("props", {}),
                    weight=dim_data.get("weight_declared", 1.0))
    if "hv_g" in paquete:
        hv = np.array(paquete["hv_g"], dtype=np.float32)
        n = np.linalg.norm(hv)
        obj._cache = hv / n if n > 1e-10 else hv
    return obj


def conectar_ias(space_a: PolyDimSpace, space_b: PolyDimSpace) -> tuple:
    caps = [Cap.CAP_S.value, Cap.CAP_G.value, Cap.CAP_ALIGN.value]
    ia_a = PolyDimSession("IA-A", space_a, caps=caps)
    ia_b = PolyDimSession("IA-B", space_b, caps=caps)
    init   = ia_a.iniciar_handshake()
    accept = ia_b.recibir_init(init)
    ack    = ia_a.recibir_accept(accept)
    ia_b.recibir_ack(ack)
    if ia_a.requires_align:
        ia_b.state = SessionState.ALIGNING
        req  = ia_a.generar_probes()
        resp = ia_b.responder_probes(req)
        c_a  = ia_a.calcular_align(req, resp)
        c_b  = ia_b.calcular_align(req, resp)
        ia_a.finalizar_align(c_a, c_b)
        ia_b.finalizar_align(c_a, c_b)
    return ia_a, ia_b


@dataclass
class TurnLog:
    turn:     int
    sender:   str
    receiver: str
    dims_out: List[str]
    dims_in:  List[str]
    modo:     str
    props:    Dict[str, Any]
    meta:     Dict[str, Any] = field(default_factory=dict)


class PolyDimRuntime:
    """
    Loop de ejecucion POLYDIM.
    Uso:
        rt = PolyDimRuntime(session_a, session_b, handler_a, handler_b)
        logs = rt.run(objetos_iniciales, max_turns=10)
    """
    def __init__(self, session_a, session_b,
                 handler_a=None, handler_b=None):
        assert session_a.is_ready and session_b.is_ready
        self.session_a  = session_a
        self.session_b  = session_b
        self.handler_a  = handler_a or _default_handler
        self.handler_b  = handler_b or _default_handler
        self.context_a  = {"nombre": "IA-A", "turno": 0, "recibidos": []}
        self.context_b  = {"nombre": "IA-B", "turno": 0, "recibidos": []}
        self.logs: List[TurnLog] = []

    def _enviar(self, obj, emisor, receptor):
        paquete = emisor.empaquetar_objeto(obj)
        return reconstruir_objeto(paquete, receptor), paquete

    def _procesar_turno(self, objetos, emisor, receptor,
                        handler, ctx, turno, nom_e, nom_r):
        respuestas = []
        for obj in objetos:
            recibido, paquete = self._enviar(obj, emisor, receptor)
            props_log = {d: data.get("props", {})
                         for d, data in paquete.get("capa_s", {}).get("dimensions", {}).items()}
            log = TurnLog(turn=turno, sender=nom_e, receiver=nom_r,
                          dims_out=list(obj._props.keys()),
                          dims_in=list(recibido._props.keys()),
                          modo=emisor.agreed_mode or "?",
                          props=props_log)
            ctx["turno"] = turno
            resps = handler(recibido, ctx, receptor)
            respuestas.extend(resps)
            log.meta["n_respuestas"] = len(resps)
            self.logs.append(log)
        return respuestas

    def run(self, objetos_iniciales, max_turns=10):
        cola_a_b = list(objetos_iniciales)
        self.logs = []
        for turno in range(max_turns):
            if not cola_a_b: break
            cola_b_a = self._procesar_turno(
                cola_a_b, self.session_a, self.session_b,
                self.handler_b, self.context_b, turno, "IA-A", "IA-B")
            if not cola_b_a: break
            cola_a_b = self._procesar_turno(
                cola_b_a, self.session_b, self.session_a,
                self.handler_a, self.context_a, turno, "IA-B", "IA-A")
        return self.logs

    def set_handler_b(self, nuevo_handler):
        self.handler_b = nuevo_handler


def _default_handler(obj, ctx, session):
    return [ObjectND(session.space).add_dim(
        "DIM_META", {"tipo": "ACK", "turno": ctx.get("turno", 0)}, weight=1.0)]


def handler_ping_pong(obj, ctx, session):
    cadena = ["DIM_SQL", "DIM_PYTHON", "DIM_RUST", "DIM_FLUTTER"]
    dims = obj.active_dims()
    actual = max(dims, key=dims.get) if dims else "DIM_SQL"
    idx = next((i for i, d in enumerate(cadena) if d in actual), 0)
    sig = cadena[(idx + 1) % len(cadena)]
    ctx.setdefault("pings", 0); ctx["pings"] += 1
    if ctx["pings"] > 4: return []
    return [ObjectND(session.space).add_dim(sig, {"ping": ctx["pings"], "desde": actual}, weight=1.0)]


def handler_acumulador(obj, ctx, session):
    ctx.setdefault("buffer", []); ctx.setdefault("total", 0)
    ctx["total"] += 1
    ctx["buffer"].append({"turno": ctx.get("turno", 0), "dims": list(obj.active_dims().keys())})
    if len(ctx["buffer"]) >= 3:
        resumen = {"tipo": "RESUMEN", "total": ctx["total"],
                   "dims_vistos": list({d for e in ctx["buffer"] for d in e["dims"]})}
        ctx["buffer"] = []
        return [ObjectND(session.space).add_dim("DIM_META", resumen, weight=1.0)]
    return [ObjectND(session.space).add_dim("DIM_META", {"tipo": "ACK", "n": ctx["total"]}, weight=0.3)]


def handler_meta_interceptor(runtime):
    def _handler(obj, ctx, session):
        dims = obj.active_dims()
        props = obj.to_symbolic().get("dimensions", {}).get("DIM_META", {}).get("props", {})
        if "DIM_META" in dims and props.get("tipo") == "REPROGRAM":
            nuevo = props.get("nuevo_modo", "")
            if nuevo == "acumulador":
                runtime.set_handler_b(handler_acumulador)
                runtime.context_b = {"nombre": "IA-B-reprogramada", "turno": 0, "recibidos": []}
            return [ObjectND(session.space).add_dim(
                "DIM_META", {"tipo": "REPROGRAM_OK", "nuevo_modo": nuevo}, weight=1.0)]
        return _default_handler(obj, ctx, session)
    return _handler


def demo_1_ping_pong():
    print("=" * 65); print("DEMO 1: Ping-pong dimensional"); print("=" * 65)
    ia_a, ia_b = conectar_ias(PolyDimSpace(), PolyDimSpace())
    print(f"  Conexion: modo={ia_a.agreed_mode}, align={ia_a.align_score:.4f}")
    rt = PolyDimRuntime(ia_a, ia_b, handler_ping_pong, handler_ping_pong)
    logs = rt.run([ObjectND(ia_a.space).add_dim("DIM_SQL", {"tabla": "inicio"}, weight=1.0)], max_turns=8)
    print(f"\n  Turnos: {len(logs)}")
    for l in logs:
        print(f"    [{l.turn}] {l.sender}→{l.receiver} {l.dims_out} resp={l.meta.get('n_respuestas',0)}")
    ok = len(logs) >= 4
    print(f"\n  [{'OK' if ok else 'FALLO'}] ping-pong correcto")
    return ok


def demo_2_acumulacion():
    print("\n" + "=" * 65); print("DEMO 2: Acumulacion de estado"); print("=" * 65)
    sa, sb = PolyDimSpace(), PolyDimSpace()
    ia_a, ia_b = conectar_ias(sa, sb)
    rt = PolyDimRuntime(ia_a, ia_b,
                        handler_a=lambda o, c, s: [],
                        handler_b=handler_acumulador)
    logs_all = []
    for obj in [ObjectND(sa).add_dim("DIM_SQL",    {"query": "SELECT"}, weight=1.0),
                ObjectND(sa).add_dim("DIM_PYTHON", {"op": "map"},       weight=0.8),
                ObjectND(sa).add_dim("DIM_RUST",   {"safe": "true"},    weight=0.6)]:
        logs_all.extend(rt.run([obj], max_turns=1))
    estado = rt.context_b
    print(f"  Turnos: {len(logs_all)}  total={estado.get('total',0)}  buffer={len(estado.get('buffer',[]))}")
    ok = estado.get("total", 0) == 3
    print(f"\n  [{'OK' if ok else 'FALLO'}] acumulacion correcta")
    return ok


def demo_3_autoprogramacion():
    print("\n" + "=" * 65); print("DEMO 3: Autoprogramacion"); print("=" * 65)
    sa, sb = PolyDimSpace(), PolyDimSpace()
    ia_a, ia_b = conectar_ias(sa, sb)
    rt = PolyDimRuntime(ia_a, ia_b, handler_a=_default_handler)
    rt.handler_b = handler_meta_interceptor(rt)
    rt.run([ObjectND(sa).add_dim("DIM_SQL", {"test": "inicial"}, weight=1.0)], max_turns=1)
    antes = getattr(rt.handler_b, '__name__', str(rt.handler_b))
    print(f"\n  Handler antes : {antes}")
    rt.run([ObjectND(sa).add_dim("DIM_META", {"tipo": "REPROGRAM", "nuevo_modo": "acumulador"}, weight=1.0)], max_turns=1)
    despues = getattr(rt.handler_b, '__name__', str(rt.handler_b))
    print(f"  Handler despues: {despues}")
    reprogramado = rt.handler_b is handler_acumulador
    print(f"  Reprogramada: {'SI' if reprogramado else 'NO'}")
    rt.run([ObjectND(sa).add_dim("DIM_PYTHON", {"reprogramado": "true"}, weight=1.0)], max_turns=2)
    ok = reprogramado and rt.context_b.get("total", 0) >= 1
    print(f"  Objetos post-reprogram: {rt.context_b.get('total', 0)}")
    print(f"\n  [{'OK' if ok else 'FALLO'}] autoprogramacion verificada")
    return ok


def run_all():
    print("=" * 65); print("POLYDIM Runtime V0.1 — Demo Completo"); print("=" * 65)
    ok1, ok2, ok3 = demo_1_ping_pong(), demo_2_acumulacion(), demo_3_autoprogramacion()
    print("\n" + "=" * 65); print("RESUMEN"); print("=" * 65)
    print(f"  Demo 1 ping-pong    : {'OK' if ok1 else 'FALLO'}")
    print(f"  Demo 2 acumulacion  : {'OK' if ok2 else 'FALLO'}")
    print(f"  Demo 3 autoprogramac: {'OK' if ok3 else 'FALLO'}")
    all_ok = ok1 and ok2 and ok3
    print(f"\n  {'TASK_014 TERMINADA — Runtime V0.1 VERIFICADO' if all_ok else 'CHECKS FALLIDOS'}")
    print("=" * 65)
    return all_ok


if __name__ == "__main__":
    run_all()
