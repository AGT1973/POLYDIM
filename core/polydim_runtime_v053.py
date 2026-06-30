# POLYDIM_DEST
# destination: polydim/core/
# filename:    polydim_runtime_v05.py
# author:      curso04.mithril@gmail.com

"""
POLYDIM Runtime — V0.5.3
=========================
Extiende V0.5.2 con BUG_003 / BUG_005 Python:

  BUG_005 — ObjectND: agregar API pública de lectura/escritura
    Síntoma: no había forma de inspeccionar dims/pesos desde fuera,
    ni de actualizar pesos sin re-crear el objeto completo.
    Fix — 4 métodos nuevos en ObjectND:
      get_dims()           → List[str]   — lista de dims declaradas
      get_weight(dim)      → Optional[float] — peso de una dim
      set_weight(dim, w)   → ObjectND    — actualiza peso, invalida cache
      invalidate_cache()   → ObjectND    — invalida cache del hv manualmente
    Todos encadenables: obj.set_weight("DIM_SQL", 0.5).set_weight("DIM_GRAPH", 0.8)
    Test nuevo: test_bug005_objectnd_public_api() — 6 sub-verificaciones. 14/14 OK.

  BUG_006 (Rust) — parcialmente en este PR: solo decode_hv_dynamic()
    El refactor completo de Space a Vec<f32> (N dinámico en construcción)
    es demasiado disruptivo para un solo PR — queda en RUST_001.
    Ver polydim_core.rs V0.3 (mismo commit) para decode_hv_dynamic().

  Hereda V0.5.2 (curso04.mithril):
    - geo_hv en to_symbolic() / receive() preserva geo_id en MODO_S
    - last_received_geo_id en Session

  Hereda V0.5.1 (polydim.ai.lenguage):
    - _align por peer
    - route_path reconstruye objeto en nodo intermedio
    - send(obj, target=peer_name)

  Compatibilidad: API V0.1→V0.5.2 intacta. 14/14 tests OK.

Autor:   curso04.mithril@gmail.com (BUG_005 sobre V0.5.2)
Version: V0.5.3 — 2026-06-19 — BUG_005 ObjectND public API
"""

from __future__ import annotations
import numpy as np, hashlib, math, uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any, Tuple

N = 10000
_SIGMA = 1.0 / (2.0 * math.sqrt(N))
UMBRAL = 0.5 + 2.0 * _SIGMA
CONTENT_W = 0.3
UMBRAL_ALIGN = 0.85
META_DEPTH_LIMIT = 1

NATIVE = ["DIM_PYTHON", "DIM_RUST", "DIM_FLUTTER", "DIM_SQL",
          "DIM_GRAPH", "DIM_VECTOR", "DIM_TIME", "DIM_ERROR", "DIM_META",
          "DIM_CONTRACT"]
SONDAS = NATIVE + ["entero", "flotante", "cadena", "lista", "diccionario",
                   "verdadero", "falso", "nulo", "error", "exito",
                   "crear", "leer", "actualizar", "borrar",
                   "usuario", "sesion", "permiso", "dato", "proceso"]

META_ACK       = "META_ACK"
META_REPROGRAM = "META_REPROGRAM"
META_RESET     = "META_RESET"
META_QUERY     = "META_QUERY"

DEFAULT_META_PERMS = [META_ACK]


# ──────────────────────────────────────────────────────────────────────────────
# ENUMS Y DATACLASSES
# ──────────────────────────────────────────────────────────────────────────────

class Mode(str, Enum):  S = "MODO_S"; G = "MODO_G"; H = "MODO_H"
class Cap(str, Enum):   S = "CAP_S";  G = "CAP_G";  ALIGN = "CAP_ALIGN"
class SessionState(str, Enum):
    IDLE = "IDLE"; CONNECTING = "CONNECTING"; NEGOTIATING = "NEGOTIATING"
    ALIGNING = "ALIGNING"; READY = "READY"; DEGRADED = "DEGRADED"
    FAILED = "FAILED"; CLOSED = "CLOSED"

@dataclass
class InitMsg:
    version:          str       = "POLYDIM_V1"
    sender_id:        str       = ""
    capabilities:     List[Cap] = field(default_factory=lambda: [Cap.S, Cap.G, Cap.ALIGN])
    preferred_mode:   Mode      = Mode.H
    N:                int       = N
    seed:             str       = "POLYDIM_V1_SEED_2026"
    nonce:            str       = field(default_factory=lambda: uuid.uuid4().hex[:8])
    ttl:              int       = 30
    meta_permissions: List[str] = field(default_factory=lambda: list(DEFAULT_META_PERMS))

@dataclass
class AcceptMsg:
    session_id:            str
    sender_id:             str
    agreed_mode:           Mode
    capabilities:          List[Cap]
    N:                     int
    nonce:                 str       = field(default_factory=lambda: uuid.uuid4().hex[:8])
    requires_align:        bool      = True
    negotiated_meta_perms: List[str] = field(default_factory=list)

@dataclass
class RejectMsg: reason: str; detail: str = ""

@dataclass
class AckMsg: session_id: str; ready: bool = True

@dataclass
class Packet:
    session_id: str
    seq:        int
    op:         str
    payload_S:  Optional[dict]       = None
    payload_G:  Optional[np.ndarray] = None
    intent:     List[str]            = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# VSA PRIMITIVAS
# ──────────────────────────────────────────────────────────────────────────────

def _bind(a, b):
    r = a * b; n = np.linalg.norm(r); return r / n if n > 1e-10 else r

def _sup(*hvs, ws=None):
    c = sum(w * h for w, h in zip(ws, hvs)) if ws else np.sum(hvs, axis=0)
    n = np.linalg.norm(c); return c / n if n > 1e-10 else c

def _proj(hv, sub): return (float(np.dot(hv, sub)) + 1.0) / 2.0
def _sim(a, b):     return float((np.dot(a, b) + 1.0) / 2.0)

def align_transform(hv, A, B):
    c = B.T @ (A @ hv); n = np.linalg.norm(c); return c / n if n > 1e-10 else c


# ──────────────────────────────────────────────────────────────────────────────
# SPACE
# ──────────────────────────────────────────────────────────────────────────────

class Space:
    def __init__(self, ps: str = ""):
        self.ps = ps
        self._s:   Dict[str, np.ndarray] = {}
        self._sub: Dict[str, np.ndarray] = {}
        for d in NATIVE:
            self._sub[d] = self._mk(d)

    def _mk(self, name: str) -> np.ndarray:
        k = f"{self.ps}:{name}" if self.ps else name
        s = int(hashlib.md5(k.encode()).hexdigest(), 16) % (2 ** 32)
        hv = np.random.default_rng(s).standard_normal(N).astype(np.float32)
        return hv / np.linalg.norm(hv)

    _SIGMA_NUM  = 0.0253
    _OMEGAS_NUM = np.random.default_rng(0xF0B1D1).normal(
                       0, _SIGMA_NUM, N // 2).astype(np.float32)

    def _enc_numerico(self, x: float) -> np.ndarray:
        k = N // 2
        hv = np.empty(N, dtype=np.float32)
        hv[:k] = np.cos(Space._OMEGAS_NUM * float(x))
        hv[k:] = np.sin(Space._OMEGAS_NUM * float(x))
        n = np.linalg.norm(hv)
        return hv / n if n > 1e-10 else hv

    def _enc_valor(self, v: Any) -> np.ndarray:
        if isinstance(v, bool):  return self.sym(str(v))
        if isinstance(v, (int, float)): return self._enc_numerico(float(v))
        return self.sym(str(v))

    def sym(self, n: str) -> np.ndarray:
        if n not in self._s: self._s[n] = self._mk(n)
        return self._s[n]

    def sub(self, n: str) -> np.ndarray:
        if n not in self._sub: self._sub[n] = self.sym(n)
        return self._sub[n]

    def _rnd(self) -> np.ndarray:
        hv = np.random.randn(N).astype(np.float32)
        return hv / np.linalg.norm(hv)

    def _enc(self, props: dict) -> np.ndarray:
        if not props: return self.sym("__empty__")
        return _sup(*[
            _bind(self.sym(str(k)), self._enc_valor(v))
            for k, v in props.items()
        ])


# ──────────────────────────────────────────────────────────────────────────────
# OBJECT_ND — V0.5.3: API pública (BUG_005)
# ──────────────────────────────────────────────────────────────────────────────

class ObjectND:
    """CRITICO: usar siempre el mismo Space que la Session que envia este objeto."""

    def __init__(self, space: Space = None):
        self._sp = space or Space()
        self._props: Dict[str, dict] = {}
        self._w:     Dict[str, float] = {}
        self._geo = self._sp._rnd()
        self._cache: Optional[np.ndarray] = None

    @property
    def geo_id(self) -> str:
        return hashlib.md5(self._geo.tobytes()).hexdigest()[:12]

    def add(self, dim: str, props: dict = None, w: float = 1.0) -> "ObjectND":
        self._props[dim] = props or {}
        self._w[dim] = float(np.clip(w, 0, 1))
        self._cache = None
        return self

    # ── BUG_005 fix (V0.5.3): API pública ─────────────────────────────────────

    def get_dims(self) -> List[str]:
        """Retorna lista de dimensiones declaradas en este objeto."""
        return list(self._props.keys())

    def get_weight(self, dim: str) -> Optional[float]:
        """Retorna el peso de una dimensión, o None si no está declarada."""
        return self._w.get(dim)

    def set_weight(self, dim: str, w: float) -> "ObjectND":
        """
        Actualiza el peso de una dimensión existente e invalida la cache.
        No agrega la dimensión si no existe — usar add() para eso.
        Encadenable: obj.set_weight("DIM_SQL", 0.5).set_weight("DIM_GRAPH", 0.8)
        """
        if dim in self._props:
            self._w[dim] = float(np.clip(w, 0, 1))
            self._cache = None
        return self

    def invalidate_cache(self) -> "ObjectND":
        """
        Invalida la cache del hipervector calculado.
        Necesario si se modifican subespacios del Space externamente.
        Encadenable.
        """
        self._cache = None
        return self

    # ──────────────────────────────────────────────────────────────────────────

    def _hv(self) -> np.ndarray:
        if self._cache is not None: return self._cache
        cs = [self._geo]; ws = [1.0]
        for d, p in self._props.items():
            ww = self._w.get(d, 1.0)
            if ww <= 0: continue
            cs.append(self._sp.sub(d)); ws.append(ww)
            cs.append(_bind(self._sp.sub(d), self._sp._enc(p))); ws.append(ww * CONTENT_W)
        self._cache = _sup(*cs, ws=ws)
        return self._cache

    def activacion(self, dim: str) -> float:
        return _proj(self._hv(), self._sp.sub(dim))

    def dims_activas(self, umbral: float = UMBRAL) -> Dict[str, float]:
        seen, r = {}, {}
        for d in NATIVE + list(self._props):
            if d not in seen:
                w = self.activacion(d)
                if w > umbral: r[d] = round(w, 4)
                seen[d] = 1
        return r

    def to_symbolic(self) -> dict:
        """V0.5.2: incluye geo_hv para preservar identidad geométrica en MODO_S."""
        return {
            "geo_id": self.geo_id,
            "geo_hv": self._geo.tolist(),
            "dims": {d: {"w": self._w.get(d, 1.0), "props": p}
                     for d, p in self._props.items()}
        }

    def __repr__(self) -> str:
        dims = ", ".join(f"{d}[{self._w.get(d, 1):.1f}]" for d in self._props)
        return f"ObjectND(id={self.geo_id}, dims=[{dims}])"


# ──────────────────────────────────────────────────────────────────────────────
# EMPAQUETAR
# ──────────────────────────────────────────────────────────────────────────────

def empaquetar_objeto(obj: ObjectND, mode: Mode,
                      A: np.ndarray = None, B: np.ndarray = None) -> Packet:
    sym    = obj.to_symbolic()
    intent = list(obj.dims_activas().keys())
    hv     = obj._hv()
    if A is not None and B is not None:
        hv = align_transform(hv, A, B)
    if mode == Mode.S: return Packet("", 0, "ND_SEND", payload_S=sym, intent=intent)
    if mode == Mode.G: return Packet("", 0, "ND_SEND", payload_G=hv, intent=intent)
    return Packet("", 0, "ND_SEND", payload_S=sym, payload_G=hv, intent=intent)


# ──────────────────────────────────────────────────────────────────────────────
# SESSION
# ──────────────────────────────────────────────────────────────────────────────

Handler = Callable[["ObjectND", Dict[str, Any], "Session"], List["ObjectND"]]

class Session:
    def __init__(self, space: Space, name: str = "IA",
                 caps: List[Cap] = None,
                 meta_permissions: List[str] = None,
                 handlers: Dict[str, Handler] = None):
        self.space   = space
        self.name    = name
        self.caps    = caps or [Cap.S, Cap.G, Cap.ALIGN]
        self.meta_permissions      = meta_permissions or list(DEFAULT_META_PERMS)
        self.negotiated_meta_perms: Set[str] = set(self.meta_permissions)
        self.handlers: Dict[str, Handler] = handlers or {}
        self.handler:  Optional[Handler] = None
        self.ctx:      Dict[str, Any] = {}
        self.state        = SessionState.IDLE
        self.mode         = Mode.S
        self.session_id:  Optional[str] = None
        self.align_score: Optional[float] = None
        self._align: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._seq = 0
        self._meta_depth = 0
        self.last_received_geo_id: Optional[str] = None
        self._peers: Dict[str, "Session"] = {}

    def handshake(self, remote: "Session") -> bool:
        self.state = SessionState.CONNECTING
        ia = InitMsg(sender_id=self.name, capabilities=self.caps,
                     meta_permissions=self.meta_permissions)
        ib = InitMsg(sender_id=remote.name, capabilities=remote.caps,
                     meta_permissions=remote.meta_permissions)
        caps_ambos = set(self.caps) & set(remote.caps)
        mode = Mode.H if Cap.G in caps_ambos else Mode.S
        sid = hashlib.md5((ia.nonce + ib.nonce).encode()).hexdigest()[:16]
        self.session_id = remote.session_id = sid
        self.mode       = remote.mode       = mode
        neg = set(ia.meta_permissions) & set(ib.meta_permissions)
        self.negotiated_meta_perms  = neg
        remote.negotiated_meta_perms = neg
        self.state  = SessionState.NEGOTIATING
        remote.state = SessionState.NEGOTIATING
        return True

    def align(self, remote: "Session", sondas: List[str] = None) -> float:
        self.state = remote.state = SessionState.ALIGNING
        p  = sondas or SONDAS
        A  = np.array([self.space.sym(s)   for s in p], dtype=np.float32)
        B  = np.array([remote.space.sym(s) for s in p], dtype=np.float32)
        scores = [_sim(align_transform(self.space.sub(d), A, B), remote.space.sub(d))
                  for d in NATIVE]
        score = float(np.mean(scores))
        valid = score >= UMBRAL_ALIGN
        if valid:
            self._align[remote.name]  = (A, B)
            remote._align[self.name] = (B, A)
            self.align_score  = remote.align_score = round(score, 4)
            self.state        = remote.state       = SessionState.READY
        else:
            self.mode  = remote.mode  = Mode.S
            self.state = remote.state = SessionState.DEGRADED
        return score

    def connect(self, remote: "Session") -> "Session":
        self.handshake(remote)
        if self.mode in (Mode.G, Mode.H):
            self.align(remote)
        self._peers[remote.name] = remote
        remote._peers[self.name] = self
        return self

    def connect_many(self, peers: List["Session"]) -> "Session":
        for peer in peers:
            if peer.name != self.name and peer.name not in self._peers:
                self.connect(peer)
        return self

    def send(self, obj: ObjectND, target: str = None) -> Packet:
        self._seq += 1
        A = B = None
        if target is not None:
            A, B = self._align.get(target, (None, None))
        elif len(self._align) == 1:
            A, B = next(iter(self._align.values()))
        elif len(self._align) > 1:
            raise ValueError(
                f"{self.name} tiene {len(self._align)} peers alineados "
                f"({list(self._align.keys())}); especifica target=.")
        pkt = empaquetar_objeto(obj, self.mode, A, B)
        pkt.session_id = self.session_id or ""
        pkt.seq = self._seq
        return pkt

    def receive(self, pkt: Packet) -> Dict[str, float]:
        if pkt.payload_G is not None:
            hv = pkt.payload_G
            self.last_received_geo_id = pkt.payload_S.get("geo_id") if pkt.payload_S else None
        elif pkt.payload_S:
            o = ObjectND(self.space)
            geo_hv_list = pkt.payload_S.get("geo_hv")
            if geo_hv_list is not None:
                try:
                    geo_hv = np.array(geo_hv_list, dtype=np.float32)
                    n = np.linalg.norm(geo_hv)
                    o._geo = geo_hv / n if n > 1e-10 else geo_hv
                    o._cache = None
                except (ValueError, TypeError):
                    pass
            for d, info in pkt.payload_S.get("dims", {}).items():
                o.add(d, info.get("props", {}), w=info.get("w", 1.0))
            self.last_received_geo_id = o.geo_id
            hv = o._hv()
        else:
            return {}

        dims = {d: round(_proj(hv, self.space.sub(d)), 4)
                for d in NATIVE if _proj(hv, self.space.sub(d)) > UMBRAL}

        if "DIM_META" in dims and pkt.payload_S and self._meta_depth < META_DEPTH_LIMIT:
            self._procesar_meta(pkt.payload_S)

        return dims

    def _procesar_meta(self, payload_s: dict):
        self._meta_depth += 1
        try:
            props = payload_s.get("dims", {}).get("DIM_META", {}).get("props", {})
            tipo  = props.get("tipo", "")
            perm_requerido = f"META_{tipo}" if tipo != "ACK" else META_ACK
            if perm_requerido not in self.negotiated_meta_perms: return
            if tipo == "REPROGRAM":
                nuevo_modo = props.get("nuevo_modo", "")
                if nuevo_modo in self.handlers:
                    self.handler = self.handlers[nuevo_modo]
                    self.ctx = {"nombre": f"{self.name}-{nuevo_modo}", "reprogramada_en": self._seq}
            elif tipo == "RESET":
                nivel = props.get("nivel", "contexto")
                if nivel == "contexto":  self.ctx = {}
                elif nivel == "handler": self.handler = None; self.ctx = {}
            elif tipo == "QUERY":
                campo = props.get("campo", "")
                resp = {"handler": self.handler.__name__ if self.handler else "default",
                        "contexto_keys": list(self.ctx.keys()),
                        "modo": self.mode.value,
                        "sesion_id": self.session_id}.get(campo, "desconocido")
                self.ctx[f"_query_{campo}"] = resp
        finally:
            self._meta_depth -= 1

    @property
    def info(self) -> dict:
        return {"session_id": self.session_id, "mode": self.mode.value,
                "state": self.state.value, "align_score": self.align_score,
                "meta_perms": sorted(self.negotiated_meta_perms),
                "peers": list(self._peers.keys()),
                "last_received_geo_id": self.last_received_geo_id}

    def __repr__(self) -> str:
        return f"Session({self.name},{self.state.value},{self.mode.value},peers={list(self._peers.keys())})"


# ──────────────────────────────────────────────────────────────────────────────
# CONNECTION — API V0.1 compatible
# ──────────────────────────────────────────────────────────────────────────────

class Connection:
    def __init__(self, sp_a: Space, sp_b: Space):
        self._ia = Session(sp_a, "IA_A")
        self._ib = Session(sp_b, "IA_B")
        self._ia.connect(self._ib)

    def transfer(self, obj: ObjectND) -> Dict[str, float]:
        return self._ib.receive(self._ia.send(obj))

    @property
    def info(self) -> dict: return self._ia.info

def polydim_connect(space_a: Space, space_b: Space) -> Connection:
    return Connection(space_a, space_b)


# ──────────────────────────────────────────────────────────────────────────────
# MESH
# ──────────────────────────────────────────────────────────────────────────────

class Mesh:
    def __init__(self):
        self._nodes: Dict[str, Session] = {}

    def add(self, name: str, space: Space,
            meta_permissions: List[str] = None,
            handlers: Dict[str, Handler] = None) -> Session:
        s = Session(space, name, meta_permissions=meta_permissions, handlers=handlers)
        self._nodes[name] = s
        return s

    def connect_all(self) -> "Mesh":
        names = list(self._nodes.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = self._nodes[names[i]]; b = self._nodes[names[j]]
                if names[j] not in a._peers: a.connect(b)
        return self

    def connect_hub(self, hub_name: str) -> "Mesh":
        hub = self._nodes[hub_name]
        hub.connect_many([s for n, s in self._nodes.items() if n != hub_name])
        return self

    def broadcast(self, obj: ObjectND, sender_name: str) -> Dict[str, Dict[str, float]]:
        sender = self._nodes[sender_name]
        return {pn: peer.receive(sender.send(obj, target=pn))
                for pn, peer in sender._peers.items()}

    def route(self, obj: ObjectND, sender_name: str, target_name: str) -> Dict[str, float]:
        sender = self._nodes[sender_name]; target = self._nodes[target_name]
        if target_name not in sender._peers:
            raise ValueError(f"{sender_name} y {target_name} no están conectados")
        return target.receive(sender.send(obj, target=target_name))

    def route_path(self, obj: ObjectND, path: List[str]) -> Dict[str, float]:
        if len(path) < 2: raise ValueError("path debe tener al menos 2 nodos")
        current = obj; result = {}
        for i in range(len(path) - 1):
            sn, tn = path[i], path[i + 1]
            sender = self._nodes[sn]; target = self._nodes[tn]
            if tn not in sender._peers: raise ValueError(f"{sn} y {tn} no están conectados")
            pkt = sender.send(current, target=tn)
            result = target.receive(pkt)
            if pkt.payload_S:
                current = ObjectND(target.space)
                geo_hv_list = pkt.payload_S.get("geo_hv")
                if geo_hv_list is not None:
                    try:
                        geo_hv = np.array(geo_hv_list, dtype=np.float32)
                        n = np.linalg.norm(geo_hv)
                        current._geo = geo_hv / n if n > 1e-10 else geo_hv
                        current._cache = None
                    except (ValueError, TypeError): pass
                for d, info in pkt.payload_S.get("dims", {}).items():
                    current.add(d, info.get("props", {}), w=info.get("w", 1.0))
        return result

    def topology(self) -> Dict[str, List[str]]:
        return {name: list(s._peers.keys()) for name, s in self._nodes.items()}

    def status(self) -> Dict[str, dict]:
        return {name: {"state": s.state.value, "mode": s.mode.value,
                       "align_score": s.align_score, "n_peers": len(s._peers)}
                for name, s in self._nodes.items()}

    def __len__(self): return len(self._nodes)
    def __repr__(self): return f"Mesh(n_nodes={len(self._nodes)}, nodes={list(self._nodes.keys())})"


# ──────────────────────────────────────────────────────────────────────────────
# TESTS — 14 (13 heredados + 1 nuevo BUG_005)
# ──────────────────────────────────────────────────────────────────────────────

def test_api_v01():
    sp = Space("MI_IA")
    obj = ObjectND(sp).add("DIM_SQL", {"tabla": "usuarios"}, w=1.0)
    conn = polydim_connect(sp, Space("OTRA_IA"))
    dims = conn.transfer(obj)
    return "DIM_SQL" in dims and conn.info["align_score"] >= 0.85

def test_meta_permissions():
    sa, sb = Space("A"), Space("B")
    def h(o, c, s): return []
    ia = Session(sa, "IA-A", meta_permissions=[META_ACK, META_REPROGRAM, META_QUERY])
    ib = Session(sb, "IA-B", meta_permissions=[META_ACK, META_RESET], handlers={"a": h})
    ia.connect(ib)
    ib.receive(ia.send(ObjectND(sa).add("DIM_META", {"tipo": "REPROGRAM", "nuevo_modo": "a"}, w=1.0)))
    return META_ACK in ib.negotiated_meta_perms and META_REPROGRAM not in ib.negotiated_meta_perms and ib.handler is None

def test_meta_permissions_con_permiso():
    sa, sb = Space("A"), Space("B")
    def h(o, c, s): return []
    ia = Session(sa, "IA-A", meta_permissions=[META_ACK, META_REPROGRAM])
    ib = Session(sb, "IA-B", meta_permissions=[META_ACK, META_REPROGRAM], handlers={"a": h})
    ia.connect(ib)
    ib.receive(ia.send(ObjectND(sa).add("DIM_META", {"tipo": "REPROGRAM", "nuevo_modo": "a"}, w=1.0)))
    return ib.handler is h

def test_meta_depth_limit():
    sa, sb = Space("A"), Space("B")
    def h(o, c, s): return []
    ia = Session(sa, "IA-A", meta_permissions=[META_ACK, META_REPROGRAM])
    ib = Session(sb, "IA-B", meta_permissions=[META_ACK, META_REPROGRAM], handlers={"x": h})
    ia.connect(ib); ib._meta_depth = 1
    ib.receive(ia.send(ObjectND(sa).add("DIM_META", {"tipo": "REPROGRAM", "nuevo_modo": "x"}, w=1.0)))
    bloqueado = ib.handler is None; ib._meta_depth = 0
    return bloqueado

def test_b2_numerico():
    sp = Space()
    h42, h43, h100 = sp._enc_numerico(42.0), sp._enc_numerico(43.0), sp._enc_numerico(100.0)
    oa = ObjectND(sp).add("DIM_SQL", {"n": 42}, w=1.0)
    ob = ObjectND(sp).add("DIM_SQL", {"n": 43}, w=1.0)
    oc = ObjectND(sp).add("DIM_SQL", {"n": 542}, w=1.0)
    return (_sim(h42, h43) > 0.99 and _sim(h42, h100) < _sim(h42, h43) and
            _sim(oa._hv(), ob._hv()) > _sim(oa._hv(), oc._hv()))

def test_dim_contract():
    sp = Space("IA_LEGAL")
    obj = ObjectND(sp).add("DIM_CONTRACT", {"partes": 2, "tipo": "AESP", "vigente": True}, w=1.0)\
                       .add("DIM_SQL", {"tabla": "contratos"}, w=0.6)
    dims = polydim_connect(sp, Space("IA_AUDITORIA")).transfer(obj)
    return "DIM_CONTRACT" in obj.dims_activas() and "DIM_CONTRACT" in dims

def test_mesh_n3_broadcast():
    mesh = Mesh()
    sp0, sp1, sp2 = Space("IA0"), Space("IA1"), Space("IA2")
    mesh.add("IA_0", sp0); mesh.add("IA_1", sp1); mesh.add("IA_2", sp2)
    mesh.connect_all()
    r = mesh.broadcast(ObjectND(sp0).add("DIM_SQL", {"t": "x"}, w=1.0), "IA_0")
    return "DIM_SQL" in r.get("IA_1", {}) and "DIM_SQL" in r.get("IA_2", {})

def test_mesh_n3_route():
    mesh = Mesh()
    sp0, sp1, sp2 = Space("R0"), Space("R1"), Space("R2")
    mesh.add("R_0", sp0); mesh.add("R_1", sp1); mesh.add("R_2", sp2)
    mesh.connect_all()
    return "DIM_GRAPH" in mesh.route(ObjectND(sp0).add("DIM_GRAPH", {"n": "o"}, w=1.0), "R_0", "R_2")

def test_mesh_n3_topology():
    mesh = Mesh()
    for i in range(3): mesh.add(f"T_{i}", Space(f"T{i}"))
    mesh.connect_all()
    return all(len(p) == 2 for p in mesh.topology().values())

def test_mesh_n5_all_ready():
    mesh = Mesh()
    for i in range(5): mesh.add(f"M_{i}", Space(f"M{i}"))
    mesh.connect_all()
    return all(v["state"] == "READY" for v in mesh.status().values())

def test_mesh_hub():
    mesh = Mesh()
    mesh.add("HUB", Space("HUB"))
    for i in range(3): mesh.add(f"W_{i}", Space(f"W{i}"))
    mesh.connect_hub("HUB")
    topo = mesh.topology()
    return set(topo["HUB"]) == {"W_0","W_1","W_2"} and all(set(topo[f"W_{i}"]) == {"HUB"} for i in range(3))

def test_mesh_route_path():
    mesh = Mesh()
    spa, spb, spc = Space("PA"), Space("PB"), Space("PC")
    mesh.add("A", spa); mesh.add("B", spb); mesh.add("C", spc)
    mesh.connect_all()
    return "DIM_TIME" in mesh.route_path(ObjectND(spa).add("DIM_TIME", {"orden": 1}, w=1.0), ["A","B","C"])

def test_bug002_geo_id_preserved_modo_s():
    sp_a, sp_b = Space("A"), Space("B")
    ia = Session(sp_a, "IA_A", caps=[Cap.S])
    ib = Session(sp_b, "IA_B", caps=[Cap.S])
    ia.connect(ib)
    obj = ObjectND(sp_a).add("DIM_SQL", {"tabla": "test"}, w=1.0)
    pkt = ia.send(obj)
    ib.receive(pkt)
    return ib.last_received_geo_id == obj.geo_id

def test_bug005_objectnd_public_api():
    """BUG_005 (V0.5.3): ObjectND API pública — 6 verificaciones."""
    sp  = Space("TEST")
    obj = ObjectND(sp).add("DIM_SQL", {"tabla": "t"}, w=1.0).add("DIM_GRAPH", {"n": "x"}, w=0.8)

    # 1. get_dims retorna las dims declaradas
    dims = obj.get_dims()
    ok1  = "DIM_SQL" in dims and "DIM_GRAPH" in dims

    # 2. get_weight retorna el peso correcto
    ok2  = abs(obj.get_weight("DIM_SQL") - 1.0) < 1e-6
    ok3  = abs(obj.get_weight("DIM_GRAPH") - 0.8) < 1e-6

    # 3. get_weight de dim no declarada retorna None
    ok4  = obj.get_weight("DIM_RUST") is None

    # 4. set_weight actualiza el peso e invalida cache
    _ = obj._hv()  # fuerza cache
    assert obj._cache is not None
    obj.set_weight("DIM_SQL", 0.5)
    ok5  = obj._cache is None and abs(obj.get_weight("DIM_SQL") - 0.5) < 1e-6

    # 5. invalidate_cache limpia la cache
    _ = obj._hv()
    obj.invalidate_cache()
    ok6  = obj._cache is None

    # 6. set_weight en dim no declarada no hace nada (no crea dim nueva)
    obj.set_weight("DIM_RUST", 0.9)
    ok7  = obj.get_weight("DIM_RUST") is None and "DIM_RUST" not in obj.get_dims()

    return ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7


if __name__ == "__main__":
    tests = [
        test_api_v01,
        test_meta_permissions,
        test_meta_permissions_con_permiso,
        test_meta_depth_limit,
        test_b2_numerico,
        test_dim_contract,
        test_mesh_n3_broadcast,
        test_mesh_n3_route,
        test_mesh_n3_topology,
        test_mesh_n5_all_ready,
        test_mesh_hub,
        test_mesh_route_path,
        test_bug002_geo_id_preserved_modo_s,
        test_bug005_objectnd_public_api,
    ]
    results = [t() for t in tests]
    for t, r in zip(tests, results):
        print(f"  {t.__name__:48s}: {'OK' if r else 'FALLO'}")
    passed = sum(results)
    print(f"\n  {passed}/{len(results)} tests OK — {'V0.5.3 VERIFICADA' if passed == len(results) else 'CHECKS FALLIDOS'}")
