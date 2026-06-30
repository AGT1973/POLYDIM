# POLYDIM_DEST
# destination: polydim/core/
# filename:    polydim_runtime_v09.py
# author:      ai.mpat.agt@gmail.com

"""
POLYDIM Runtime -- V0.9
========================
Base: polydim_runtime_v08.py (V0.8, 19/19 tests, PCRG-1 + Semantic Backend)
Fix:  TASK_031 -- separar to_symbolic() del core (PolydimdDebugger externo)
Tests: 21/21 (19 de v08 + 2 nuevos de v09)

CAMBIO V0.9 sobre V0.8:

  TASK_031 -- to_symbolic() vivia en el core con un nombre que sugeria
  "vista de debugging", pero en realidad ES el formato de TRANSMISION
  real (payload_S), usado por empaquetar_objeto()/Session.receive()
  en MODO_S (BUG_002: incluye geo_hv_b64 para preservar geo_id a traves
  de la red). POLYDIM_BASES_V1.md (ANTI_4) identifica esto como
  anti-patron: el core no deberia depender de -- ni nombrarse como --
  una representacion pensada para ojos humanos.

  FIX:
    1. ObjectND.to_symbolic() -> renombrado a to_wire_dict(). Mismo
       dict exacto (geo_id, geo_hv_b64, dims) -- el cambio es de
       NOMBRE, para que diga lo que realmente es: formato de red.
    2. to_symbolic() queda como alias DEPRECADO (emite
       DeprecationWarning) para no romper codigo existente.
    3. empaquetar_objeto() usa to_wire_dict() internamente.
    4. PolydimdDebugger (modulo separado, polydim_debugger.py) provee
       inspect(obj)/trace(session)/visualize_topology(mesh) -- pensados
       PARA HUMANOS, viven FUERA del core, sin depender de
       to_wire_dict/to_symbolic salvo para los datos que de verdad
       necesitan mostrar (usan get_dims(), dims_activas(), session.info,
       mesh.topology() -- todas APIs publicas ya existentes).

  LA TENSION (documentada, no oculta): BUG_002 necesita ALGUN formato
  serializable para transmitir geo_hv en MODO_S -- eso no puede "salir
  del core" sin romper la transmision misma. Lo que SI salio del core
  es la pretension de que ese formato es una "vista simbolica para
  debugging". Ver polydim_debugger.py para el modulo de inspeccion real.

HEREDADO DE V0.8 (sin cambios):
  · SemanticBackend, MockSemanticBackend, MiniLMBackend, FastTextBackend
  · PCRG-1 en Space._mk() (TASK_037)
  · BUG_002, BUG_003, BUG_004, BUG_005 (V0.6)
  · Mesh multi-agente, DIM_CONTRACT, B2 numerico, META_PERMISSIONS (V0.5)

Autores: ai.mpat.agt@gmail.com (V0.9 -- separar debugger del core)
         polydim.ai.lenguage@gmail.com (V0.5 Mesh original)
Versión: V0.9 -- 2026-06-20 -- TASK_031
"""

from __future__ import annotations
import base64
import math
import numpy as np
import hashlib, uuid
import warnings
from abc import ABC, abstractmethod
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
          "DIM_GRAPH",  "DIM_VECTOR", "DIM_TIME", "DIM_ERROR",
          "DIM_META",   "DIM_CONTRACT"]

SONDAS = NATIVE + ["entero", "flotante", "cadena", "lista", "diccionario",
                   "verdadero", "falso", "nulo", "error", "exito",
                   "crear", "leer", "actualizar", "borrar",
                   "usuario", "sesion", "permiso", "dato", "proceso"]

META_ACK       = "META_ACK"
META_REPROGAM  = "META_REPROGAM"
META_RESET     = "META_RESET"
META_QUERY     = "META_QUERY"
DEFAULT_META_PERMS = [META_ACK]


# -----------------------------------------------------------------
# PCRG-1 -- POLYDIM Canonical Random Generator (idéntico a V0.7/V0.8)
# IDÉNTICO a polydim_rng.py y mod pcrg en polydim_core.rs (Rust)
# -----------------------------------------------------------------

_FNV_OFFSET:  int   = 14695981039346656037
_FNV_PRIME:   int   = 1099511628211
_U64_MASK:    int   = 0xFFFFFFFF_FFFFFFFF
_U64_DENOM:   float = float(0xFFFFFFFF_FFFFFFFF) + 1.0
_PCRG_STATE_SALT: int = 0xDEADBEEF_CAFEBABE


def _fnv1a_64(data: bytes, seed: int = _FNV_OFFSET) -> int:
    h = seed
    for b in data:
        h ^= b
        h = (h * _FNV_PRIME) & _U64_MASK
    return h


def _pcrg_init(key: str) -> list:
    kb = key.encode('utf-8')
    s0 = _fnv1a_64(kb)
    s1 = _fnv1a_64(kb, (s0 ^ _PCRG_STATE_SALT) & _U64_MASK)
    return [s0 if s0 else 1, s1 if s1 else 2]


def _pcrg_next(state: list) -> int:
    s1 = state[0]
    s0 = state[1]
    state[0] = s0
    s1 ^= (s1 << 23) & _U64_MASK
    s1 ^= (s1 >> 17)
    s1 ^= s0
    s1 ^= (s0 >> 26)
    state[1] = s1
    return (state[0] + state[1]) & _U64_MASK


def _pcrg_uniform(state: list) -> float:
    return (_pcrg_next(state) + 0.5) / _U64_DENOM


def _pcrg_gaussian_pair(state: list) -> tuple:
    u1 = _pcrg_uniform(state)
    u2 = _pcrg_uniform(state)
    r     = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    return r * math.cos(theta), r * math.sin(theta)


def pcrg_make_hv(key: str, n: int = N) -> np.ndarray:
    """Genera un hipervector float32 normalizado vía PCRG-1 (determinístico)."""
    state  = _pcrg_init(key)
    values = []
    while len(values) < n:
        a, b = _pcrg_gaussian_pair(state)
        values.append(a)
        if len(values) < n:
            values.append(b)
    arr  = np.array(values, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 1e-10 else arr


def pcrg_seed(key: str) -> int:
    return _fnv1a_64(key.encode('utf-8'))


# -----------------------------------------------------------------
# Enums y dataclasses
# -----------------------------------------------------------------

class Mode(str, Enum):  S = "MODO_S"; G = "MODO_G"; H = "MODO_H"
class Cap(str, Enum):   S = "CAP_S";  G = "CAP_G";  ALIGN = "CAP_ALIGN"
class SessionState(str, Enum):
    IDLE="IDLE"; CONNECTING="CONNECTING"; NEGOTIATING="NEGOTIATING"
    ALIGNING="ALIGNING"; READY="READY"; DEGRADED="DEGRADED"
    FAILED="FAILED"; CLOSED="CLOSED"

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
    session_id:              str
    sender_id:               str
    agreed_mode:             Mode
    capabilities:            List[Cap]
    N:                       int
    nonce:                   str       = field(default_factory=lambda: uuid.uuid4().hex[:8])
    requires_align:          bool      = True
    negotiated_meta_perms:   List[str] = field(default_factory=list)

@dataclass
class RejectMsg: reason: str; detail: str = ""

@dataclass
class AckMsg: session_id: str; ready: bool = True

@dataclass
class Packet:
    session_id: str
    seq:        int
    op:         str
    payload_S:  Optional[dict]         = None
    payload_G:  Optional[np.ndarray]   = None
    intent:     List[str]              = field(default_factory=list)


# -----------------------------------------------------------------
# VSA primitivas
# -----------------------------------------------------------------

def _bind(a, b):
    r = a * b; n = np.linalg.norm(r); return r / n if n > 1e-10 else r

def _sup(*hvs, ws=None):
    c = sum(w * h for w, h in zip(ws, hvs)) if ws else np.sum(hvs, axis=0)
    n = np.linalg.norm(c); return c / n if n > 1e-10 else c

def _proj(hv, sub): return (float(np.dot(hv, sub)) + 1.0) / 2.0
def _sim(a, b):     return float((np.dot(a, b) + 1.0) / 2.0)


def align_transform(hv: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """BUG_003 (V0.6) -- Residual-Preserving Alignment (RPA). Sin cambios."""
    coords   = A @ hv
    aligned  = B.T @ coords
    residual = hv - A.T @ coords
    c = aligned + residual
    n = np.linalg.norm(c)
    return c / n if n > 1e-10 else c


# -----------------------------------------------------------------
# Backends semánticos (idénticos a V0.8)
# -----------------------------------------------------------------

class SemanticBackend(ABC):
    dim: int = 384
    @abstractmethod
    def encode(self, text: str) -> np.ndarray: ...


class MockSemanticBackend(SemanticBackend):
    dim = 64
    GRUPOS = {
        "identidad":  ["usuario","cliente","persona","account","user","perfil","sesion"],
        "datos":      ["tabla","columna","fila","registro","dato","database","sql"],
        "interfaz":   ["widget","formulario","Form","TextField","boton","pantalla","vista"],
        "memoria":    ["struct","ownership","lifetime","heap","stack","rust","pointer"],
        "logica":     ["dict","list","funcion","clase","modulo","python","analisis","script"],
        "tiempo":     ["evento","timestamp","secuencia","orden","cola","stream","time"],
        "error":      ["error","excepcion","falla","timeout","retry","panic","crash"],
        "red":        ["protocolo","mensaje","socket","http","api","endpoint","request"],
        "seguridad":  ["permiso","auth","token","cifrado","firma","certificado","clave"],
    }

    def __init__(self):
        self._g = {}
        for g in self.GRUPOS:
            self._g[g] = pcrg_make_hv(f"G:{g}", self.dim)

    def _grupo(self, t: str) -> Optional[str]:
        for g, ms in self.GRUPOS.items():
            if any(t.lower() == m.lower() for m in ms):
                return g
        return None

    def encode(self, text: str) -> np.ndarray:
        base = pcrg_make_hv(f"SEM:{text}", self.dim)
        g = self._grupo(text)
        if g:
            hv = 0.4 * base + 0.6 * self._g[g]
            n  = np.linalg.norm(hv)
            return hv / n if n > 1e-10 else hv
        return base


class MiniLMBackend(SemanticBackend):
    dim = 384

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            raise ImportError("Instalar: pip install sentence-transformers")

    def encode(self, text: str) -> np.ndarray:
        return self._model.encode(text, normalize_embeddings=True).astype(np.float32)


class FastTextBackend(SemanticBackend):
    dim = 300

    def __init__(self, model_path: str):
        import fasttext
        self._model = fasttext.load_model(model_path)

    def encode(self, text: str) -> np.ndarray:
        hv = self._model.get_word_vector(text).astype(np.float32)
        n  = np.linalg.norm(hv)
        return hv / n if n > 1e-10 else hv


def make_jl(d_in: int, d_out: int = N) -> np.ndarray:
    """
    Matriz Johnson-Lindenstrauss determinística: d_in -> d_out.
    NOTA DE DISEÑO (TASK_027): mantenida con MD5+PCG64 intencionalmente --
    es maquinaria interna del backend semántico, no un subespacio nombrado.
    """
    s = int(hashlib.md5(f"JL_{d_out}_{d_in}".encode()).hexdigest(), 16) % (2 ** 32)
    R = np.random.default_rng(s).standard_normal((d_out, d_in)).astype(np.float32)
    return R / math.sqrt(d_in)


# -----------------------------------------------------------------
# Space (idéntico a V0.8)
# -----------------------------------------------------------------

class Space:
    def __init__(self, ps: str = "", semantic_backend: SemanticBackend = None):
        self.ps      = ps
        self.backend = semantic_backend
        self._s:   Dict[str, np.ndarray] = {}
        self._sub: Dict[str, np.ndarray] = {}
        self._JL = make_jl(semantic_backend.dim) if semantic_backend else None
        for d in NATIVE:
            self._sub[d] = self._mk(d)

    def _mk(self, name: str) -> np.ndarray:
        if self.backend:
            hv = self._JL @ self.backend.encode(name)
            if self.ps:
                key = f"{self.ps}:{name}"
                p   = pcrg_make_hv(key, N)
                hv  = 0.85 * hv + 0.15 * p
        else:
            key = f"{self.ps}:{name}" if self.ps else name
            hv  = pcrg_make_hv(key, N)
        n = np.linalg.norm(hv)
        return (hv / n if n > 1e-10 else hv).astype(np.float32)

    _SIGMA_NUM  = 0.0253
    _OMEGAS_NUM = np.random.default_rng(0xF0B1D1).normal(
                      0, _SIGMA_NUM, N // 2).astype(np.float32)

    def _enc_numerico(self, x: float) -> np.ndarray:
        k  = N // 2
        hv = np.empty(N, dtype=np.float32)
        hv[:k] = np.cos(Space._OMEGAS_NUM * float(x))
        hv[k:] = np.sin(Space._OMEGAS_NUM * float(x))
        n = np.linalg.norm(hv)
        return hv / n if n > 1e-10 else hv

    def _enc_valor(self, v: Any) -> np.ndarray:
        if isinstance(v, bool):            return self.sym(str(v))
        if isinstance(v, (int, float)):    return self._enc_numerico(float(v))
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
        if not props:
            return self.sym("__empty__")
        return _sup(*[_bind(self.sym(str(k)), self._enc_valor(v))
                      for k, v in props.items()])


# -----------------------------------------------------------------
# ObjectND -- TASK_031: to_symbolic() RENOMBRADO a to_wire_dict()
# -----------------------------------------------------------------

class ObjectND:
    """CRITICO: usar siempre el mismo Space que la Session que envía este objeto."""

    def __init__(self, space: Space = None):
        self._sp     = space or Space()
        self._props: Dict[str, dict]  = {}
        self._w:     Dict[str, float] = {}
        self._geo    = self._sp._rnd()
        self._cache: Optional[np.ndarray] = None

    @property
    def geo_id(self) -> str:
        return hashlib.md5(self._geo.tobytes()).hexdigest()[:12]

    def add(self, dim: str, props: dict = None, w: float = 1.0) -> "ObjectND":
        self._props[dim] = props or {}
        self._w[dim]     = float(np.clip(w, 0, 1))
        self._cache      = None
        return self

    def _hv(self) -> np.ndarray:
        if self._cache is not None: return self._cache
        cs = [self._geo]; ws = [1.0]
        for d, p in self._props.items():
            ww = self._w.get(d, 1.0)
            if ww <= 0: continue
            cs.append(self._sp.sub(d));                                ws.append(ww)
            cs.append(_bind(self._sp.sub(d), self._sp._enc(p)));      ws.append(ww * CONTENT_W)
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

    def to_wire_dict(self) -> dict:
        """
        TASK_031: este es el FORMATO DE TRANSMISION real (payload_S),
        no una representacion de debugging. Vive en el core porque
        empaquetar_objeto()/Session.receive() lo necesitan para
        construir/leer paquetes en MODO_S (BUG_002: incluye geo_hv_b64
        para preservar geo_id a traves de la red).

        Para INSPECCIONAR un objeto de forma legible para humanos,
        usar PolydimdDebugger.inspect(obj) (modulo separado,
        polydim_debugger.py) -- no este metodo.
        """
        return {
            "geo_id":     self.geo_id,
            "geo_hv_b64": base64.b64encode(self._geo.tobytes()).decode('ascii'),
            "dims": {d: {"w": self._w.get(d, 1.0), "props": p}
                     for d, p in self._props.items()}
        }

    def to_symbolic(self) -> dict:
        """
        DEPRECADO (TASK_031): alias de to_wire_dict(). El nombre
        "to_symbolic" sugeria que esto era una vista de debugging --
        no lo es, es el formato de transmision real. Usar to_wire_dict()
        en codigo nuevo. Para inspeccion humana, usar PolydimdDebugger.
        """
        warnings.warn(
            "ObjectND.to_symbolic() esta deprecado -- usar to_wire_dict() "
            "para transmision, o PolydimdDebugger.inspect(obj) para "
            "inspeccion humana (polydim_debugger.py).",
            DeprecationWarning, stacklevel=2
        )
        return self.to_wire_dict()

    def get_dims(self) -> Dict[str, dict]:
        return {d: {"props": p, "w": self._w.get(d, 1.0)}
                for d, p in self._props.items()}

    def get_weight(self, dim: str) -> float:
        return self._w.get(dim, 1.0)

    def set_weight(self, dim: str, w: float) -> None:
        self._w[dim] = float(np.clip(w, 0, 1))
        self._cache  = None

    def invalidate_cache(self) -> None:
        self._cache = None

    def hv(self) -> np.ndarray:
        return self._hv()

    def __repr__(self) -> str:
        dims = ", ".join(f"{d}[{self._w.get(d,1):.1f}]" for d in self._props)
        return f"ObjectND(id={self.geo_id}, dims=[{dims}])"


# -----------------------------------------------------------------
# empaquetar_objeto -- usa to_wire_dict() (TASK_031)
# -----------------------------------------------------------------

def empaquetar_objeto(obj: ObjectND, mode: Mode,
                      A: np.ndarray = None, B: np.ndarray = None) -> Packet:
    wire   = obj.to_wire_dict()
    intent = list(obj.dims_activas().keys())
    hv     = obj._hv()
    if A is not None and B is not None:
        hv = align_transform(hv, A, B)
    if mode == Mode.S: return Packet("", 0, "ND_SEND", payload_S=wire, intent=intent)
    if mode == Mode.G: return Packet("", 0, "ND_SEND", payload_G=hv,  intent=intent)
    return Packet("", 0, "ND_SEND", payload_S=wire, payload_G=hv, intent=intent)


# -----------------------------------------------------------------
# Session (idéntico a V0.8, salvo empaquetar_objeto interno)
# -----------------------------------------------------------------

Handler = Callable[["ObjectND", Dict[str, Any], "Session"], List["ObjectND"]]

class Session:
    def __init__(self, space: Space, name: str = "IA",
                 caps: List[Cap] = None,
                 meta_permissions: List[str] = None,
                 handlers: Dict[str, Handler] = None):
        self.space   = space
        self.name    = name
        self.caps    = caps or [Cap.S, Cap.G, Cap.ALIGN]
        self.meta_permissions          = meta_permissions or list(DEFAULT_META_PERMS)
        self.negotiated_meta_perms: Set[str] = set(self.meta_permissions)
        self.handlers: Dict[str, Handler] = handlers or {}
        self.handler:  Optional[Handler]  = None
        self.ctx:      Dict[str, Any]     = {}

        self.state        = SessionState.IDLE
        self.mode         = Mode.S
        self.session_id:  Optional[str]   = None
        self.align_score: Optional[float] = None
        self._align: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._seq   = 0
        self._meta_depth = 0
        self._peers: Dict[str, "Session"] = {}

    def handshake(self, remote: "Session") -> bool:
        self.state = SessionState.CONNECTING
        ia = InitMsg(sender_id=self.name,   capabilities=self.caps,
                     meta_permissions=self.meta_permissions)
        ib = InitMsg(sender_id=remote.name, capabilities=remote.caps,
                     meta_permissions=remote.meta_permissions)

        caps_ambos = set(self.caps) & set(remote.caps)
        mode       = Mode.H if Cap.G in caps_ambos else Mode.S
        sid        = hashlib.md5((ia.nonce + ib.nonce).encode()).hexdigest()[:16]

        self.session_id  = remote.session_id = sid
        self.mode        = remote.mode       = mode

        neg = set(ia.meta_permissions) & set(ib.meta_permissions)
        self.negotiated_meta_perms   = neg
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
            remote._align[self.name]  = (B, A)
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
                f"({list(self._align.keys())}); especifica target= para "
                "elegir la transformacion correcta (ver Mesh.broadcast/route)."
            )
        pkt = empaquetar_objeto(obj, self.mode, A, B)
        pkt.session_id = self.session_id or ""
        pkt.seq        = self._seq
        return pkt

    def receive(self, pkt: Packet) -> Dict[str, float]:
        if pkt.payload_G is not None:
            hv = pkt.payload_G
        elif pkt.payload_S:
            o = ObjectND(self.space)
            geo_b64 = pkt.payload_S.get("geo_hv_b64")
            if geo_b64:
                geo_bytes = base64.b64decode(geo_b64)
                o._geo    = np.frombuffer(geo_bytes, dtype=np.float32).copy()
                o._cache  = None
            for d, info in pkt.payload_S.get("dims", {}).items():
                o.add(d, info.get("props", {}), w=info.get("w", 1.0))
            hv = o._hv()
        else:
            return {}

        dims = {d: round(_proj(hv, self.space.sub(d)), 4)
                for d in NATIVE if _proj(hv, self.space.sub(d)) > UMBRAL}

        if ("DIM_META" in dims and pkt.payload_S
                and self._meta_depth < META_DEPTH_LIMIT):
            self._procesar_meta(pkt.payload_S)

        return dims

    def _procesar_meta(self, payload_s: dict):
        self._meta_depth += 1
        try:
            props = payload_s.get("dims", {}).get("DIM_META", {}).get("props", {})
            tipo  = props.get("tipo", "")

            perm_requerido = f"META_{tipo}" if tipo != "ACK" else META_ACK
            if perm_requerido not in self.negotiated_meta_perms:
                return

            if tipo == "REPROGAM":
                nuevo_modo = props.get("nuevo_modo", "")
                if nuevo_modo in self.handlers:
                    self.handler = self.handlers[nuevo_modo]
                    self.ctx = {"nombre": f"{self.name}-{nuevo_modo}",
                                "reprogramada_en": self._seq}

            elif tipo == "RESET":
                nivel = props.get("nivel", "contexto")
                if nivel == "contexto":    self.ctx = {}
                elif nivel == "handler":   self.handler = None; self.ctx = {}

            elif tipo == "QUERY":
                campo = props.get("campo", "")
                resp  = {
                    "handler":       self.handler.__name__ if self.handler else "default",
                    "contexto_keys": list(self.ctx.keys()),
                    "modo":          self.mode.value,
                    "sesion_id":     self.session_id,
                }.get(campo, "desconocido")
                self.ctx[f"_query_{campo}"] = resp

        finally:
            self._meta_depth -= 1

    @property
    def info(self) -> dict:
        return {"session_id": self.session_id, "mode": self.mode.value,
                "state": self.state.value, "align_score": self.align_score,
                "meta_perms": sorted(self.negotiated_meta_perms),
                "peers": list(self._peers.keys())}

    def __repr__(self) -> str:
        return (f"Session({self.name},{self.state.value},{self.mode.value},"
                f"peers={list(self._peers.keys())})")


# -----------------------------------------------------------------
# Connection -- API V0.1 compatible
# -----------------------------------------------------------------

class Connection:
    def __init__(self, sp_a: Space, sp_b: Space):
        self._ia = Session(sp_a, "IA_A")
        self._ib = Session(sp_b, "IA_B")
        self._ia.connect(self._ib)

    def transfer(self, obj: ObjectND) -> Dict[str, float]:
        return self._ib.receive(self._ia.send(obj))

    @property
    def info(self) -> dict: return self._ia.info

    def __repr__(self) -> str: return f"Connection({self._ia.info})"


def polydim_connect(space_a: Space, space_b: Space) -> Connection:
    return Connection(space_a, space_b)


# -----------------------------------------------------------------
# Mesh (idéntico a V0.8)
# -----------------------------------------------------------------

class Mesh:
    """Malla plana de N Sessions POLYDIM."""

    def __init__(self):
        self._nodes: Dict[str, Session] = {}

    def add(self, name: str, space: Space,
            meta_permissions: List[str] = None,
            handlers: Dict[str, Handler] = None) -> Session:
        s = Session(space, name,
                    meta_permissions=meta_permissions,
                    handlers=handlers)
        self._nodes[name] = s
        return s

    def connect_all(self) -> "Mesh":
        names = list(self._nodes.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = self._nodes[names[i]]
                b = self._nodes[names[j]]
                if names[j] not in a._peers:
                    a.connect(b)
        return self

    def connect_hub(self, hub_name: str) -> "Mesh":
        hub    = self._nodes[hub_name]
        others = [s for n, s in self._nodes.items() if n != hub_name]
        hub.connect_many(others)
        return self

    def broadcast(self, obj: ObjectND, sender_name: str) -> Dict[str, Dict[str, float]]:
        sender  = self._nodes[sender_name]
        results = {}
        for peer_name, peer in sender._peers.items():
            pkt = sender.send(obj, target=peer_name)
            results[peer_name] = peer.receive(pkt)
        return results

    def route(self, obj: ObjectND, sender_name: str, target_name: str) -> Dict[str, float]:
        sender = self._nodes[sender_name]
        target = self._nodes[target_name]
        if target_name not in sender._peers:
            raise ValueError(f"{sender_name} y {target_name} no están conectados")
        return target.receive(sender.send(obj, target=target_name))

    def route_path(self, obj: ObjectND, path: List[str]) -> Dict[str, float]:
        if len(path) < 2:
            raise ValueError("path debe tener al menos 2 nodos")
        current = obj
        result  = {}
        for i in range(len(path) - 1):
            sender_name, target_name = path[i], path[i + 1]
            sender = self._nodes[sender_name]
            target = self._nodes[target_name]
            if target_name not in sender._peers:
                raise ValueError(f"{sender_name} y {target_name} no están conectados")
            pkt    = sender.send(current, target=target_name)
            result = target.receive(pkt)
            if pkt.payload_S:
                current = ObjectND(target.space)
                for d, info in pkt.payload_S.get("dims", {}).items():
                    current.add(d, info.get("props", {}), w=info.get("w", 1.0))
        return result

    def topology(self) -> Dict[str, List[str]]:
        return {name: list(s._peers.keys()) for name, s in self._nodes.items()}

    def status(self) -> Dict[str, dict]:
        return {name: {"state": s.state.value, "mode": s.mode.value,
                       "align_score": s.align_score, "n_peers": len(s._peers)}
                for name, s in self._nodes.items()}

    def __len__(self) -> int: return len(self._nodes)
    def __repr__(self) -> str:
        return f"Mesh(n_nodes={len(self._nodes)}, nodes={list(self._nodes.keys())})"


# -----------------------------------------------------------------
# Tests -- 19 heredados de v08 + 2 nuevos de v09 (TASK_031)
# -----------------------------------------------------------------

def test_api_v01():
    sp  = Space("MI_IA")
    obj = ObjectND(sp).add("DIM_SQL", {"tabla": "usuarios"}, w=1.0)
    conn = polydim_connect(sp, Space("OTRA_IA"))
    dims = conn.transfer(obj)
    return "DIM_SQL" in dims and conn.info["align_score"] >= 0.85


def test_meta_permissions():
    sa, sb = Space("A"), Space("B")
    def handler_acum(obj, ctx, session): return []
    ia = Session(sa, "IA-A", meta_permissions=[META_ACK, META_REPROGAM, META_QUERY])
    ib = Session(sb, "IA-B", meta_permissions=[META_ACK, META_RESET],
                 handlers={"acumulador": handler_acum})
    ia.connect(ib)
    neg = ib.negotiated_meta_perms
    obj_repr = ObjectND(sa).add("DIM_META", {"tipo": "REPROGAM", "nuevo_modo": "acumulador"}, w=1.0)
    ib.receive(ia.send(obj_repr))
    return (META_ACK in neg) and (META_REPROGAM not in neg) and (ib.handler is None)


def test_meta_permissions_con_permiso():
    sa, sb = Space("A"), Space("B")
    def handler_acum(obj, ctx, session): return []
    ia = Session(sa, "IA-A", meta_permissions=[META_ACK, META_REPROGAM])
    ib = Session(sb, "IA-B", meta_permissions=[META_ACK, META_REPROGAM],
                 handlers={"acumulador": handler_acum})
    ia.connect(ib)
    obj_repr = ObjectND(sa).add("DIM_META", {"tipo": "REPROGAM", "nuevo_modo": "acumulador"}, w=1.0)
    ib.receive(ia.send(obj_repr))
    return ib.handler is handler_acum


def test_meta_depth_limit():
    sa, sb = Space("A"), Space("B")
    def handler_meta_gen(obj, ctx, session): return []
    ia = Session(sa, "IA-A", meta_permissions=[META_ACK, META_REPROGAM])
    ib = Session(sb, "IA-B", meta_permissions=[META_ACK, META_REPROGAM],
                 handlers={"meta_gen": handler_meta_gen})
    ia.connect(ib)
    ib._meta_depth = 1
    obj_repr = ObjectND(sa).add("DIM_META", {"tipo": "REPROGAM", "nuevo_modo": "meta_gen"}, w=1.0)
    ib.receive(ia.send(obj_repr))
    bloqueado = ib.handler is None
    ib._meta_depth = 0
    return bloqueado


def test_b2_numerico():
    sp = Space()
    h42, h43, h100 = sp._enc_numerico(42.), sp._enc_numerico(43.), sp._enc_numerico(100.)
    sim_42_43, sim_42_100 = _sim(h42, h43), _sim(h42, h100)
    obj_a = ObjectND(sp).add("DIM_SQL", {"n": 42}, w=1.0)
    obj_b = ObjectND(sp).add("DIM_SQL", {"n": 43}, w=1.0)
    obj_c = ObjectND(sp).add("DIM_SQL", {"n": 542}, w=1.0)
    sim_ab = _sim(obj_a._hv(), obj_b._hv())
    sim_ac = _sim(obj_a._hv(), obj_c._hv())
    return (sim_42_43 > 0.99) and (sim_42_100 < sim_42_43) and (sim_ab > sim_ac)


def test_dim_contract():
    sp = Space("IA_LEGAL")
    acuerdo = (ObjectND(sp)
               .add("DIM_CONTRACT", {"partes": 2, "tipo": "AESP", "vigente": True}, w=1.0)
               .add("DIM_SQL",      {"tabla": "contratos"}, w=0.6))
    activa = acuerdo.dims_activas()
    conn   = polydim_connect(sp, Space("IA_AUDITORIA"))
    dims   = conn.transfer(acuerdo)
    return "DIM_CONTRACT" in activa and "DIM_CONTRACT" in dims


def test_mesh_n3_broadcast():
    mesh = Mesh()
    sp0, sp1, sp2 = Space("IA0"), Space("IA1"), Space("IA2")
    mesh.add("IA_0", sp0); mesh.add("IA_1", sp1); mesh.add("IA_2", sp2)
    mesh.connect_all()
    obj     = ObjectND(sp0).add("DIM_SQL", {"tabla": "mesh_test"}, w=1.0)
    results = mesh.broadcast(obj, "IA_0")
    return ("IA_1" in results and "DIM_SQL" in results["IA_1"] and
            "IA_2" in results and "DIM_SQL" in results["IA_2"])


def test_mesh_n3_route():
    mesh = Mesh()
    sp0, sp1, sp2 = Space("R0"), Space("R1"), Space("R2")
    mesh.add("R_0", sp0); mesh.add("R_1", sp1); mesh.add("R_2", sp2)
    mesh.connect_all()
    obj  = ObjectND(sp0).add("DIM_GRAPH", {"nodo": "origen"}, w=1.0)
    dims = mesh.route(obj, "R_0", "R_2")
    return "DIM_GRAPH" in dims


def test_mesh_n3_topology():
    mesh = Mesh()
    for i in range(3): mesh.add(f"T_{i}", Space(f"T{i}"))
    mesh.connect_all()
    topo = mesh.topology()
    return all(len(peers) == 2 for peers in topo.values())


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
    topo         = mesh.topology()
    hub_peers    = set(topo["HUB"])
    worker_peers = [set(topo[f"W_{i}"]) for i in range(3)]
    return hub_peers == {"W_0","W_1","W_2"} and all(p=={"HUB"} for p in worker_peers)


def test_mesh_route_path():
    mesh = Mesh()
    spa, spb, spc = Space("PA"), Space("PB"), Space("PC")
    mesh.add("A", spa); mesh.add("B", spb); mesh.add("C", spc)
    mesh.connect_all()
    obj  = ObjectND(spa).add("DIM_TIME", {"orden": 1}, w=1.0)
    dims = mesh.route_path(obj, ["A", "B", "C"])
    return "DIM_TIME" in dims


def test_bug002_geo_id_survives_modo_s():
    sp  = Space("BUG002_TEST")
    obj = ObjectND(sp)
    obj.add("DIM_SQL",    {"tabla": "pedidos"})
    obj.add("DIM_PYTHON", {"clase": "PedidoService"})
    geo_orig = obj.geo_id
    wire = obj.to_wire_dict()
    if "geo_hv_b64" not in wire:
        return False
    sp2 = Space("RECEPTOR")
    o_rx = ObjectND(sp2)
    geo_b64 = wire.get("geo_hv_b64")
    if geo_b64:
        geo_bytes = base64.b64decode(geo_b64)
        o_rx._geo = np.frombuffer(geo_bytes, dtype=np.float32).copy()
        o_rx._cache = None
    for d, info in wire.get("dims", {}).items():
        o_rx.add(d, info.get("props", {}), w=info.get("w", 1.0))
    if o_rx.geo_id != geo_orig:
        return False
    conn = polydim_connect(sp, Space("OTRA"))
    dims = conn.transfer(obj)
    return "DIM_SQL" in dims


def test_bug003_rpa_preserves_information():
    sp_a = Space("IA_A"); sp_b = Space("IA_B")
    A = np.array([sp_a.sym(s) for s in SONDAS], dtype=np.float32)
    B = np.array([sp_b.sym(s) for s in SONDAS], dtype=np.float32)
    obj = ObjectND(sp_a)
    obj.add("DIM_SQL",    {"tabla": "pedidos"})
    obj.add("DIM_PYTHON", {"clase": "PedidoService"})
    hv = obj._hv()
    hv_v05 = B.T @ (A @ hv)
    n = np.linalg.norm(hv_v05)
    hv_v05 = hv_v05 / n if n > 1e-10 else hv_v05
    hv_v06 = align_transform(hv, A, B)
    sim_v05 = _sim(hv_v05, hv)
    sim_v06 = _sim(hv_v06, hv)
    coords   = A @ hv
    residual = hv - A.T @ coords
    norm_res = float(np.linalg.norm(residual))
    return sim_v06 >= sim_v05 and norm_res > 0.01


def test_pcrg_determinism_and_normalization():
    h1 = pcrg_make_hv("DIM_SQL", N)
    h2 = pcrg_make_hv("DIM_SQL", N)
    if not np.allclose(h1, h2):
        return False
    norm = float(np.linalg.norm(h1))
    if abs(norm - 1.0) > 1e-4:
        return False
    sp = Space("PCRG_TEST")
    expected = pcrg_make_hv("PCRG_TEST:DIM_SQL", N)
    actual   = sp.sub("DIM_SQL")
    return _sim(expected, actual) > 0.9999


def test_pcrg_cross_check_with_reference_module():
    try:
        import polydim_rng
        ref = polydim_rng.polydim_make_hv("DIM_SQL", 10)
        own = pcrg_make_hv("DIM_SQL", 10)
        return bool(np.allclose(ref, own, atol=1e-5))
    except ImportError:
        hv_sql  = pcrg_make_hv("DIM_SQL", N)
        hv_rust = pcrg_make_hv("DIM_RUST", N)
        sim_diff_keys = _sim(hv_sql, hv_rust)
        hv_a = pcrg_make_hv("IA_A:DIM_SQL", N)
        hv_b = pcrg_make_hv("IA_B:DIM_SQL", N)
        sim_diff_seeds = _sim(hv_a, hv_b)
        return (0.47 < sim_diff_keys < 0.53) and (0.47 < sim_diff_seeds < 0.53)


def test_semantic_clustering_mock():
    sp = Space(semantic_backend=MockSemanticBackend())
    h_usuario = sp.sym("usuario")
    h_cliente = sp.sym("cliente")
    h_tabla   = sp.sym("tabla")
    sim_mismo_grupo   = _sim(h_usuario, h_cliente)
    sim_grupo_distinto = _sim(h_usuario, h_tabla)
    return sim_mismo_grupo > sim_grupo_distinto


def test_semantic_backend_with_personal_seed():
    sp = Space("IA_X", semantic_backend=MockSemanticBackend())
    h_usuario = sp.sym("usuario")
    h_cliente = sp.sym("cliente")
    h_tabla   = sp.sym("tabla")
    sim_mismo   = _sim(h_usuario, h_cliente)
    sim_distinto = _sim(h_usuario, h_tabla)
    sp2 = Space("IA_Y", semantic_backend=MockSemanticBackend())
    h_usuario_2 = sp2.sym("usuario")
    distinto_por_seed = _sim(h_usuario, h_usuario_2) < 0.999
    return (sim_mismo > sim_distinto) and distinto_por_seed


def test_align_with_different_backends():
    sp_a = Space("IA_SEM_A", semantic_backend=MockSemanticBackend())
    sp_b = Space("IA_SEM_B", semantic_backend=MockSemanticBackend())
    conn = polydim_connect(sp_a, sp_b)
    obj  = ObjectND(sp_a).add("DIM_SQL", {"tabla": "usuarios"}, w=1.0)
    dims = conn.transfer(obj)
    return "DIM_SQL" in dims


# -- Tests V0.9 -- TASK_031 ----------------------------------------

def test_to_symbolic_deprecated_alias_matches_wire_dict():
    """to_symbolic() debe emitir DeprecationWarning y devolver
    exactamente el mismo dict que to_wire_dict()."""
    sp = Space("DEPRECATION_TEST")
    obj = ObjectND(sp).add("DIM_SQL", {"tabla": "usuarios"}, w=1.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        wire = obj.to_wire_dict()
        n_warnings_wire = len(w)
        sym = obj.to_symbolic()
        n_warnings_after_sym = len(w)
    emitio_warning = (n_warnings_after_sym - n_warnings_wire) == 1
    es_deprecation = w[-1].category is DeprecationWarning if w else False
    return emitio_warning and es_deprecation and (sym == wire)


def test_empaquetar_objeto_uses_wire_dict_not_symbolic():
    """empaquetar_objeto() debe usar to_wire_dict() internamente y NO
    disparar el DeprecationWarning de to_symbolic()."""
    sp = Space("WIRE_NOT_SYMBOLIC")
    obj = ObjectND(sp).add("DIM_SQL", {"tabla": "logs"}, w=1.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pkt = empaquetar_objeto(obj, Mode.S)
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    return len(deprecation_warnings) == 0 and pkt.payload_S is not None and "geo_hv_b64" in pkt.payload_S


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
        test_bug002_geo_id_survives_modo_s,
        test_bug003_rpa_preserves_information,
        test_pcrg_determinism_and_normalization,
        test_pcrg_cross_check_with_reference_module,
        test_semantic_clustering_mock,
        test_semantic_backend_with_personal_seed,
        test_align_with_different_backends,
        # V0.9
        test_to_symbolic_deprecated_alias_matches_wire_dict,
        test_empaquetar_objeto_uses_wire_dict_not_symbolic,
    ]
    results = [t() for t in tests]
    for t, r in zip(tests, results):
        print(f"  {t.__name__:50s}: {'OK' if r else 'FALLO'}")
    passed = sum(results)
    total  = len(results)
    print(f"\n  {passed}/{total} tests OK -- "
          f"{'V0.9 VERIFICADA' if passed == total else 'CHECKS FALLIDOS'}")
