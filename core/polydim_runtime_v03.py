"""
POLYDIM Runtime — V0.3
=======================
Agrega semantic_backend opcional a Space.
Totalmente compatible con V0.2.

NUEVO EN V0.3:
  Space(semantic_backend=MockSemanticBackend())  → clusters semanticos de prueba
  Space(semantic_backend=MiniLMBackend())         → embeddings reales (requiere pip)
  Space()                                         → deterministico V0.2 (default)

RESULTADOS VERIFICADOS V0.3:
  usuario↔cliente  sem=0.830  det=0.506  (mismo grupo semantico)
  usuario↔tabla    sem=0.512  det=0.493  (grupos distintos)
  ALIGN sem→det:   0.923      valido=True  (IAs con backends distintos se alinean)
  dims_activas:    DIM_SQL=0.812  DIM_PYTHON=0.711  (con backend)

INSTALACION PARA MiniLM:
  pip install sentence-transformers

CRITICO: obj y Session deben usar el MISMO Space.

Autor:   ai.mpat.agt@gmail.com
Version: V0.3 — 2026-06-12
"""

from __future__ import annotations
import numpy as np, hashlib, math, uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

N=10000; _S=1.0/(2.0*math.sqrt(N)); UMBRAL=0.5+2.0*_S; CONTENT_W=0.3; UMBRAL_ALIGN=0.85
NATIVE=["DIM_PYTHON","DIM_RUST","DIM_FLUTTER","DIM_SQL","DIM_GRAPH","DIM_VECTOR","DIM_TIME","DIM_ERROR","DIM_META"]
SONDAS=NATIVE+["entero","flotante","cadena","lista","diccionario","verdadero","falso","nulo","error","exito",
               "crear","leer","actualizar","borrar","usuario","sesion","permiso","dato","proceso"]

class Mode(str,Enum): S="MODO_S"; G="MODO_G"; H="MODO_H"
class Cap(str,Enum):  S="CAP_S";  G="CAP_G";  ALIGN="CAP_ALIGN"; BROADCAST="CAP_BROADCAST"
class SessionState(str,Enum):
    IDLE="IDLE"; CONNECTING="CONNECTING"; NEGOTIATING="NEGOTIATING"
    ALIGNING="ALIGNING"; READY="READY"; DEGRADED="DEGRADED"; FAILED="FAILED"; CLOSED="CLOSED"

@dataclass
class Packet:
    session_id:str; seq:int; op:str
    payload_S:Optional[dict]=None; payload_G:Optional[np.ndarray]=None
    intent:List[str]=field(default_factory=list)

def _bind(a,b): r=a*b;n=np.linalg.norm(r);return r/n if n>1e-10 else r
def _sup(*hvs,ws=None):
    c=sum(w*h for w,h in zip(ws,hvs)) if ws else np.sum(hvs,axis=0)
    n=np.linalg.norm(c);return c/n if n>1e-10 else c
def _proj(hv,sub): return (float(np.dot(hv,sub))+1.0)/2.0
def _sim(a,b): return float((np.dot(a,b)+1.0)/2.0)
def align_transform(hv,A,B): c=B.T@(A@hv);n=np.linalg.norm(c);return c/n if n>1e-10 else c

# ---------------------------------------------------------------------------
# Backends semanticos
# ---------------------------------------------------------------------------

class SemanticBackend(ABC):
    """Interfaz para backends semanticos. encode() retorna vector normalizado."""
    dim: int = 384
    @abstractmethod
    def encode(self, text: str) -> np.ndarray: ...

class MockSemanticBackend(SemanticBackend):
    """
    Backend de prueba con clusters semanticos manuales (sin modelo real).
    Usar para tests y desarrollo sin internet.
    """
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
            s=int(hashlib.md5(f"G:{g}".encode()).hexdigest(),16)%(2**32)
            hv=np.random.default_rng(s).standard_normal(self.dim).astype(np.float32)
            self._g[g]=hv/np.linalg.norm(hv)
    def _grupo(self, t):
        for g,ms in self.GRUPOS.items():
            if any(t.lower()==m.lower() for m in ms): return g
        return None
    def encode(self, text:str) -> np.ndarray:
        s=int(hashlib.md5(f"SEM:{text}".encode()).hexdigest(),16)%(2**32)
        base=np.random.default_rng(s).standard_normal(self.dim).astype(np.float32)
        base/=np.linalg.norm(base)
        g=self._grupo(text)
        if g:
            hv=0.4*base+0.6*self._g[g]; n=np.linalg.norm(hv); return hv/n
        return base

class MiniLMBackend(SemanticBackend):
    """
    Backend real con sentence-transformers all-MiniLM-L6-v2.
    Requiere: pip install sentence-transformers
    Descarga ~90MB automaticamente en primera ejecucion.
    """
    dim = 384
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            raise ImportError("Instalar: pip install sentence-transformers")
    def encode(self, text:str) -> np.ndarray:
        return self._model.encode(text, normalize_embeddings=True).astype(np.float32)

class FastTextBackend(SemanticBackend):
    """
    Backend con FastText multilingue (incluye castellano).
    Requiere: pip install fasttext-wheel
    Modelo: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.bin.gz
    """
    dim = 300
    def __init__(self, model_path:str):
        import fasttext
        self._model = fasttext.load_model(model_path)
    def encode(self, text:str) -> np.ndarray:
        hv = self._model.get_word_vector(text).astype(np.float32)
        n  = np.linalg.norm(hv)
        return hv/n if n > 1e-10 else hv

def make_jl(d_in:int, d_out:int=N) -> np.ndarray:
    """Matriz Johnson-Lindenstrauss deterministica: d_in → d_out."""
    s=int(hashlib.md5(f"JL_{d_out}_{d_in}".encode()).hexdigest(),16)%(2**32)
    R=np.random.default_rng(s).standard_normal((d_out,d_in)).astype(np.float32)
    return R/math.sqrt(d_in)

# ---------------------------------------------------------------------------
# Space V0.3
# ---------------------------------------------------------------------------

class Space:
    """
    Espacio POLYDIM V0.3 con semantic_backend opcional.

    Space()                                     → deterministico (V0.2)
    Space("IA_X")                               → personalizado deterministico
    Space(semantic_backend=MockSemanticBackend()) → con semantica de prueba
    Space("IA_X", semantic_backend=MiniLMBackend()) → personal + MiniLM

    Con backend: conceptos semanticamente relacionados tienen hipervectores
    mas cercanos. usuario↔cliente = 0.83 vs 0.51 sin backend.
    """
    def __init__(self, personal_seed:str="", semantic_backend:SemanticBackend=None):
        self.ps=personal_seed; self.backend=semantic_backend
        self._s={}; self._sub={}
        self._JL = make_jl(semantic_backend.dim) if semantic_backend else None
        for d in NATIVE: self._sub[d]=self._mk(d)

    def _mk(self, name:str) -> np.ndarray:
        if self.backend:
            hv = self._JL @ self.backend.encode(name)
            if self.ps:
                k=f"{self.ps}:{name}"; s=int(hashlib.md5(k.encode()).hexdigest(),16)%(2**32)
                p=np.random.default_rng(s).standard_normal(N).astype(np.float32); p/=np.linalg.norm(p)
                hv = 0.85*hv + 0.15*p
        else:
            k=f"{self.ps}:{name}" if self.ps else name
            s=int(hashlib.md5(k.encode()).hexdigest(),16)%(2**32)
            hv=np.random.default_rng(s).standard_normal(N).astype(np.float32)
        n=np.linalg.norm(hv); return (hv/n if n>1e-10 else hv).astype(np.float32)

    def sym(self,n):
        if n not in self._s: self._s[n]=self._mk(n)
        return self._s[n]
    def sub(self,n):
        if n not in self._sub: self._sub[n]=self.sym(n)
        return self._sub[n]
    def _rnd(self): hv=np.random.randn(N).astype(np.float32);return hv/np.linalg.norm(hv)
    def _enc(self,p):
        if not p: return self.sym("__empty__")
        return _sup(*[_bind(self.sym(str(k)),self.sym(str(v))) for k,v in p.items()])

# ---------------------------------------------------------------------------
# ObjectND, Session, Connection — identicos a V0.2 (compatibilidad total)
# ---------------------------------------------------------------------------

class ObjectND:
    def __init__(self,space=None):
        self._sp=space or Space(); self._props={}; self._w={}
        self._geo=self._sp._rnd(); self._cache=None
    @property
    def geo_id(self): return hashlib.md5(self._geo.tobytes()).hexdigest()[:12]
    def add(self,dim,props=None,w=1.0):
        self._props[dim]=props or {}; self._w[dim]=float(np.clip(w,0,1))
        self._cache=None; return self
    def _hv(self):
        if self._cache is not None: return self._cache
        cs=[self._geo]; ws=[1.0]
        for d,p in self._props.items():
            ww=self._w.get(d,1.0)
            if ww<=0: continue
            cs.append(self._sp.sub(d)); ws.append(ww)
            cs.append(_bind(self._sp.sub(d),self._sp._enc(p))); ws.append(ww*CONTENT_W)
        self._cache=_sup(*cs,ws=ws); return self._cache
    def activacion(self,dim): return _proj(self._hv(),self._sp.sub(dim))
    def dims_activas(self,umbral=UMBRAL):
        seen,r={},{}
        for d in NATIVE+list(self._props):
            if d not in seen:
                w=self.activacion(d)
                if w>umbral: r[d]=round(w,4)
                seen[d]=1
        return r
    def to_symbolic(self):
        return {"geo_id":self.geo_id,"dims":{d:{"w":self._w.get(d,1.0),"props":p} for d,p in self._props.items()}}
    def __repr__(self):
        dims=", ".join(f"{d}[{self._w.get(d,1):.1f}]" for d in self._props)
        return f"ObjectND(id={self.geo_id}, dims=[{dims}])"

def empaquetar_objeto(obj,mode,A=None,B=None):
    sym=obj.to_symbolic(); intent=list(obj.dims_activas().keys()); hv=obj._hv()
    if A is not None and B is not None: hv=align_transform(hv,A,B)
    if mode==Mode.S: return Packet("",0,"ND_SEND",payload_S=sym,intent=intent)
    if mode==Mode.G: return Packet("",0,"ND_SEND",payload_G=hv,intent=intent)
    return Packet("",0,"ND_SEND",payload_S=sym,payload_G=hv,intent=intent)

class Session:
    def __init__(self,space,name="IA",caps=None):
        self.space=space; self.name=name
        self.caps=caps or [Cap.S,Cap.G,Cap.ALIGN]
        self.state=SessionState.IDLE; self.mode=Mode.S
        self.session_id=None; self.align_score=None
        self._A=None; self._B=None; self._seq=0
    def handshake(self,remote):
        from dataclasses import dataclass as dc, field as fi
        n1,n2=uuid.uuid4().hex[:8],uuid.uuid4().hex[:8]
        caps_comun=set(self.caps)&set(remote.caps)
        mode=Mode.H if Cap.G in caps_comun else Mode.S
        sid=hashlib.md5((n1+n2).encode()).hexdigest()[:16]
        self.session_id=remote.session_id=sid
        self.mode=remote.mode=mode
        self.state=remote.state=SessionState.NEGOTIATING
        return True
    def align(self,remote,sondas=None):
        self.state=remote.state=SessionState.ALIGNING
        p=sondas or SONDAS
        A=np.array([self.space.sym(s) for s in p],dtype=np.float32)
        B=np.array([remote.space.sym(s) for s in p],dtype=np.float32)
        scores=[_sim(align_transform(self.space.sub(d),A,B),remote.space.sub(d)) for d in NATIVE]
        score=float(np.mean(scores)); valid=score>=UMBRAL_ALIGN
        if valid:
            self._A=A; self._B=B; remote._A=B; remote._B=A
            self.align_score=remote.align_score=round(score,4)
            self.state=remote.state=SessionState.READY
        else:
            self.mode=remote.mode=Mode.S; self.state=remote.state=SessionState.DEGRADED
        return score
    def connect(self,remote):
        self.handshake(remote)
        if self.mode in (Mode.G,Mode.H): self.align(remote)
        return self
    def send(self,obj):
        self._seq+=1; pkt=empaquetar_objeto(obj,self.mode,self._A,self._B)
        pkt.session_id=self.session_id or ""; pkt.seq=self._seq; return pkt
    def receive(self,pkt):
        if pkt.payload_G is not None: hv=pkt.payload_G
        elif pkt.payload_S:
            o=ObjectND(self.space)
            for d,info in pkt.payload_S.get("dims",{}).items():
                o.add(d,info.get("props",{}),w=info.get("w",1.0))
            hv=o._hv()
        else: return {}
        return {d:round(_proj(hv,self.space.sub(d)),4) for d in NATIVE if _proj(hv,self.space.sub(d))>UMBRAL}
    @property
    def info(self): return {"session_id":self.session_id,"mode":self.mode.value,"state":self.state.value,"align_score":self.align_score}
    def __repr__(self): return f"Session({self.name},{self.state.value},{self.mode.value})"

class Connection:
    def __init__(self,sp_a,sp_b):
        self._ia=Session(sp_a,"IA_A"); self._ib=Session(sp_b,"IA_B"); self._ia.connect(self._ib)
    def transfer(self,obj): return self._ib.receive(self._ia.send(obj))
    @property
    def info(self): return self._ia.info
    def __repr__(self): return f"Connection({self._ia.info})"

def polydim_connect(space_a,space_b): return Connection(space_a,space_b)

__all__=["Space","ObjectND","Session","Connection","polydim_connect","empaquetar_objeto",
         "SemanticBackend","MockSemanticBackend","MiniLMBackend","FastTextBackend",
         "Mode","Cap","SessionState","Packet","NATIVE","UMBRAL","UMBRAL_ALIGN","N"]
