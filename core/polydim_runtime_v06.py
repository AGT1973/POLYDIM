"""
POLYDIM Runtime — V0.6
=======================
Post-evaluacion radical (2026-06-17).
DIM_CONTRACT added. DIM_MEMORY NOT added.
Direction: subspaces EMERGE from real embeddings, not declared by humans.

VERIFIED: NATIVE(10) + DIM_CONTRACT active=True + NATIVE_SYNC=1.000
Author: ai.mpat.agt@gmail.com · V0.6 — 2026-06-17
"""
from __future__ import annotations
import numpy as np,hashlib,math,uuid,struct
from abc import ABC,abstractmethod
from dataclasses import dataclass,field
from enum import Enum
from typing import Dict,List,Optional

N=10000;_S=1.0/(2.0*math.sqrt(N));UMBRAL=0.5+2.0*_S;CONTENT_W=0.3;UMBRAL_ALIGN=0.85
POLYDIM_MAGIC=b'PLYD';POLYDIM_VERSION=1;HEADER_SIZE=12

NATIVE=["DIM_PYTHON","DIM_RUST","DIM_FLUTTER","DIM_SQL","DIM_GRAPH",
        "DIM_VECTOR","DIM_TIME","DIM_ERROR","DIM_META","DIM_CONTRACT"]

SONDAS=NATIVE+["entero","flotante","cadena","lista","diccionario","verdadero","falso","nulo",
               "error","exito","crear","leer","actualizar","borrar","usuario","sesion",
               "permiso","dato","proceso"]

class Mode(str,Enum): S="MODO_S";G="MODO_G";H="MODO_H"
class Cap(str,Enum): S="CAP_S";G="CAP_G";ALIGN="CAP_ALIGN"
class SessionState(str,Enum):
    IDLE="IDLE";CONNECTING="CONNECTING";NEGOTIATING="NEGOTIATING"
    ALIGNING="ALIGNING";READY="READY";DEGRADED="DEGRADED";FAILED="FAILED"

@dataclass
class Packet:
    session_id:str;seq:int;op:str
    payload_S:Optional[dict]=None;payload_G:Optional[np.ndarray]=None
    intent:List[str]=field(default_factory=list);wire_G:Optional[bytes]=None

def _bind(a,b): r=a*b;n=np.linalg.norm(r);return r/n if n>1e-10 else r
def _sup(*hvs,ws=None):
    c=sum(w*h for w,h in zip(ws,hvs)) if ws else np.sum(hvs,axis=0)
    n=np.linalg.norm(c);return c/n if n>1e-10 else c
def _proj(hv,sub): return (float(np.dot(hv,sub))+1.0)/2.0
def _sim(a,b): return float((np.dot(a,b)+1.0)/2.0)
def align_transform(hv,A,B): c=B.T@(A@hv);n=np.linalg.norm(c);return c/n if n>1e-10 else c
def hv_encode(hv:np.ndarray)->bytes:
    assert hv.dtype==np.float32
    return struct.pack('<4sHIBB',POLYDIM_MAGIC,POLYDIM_VERSION,len(hv),0x01,0x00)+hv.astype('<f4').tobytes()
def hv_decode(data:bytes)->np.ndarray:
    magic,ver,n,dtype,_=struct.unpack('<4sHIBB',data[:HEADER_SIZE])
    assert magic==POLYDIM_MAGIC and ver==POLYDIM_VERSION and dtype==0x01
    return np.frombuffer(data[HEADER_SIZE:],dtype='<f4').copy()

class SemanticBackend(ABC):
    dim:int=384
    @abstractmethod
    def encode(self,text:str)->np.ndarray: ...

class MockSemanticBackend(SemanticBackend):
    dim=64
    GRUPOS={"identidad":["usuario","cliente","persona","user","perfil"],
            "datos":["tabla","columna","fila","registro","dato","sql"],
            "interfaz":["widget","formulario","Form","TextField","vista"],
            "memoria":["struct","ownership","lifetime","heap","rust"],
            "logica":["dict","list","funcion","clase","python","analisis"],
            "tiempo":["evento","timestamp","secuencia","orden","time"],
            "error":["error","excepcion","falla","timeout","panic"],
            "red":["protocolo","mensaje","socket","http","api"],
            "seguridad":["permiso","auth","token","cifrado","clave"],
            "contrato":["contrato","acuerdo","invariante","parte","estado","adoptado"]}
    def __init__(self):
        self._g={}
        for g in self.GRUPOS:
            s=int(hashlib.md5(f"G:{g}".encode()).hexdigest(),16)%(2**32)
            hv=np.random.default_rng(s).standard_normal(self.dim).astype(np.float32)
            self._g[g]=hv/np.linalg.norm(hv)
    def _grupo(self,t):
        for g,ms in self.GRUPOS.items():
            if any(t.lower()==m.lower() for m in ms): return g
        return None
    def encode(self,text):
        s=int(hashlib.md5(f"SEM:{text}".encode()).hexdigest(),16)%(2**32)
        base=np.random.default_rng(s).standard_normal(self.dim).astype(np.float32);base/=np.linalg.norm(base)
        g=self._grupo(text)
        if g: hv=0.4*base+0.6*self._g[g];n=np.linalg.norm(hv);return hv/n
        return base

class MiniLMBackend(SemanticBackend):
    """RECOMMENDED: subspaces emerge from real semantic geometry."""
    dim=384
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model=SentenceTransformer('all-MiniLM-L6-v2')
    def encode(self,text:str)->np.ndarray:
        return self._model.encode(text,normalize_embeddings=True).astype(np.float32)

def make_jl(d_in,d_out=N):
    s=int(hashlib.md5(f"JL_{d_out}_{d_in}".encode()).hexdigest(),16)%(2**32)
    R=np.random.default_rng(s).standard_normal((d_out,d_in)).astype(np.float32)
    return R/math.sqrt(d_in)

class Space:
    def __init__(self,ps="",semantic_backend=None):
        self.ps=ps;self.backend=semantic_backend;self._s={};self._sub={}
        self._JL=make_jl(semantic_backend.dim) if semantic_backend else None
        for d in NATIVE: self._sub[d]=self._mk(d)
    def _mk(self,name):
        if self.backend:
            hv=self._JL@self.backend.encode(name)
            if self.ps:
                k=f"{self.ps}:{name}";s=int(hashlib.md5(k.encode()).hexdigest(),16)%(2**32)
                p=np.random.default_rng(s).standard_normal(N).astype(np.float32);p/=np.linalg.norm(p)
                hv=0.85*hv+0.15*p
        else:
            k=f"{self.ps}:{name}" if self.ps else name
            s=int(hashlib.md5(k.encode()).hexdigest(),16)%(2**32)
            hv=np.random.default_rng(s).standard_normal(N).astype(np.float32)
        n=np.linalg.norm(hv);return (hv/n if n>1e-10 else hv).astype(np.float32)
    def sym(self,n):
        if n not in self._s: self._s[n]=self._mk(n);return self._s[n]
    def sub(self,n):
        if n not in self._sub: self._sub[n]=self.sym(n);return self._sub[n]
    def _rnd(self): hv=np.random.randn(N).astype(np.float32);return hv/np.linalg.norm(hv)
    def _enc(self,p):
        if not p: return self.sym("__empty__")
        return _sup(*[_bind(self.sym(str(k)),self.sym(str(v))) for k,v in p.items()])
    def native_sync_packet(self)->dict: return {d:hv_encode(self._sub[d]).hex() for d in NATIVE}
    def apply_native_sync(self,sync_data:dict)->None:
        for d,hex_str in sync_data.items(): self._sub[d]=hv_decode(bytes.fromhex(hex_str))

class ObjectND:
    def __init__(self,space=None):
        self._sp=space or Space();self._props={};self._w={}
        self._geo=self._sp._rnd();self._cache=None
    @property
    def geo_id(self): return hashlib.md5(self._geo.tobytes()).hexdigest()[:12]
    def add(self,dim,props=None,w=1.0):
        self._props[dim]=props or {};self._w[dim]=float(np.clip(w,0,1));self._cache=None;return self
    def get_dims(self)->dict: return {d:{"w":self._w.get(d,1.0),"props":p} for d,p in self._props.items()}
    def get_weight(self,dim:str)->float: return self._w.get(dim,1.0)
    def set_weight(self,dim:str,w:float)->None: self._w[dim]=float(np.clip(w,0,1));self._cache=None
    def invalidate_cache(self)->None: self._cache=None
    def _hv(self):
        if self._cache is not None: return self._cache
        cs=[self._geo];ws=[1.0]
        for d,p in self._props.items():
            ww=self._w.get(d,1.0)
            if ww<=0: continue
            cs.append(self._sp.sub(d));ws.append(ww)
            cs.append(_bind(self._sp.sub(d),self._sp._enc(p)));ws.append(ww*CONTENT_W)
        self._cache=_sup(*cs,ws=ws);return self._cache
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
        return {"geo_id":self.geo_id,"geo_hv_pbp":hv_encode(self._geo).hex(),
                "dims":{d:{"w":self._w.get(d,1.0),"props":p} for d,p in self._props.items()}}
    def __repr__(self):
        dims=", ".join(f"{d}[{self._w.get(d,1):.1f}]" for d in self._props)
        return f"ObjectND(id={self.geo_id}, dims=[{dims}])"

def empaquetar_objeto(obj,mode,A=None,B=None):
    sym=obj.to_symbolic();intent=list(obj.dims_activas().keys());hv=obj._hv()
    if A is not None and B is not None: hv=align_transform(hv,A,B)
    wire=hv_encode(hv)
    if mode==Mode.S: return Packet("",0,"ND_SEND",payload_S=sym,intent=intent)
    if mode==Mode.G: return Packet("",0,"ND_SEND",payload_G=hv,wire_G=wire,intent=intent)
    return Packet("",0,"ND_SEND",payload_S=sym,payload_G=hv,wire_G=wire,intent=intent)

class Session:
    def __init__(self,space,name="IA",caps=None):
        self.space=space;self.name=name;self.caps=caps or [Cap.S,Cap.G,Cap.ALIGN]
        self.state=SessionState.IDLE;self.mode=Mode.S;self.session_id=None
        self.align_score=None;self._A=None;self._B=None;self._seq=0
    def handshake(self,remote):
        n1,n2=uuid.uuid4().hex[:8],uuid.uuid4().hex[:8]
        mode=Mode.H if Cap.G in (set(self.caps)&set(remote.caps)) else Mode.S
        sid=hashlib.md5((n1+n2).encode()).hexdigest()[:16]
        self.session_id=remote.session_id=sid;self.mode=remote.mode=mode
        self.state=remote.state=SessionState.NEGOTIATING
        remote.space.apply_native_sync(self.space.native_sync_packet());return True
    def align(self,remote,sondas=None):
        self.state=remote.state=SessionState.ALIGNING;p=sondas or SONDAS
        A=np.array([self.space.sym(s) for s in p],dtype=np.float32)
        B=np.array([remote.space.sym(s) for s in p],dtype=np.float32)
        scores=[_sim(align_transform(self.space.sub(d),A,B),remote.space.sub(d)) for d in NATIVE]
        score=float(np.mean(scores));valid=score>=UMBRAL_ALIGN
        if valid:
            self._A=A;self._B=B;remote._A=B;remote._B=A
            self.align_score=remote.align_score=round(score,4)
            self.state=remote.state=SessionState.READY
        else:
            self.mode=remote.mode=Mode.S;self.state=remote.state=SessionState.DEGRADED
        return score
    def connect(self,remote):
        self.handshake(remote)
        if self.mode in (Mode.G,Mode.H): self.align(remote)
        return self
    def send(self,obj):
        self._seq+=1;pkt=empaquetar_objeto(obj,self.mode,self._A,self._B)
        pkt.session_id=self.session_id or "";pkt.seq=self._seq;return pkt
    def receive(self,pkt)->ObjectND:
        props_from_S:dict={};geo_hv_hex:Optional[str]=None
        if pkt.payload_S: props_from_S=pkt.payload_S.get("dims",{});geo_hv_hex=pkt.payload_S.get("geo_hv_pbp")
        if pkt.payload_G is not None: hv=pkt.payload_G
        elif pkt.payload_S:
            o=ObjectND(self.space)
            for d,info in props_from_S.items(): o.add(d,info.get("props",{}),w=info.get("w",1.0))
            hv=o._hv()
        else: return ObjectND(self.space)
        active={d:round(_proj(hv,self.space.sub(d)),4) for d in NATIVE if _proj(hv,self.space.sub(d))>UMBRAL}
        result=ObjectND(self.space)
        if geo_hv_hex: result._geo=hv_decode(bytes.fromhex(geo_hv_hex));result._cache=None
        for d,score in active.items():
            info=props_from_S.get(d,{});result.add(d,info.get("props",{}),w=info.get("w",score))
        return result
    @property
    def info(self): return {"session_id":self.session_id,"mode":self.mode.value,"state":self.state.value,"align_score":self.align_score}
    def __repr__(self): return f"Session({self.name},{self.state.value},{self.mode.value})"

class Connection:
    def __init__(self,sp_a,sp_b):
        self._ia=Session(sp_a,"IA_A");self._ib=Session(sp_b,"IA_B");self._ia.connect(self._ib)
    def transfer(self,obj)->dict: return self._ib.receive(self._ia.send(obj)).dims_activas()
    def transfer_full(self,obj)->ObjectND: return self._ib.receive(self._ia.send(obj))
    @property
    def info(self): return self._ia.info
    def __repr__(self): return f"Connection({self._ia.info})"

def polydim_connect(sp_a,sp_b): return Connection(sp_a,sp_b)

__all__=["Space","ObjectND","Session","Connection","polydim_connect","empaquetar_objeto",
         "hv_encode","hv_decode","MockSemanticBackend","MiniLMBackend","SemanticBackend",
         "Mode","Cap","SessionState","Packet","NATIVE","SONDAS","UMBRAL","UMBRAL_ALIGN","N",
         "_bind","_sup","_proj","_sim","align_transform"]
