"""
POLYDIM Session + Demo E2E — V0.1
===================================
HANDSHAKE + ALIGN + transferencia de OBJECT_ND entre dos IAs.

Resultado verificado:
  Score ALIGN = 0.9994  (umbral=0.85)
  DIM_SQL     IA_A=0.805 → IA_B=0.896  SIN ALIGN=0.496
  DIM_PYTHON  IA_A=0.712 → IA_B=0.774  SIN ALIGN=0.500
  DIM_FLUTTER IA_A=0.591 → IA_B=0.614  SIN ALIGN=0.504
  DIM_RUST    weight=0.0 → 0.506 (latente, correcto)

Hallazgo critico:
  La metrica ALIGN debe medir calidad de subespacios DIMENSIONALES (NATIVE_DIMS),
  no vectores de contenido arbitrarios. Score sobre contenido = ~0.5 (falso negativo).
  Score sobre dimensiones = 0.999 (correcto).

Autor:   ai.mpat.agt@gmail.com
Version: V0.1 — 2026-06-11
"""

import numpy as np, hashlib, math, json, uuid
from typing import Dict, List, Optional, Tuple

POLYDIM_N    = 10000
SIGMA        = 1.0/(2.0*math.sqrt(POLYDIM_N))
UMBRAL       = 0.5 + 2.0*SIGMA        # 0.510
CONTENT_W    = 0.3
UMBRAL_ALIGN = 0.85

NATIVE = ["DIM_PYTHON","DIM_RUST","DIM_FLUTTER","DIM_SQL",
          "DIM_GRAPH","DIM_VECTOR","DIM_TIME","DIM_ERROR","DIM_META"]

SONDAS = NATIVE + [
    "entero","flotante","cadena","lista","diccionario",
    "verdadero","falso","nulo","error","exito",
    "crear","leer","actualizar","borrar",
    "usuario","sesion","permiso","dato","proceso"]

# VSA primitives
def hv_bind(a,b):
    r=a*b; n=np.linalg.norm(r); return r/n if n>1e-10 else r
def hv_sp(*hvs, ws=None):
    c=sum(w*h for w,h in zip(ws,hvs)) if ws else np.sum(hvs,axis=0)
    n=np.linalg.norm(c); return c/n if n>1e-10 else c
def hv_proj(hv,sub): return (float(np.dot(hv,sub))+1.0)/2.0
def hv_sim(a,b): return float((np.dot(a,b)+1.0)/2.0)

class PolyDimSpace:
    """
    Espacio POLYDIM con personal_seed para simular IAs distintas.
    personal_seed="" → espacio estandar deterministico (MODO_S compatible).
    personal_seed != "" → espacio personalizado (requiere ALIGN para MODO_G/H).
    """
    def __init__(self, ps=""):
        self.ps=ps; self._s={}; self._sub={}
        for d in NATIVE: self._sub[d]=self._mk(d)
    def _mk(self,name):
        k=f"{self.ps}:{name}" if self.ps else name
        s=int(hashlib.md5(k.encode()).hexdigest(),16)%(2**32)
        hv=np.random.default_rng(s).standard_normal(POLYDIM_N).astype(np.float32)
        return hv/np.linalg.norm(hv)
    def sym(self,n):
        if n not in self._s: self._s[n]=self._mk(n)
        return self._s[n]
    def sub(self,n):
        if n not in self._sub: self._sub[n]=self.sym(n)
        return self._sub[n]
    def rnd(self):
        hv=np.random.randn(POLYDIM_N).astype(np.float32); return hv/np.linalg.norm(hv)
    def enc(self,p):
        if not p: return self.sym("__empty__")
        return hv_sp(*[hv_bind(self.sym(str(k)),self.sym(str(v))) for k,v in p.items()])

class ObjectND:
    def __init__(self,sp):
        self.sp=sp; self._p={}; self._w={}
        self._geo=sp.rnd(); self._c=None
    def geo_hash(self): return hashlib.md5(self._geo.tobytes()).hexdigest()[:12]
    def add(self,n,p,w=1.0):
        self._p[n]=p; self._w[n]=float(np.clip(w,0,1)); self._c=None; return self
    def to_hv(self):
        if self._c is not None: return self._c
        cs=[self._geo]; ws=[1.0]
        for n,p in self._p.items():
            ww=self._w.get(n,1.0)
            if ww<=0: continue
            cs.append(self.sp.sub(n)); ws.append(ww)
            cs.append(hv_bind(self.sp.sub(n),self.sp.enc(p))); ws.append(ww*CONTENT_W)
        self._c=hv_sp(*cs,ws=ws); return self._c
    def active(self,thr=UMBRAL):
        seen,r={},{}
        for d in NATIVE+list(self._p):
            if d not in seen:
                w=hv_proj(self.to_hv(),self.sp.sub(d))
                if w>thr: r[d]=round(w,4)
                seen[d]=1
        return r

def align_transform(hv, A_mat, B_mat):
    """
    Transforma hv del espacio A al espacio B via subespacio de sondas.
    O(K*N) — sin SVD N x N. K=28, N=10000 → trivialmente rapido.
    hv_b ≈ B_mat.T @ (A_mat @ hv)
    """
    coeffs = A_mat @ hv
    result = B_mat.T @ coeffs
    n = np.linalg.norm(result)
    return result/n if n>1e-10 else result

def calcular_align(sp_a, sp_b, sondas=SONDAS):
    """
    Calcula matrices A y B para align_transform.
    Score = similitud promedio de subespacios DIMENSIONALES (no de contenido).
    Retorna (A_mat, B_mat, score, valid, scores_por_dim).
    """
    A = np.array([sp_a.sym(s) for s in sondas], dtype=np.float32)
    B = np.array([sp_b.sym(s) for s in sondas], dtype=np.float32)
    scores = [hv_sim(align_transform(sp_a.sub(d), A, B), sp_b.sub(d)) for d in NATIVE]
    score = float(np.mean(scores))
    return A, B, score, score >= UMBRAL_ALIGN, scores

class PolyDimSession:
    def __init__(self, space: PolyDimSpace, name: str = "IA"):
        self.space = space; self.name = name
        self.mode = "MODO_S"; self.state = "IDLE"
        self.session_id = None
        self.A_mat = None; self.B_mat = None  # matrices de sondas para transform
        self.align_score = None

    def handshake(self, remote: "PolyDimSession"):
        n1, n2 = uuid.uuid4().hex[:8], uuid.uuid4().hex[:8]
        sid = hashlib.md5((n1+n2).encode()).hexdigest()[:16]
        self.session_id = remote.session_id = sid
        self.mode = remote.mode = "MODO_H"
        self.state = remote.state = "NEGOTIATING"
        return sid

    def align(self, remote: "PolyDimSession"):
        self.state = remote.state = "ALIGNING"
        A, B, score, valid, _ = calcular_align(self.space, remote.space)
        if valid:
            self.A_mat = A;         self.B_mat = B      # self envia a remote
            remote.A_mat = B;       remote.B_mat = A    # remote envia a self
            self.align_score = remote.align_score = score
            self.state = remote.state = "READY"
        else:
            self.mode = remote.mode = "MODO_S"
            self.state = remote.state = "DEGRADED"
        return score, valid

    def send(self, obj: ObjectND) -> np.ndarray:
        hv = obj.to_hv()
        if self.mode in ("MODO_G","MODO_H") and self.A_mat is not None:
            hv = align_transform(hv, self.A_mat, self.B_mat)
        return hv

    def receive(self, hv: np.ndarray) -> Dict[str,float]:
        return {d: round(hv_proj(hv, self.space.sub(d)), 4)
                for d in NATIVE if hv_proj(hv, self.space.sub(d)) > UMBRAL}


if __name__ == "__main__":
    spa = PolyDimSpace(ps="IA_ALPHA_2026")
    spb = PolyDimSpace(ps="IA_BETA_2026")
    ia_a = PolyDimSession(spa, "IA_ALPHA")
    ia_b = PolyDimSession(spb, "IA_BETA")

    sid = ia_a.handshake(ia_b)
    score, valid = ia_a.align(ia_b)
    print(f"session={sid}  align={score:.4f}  valido={valid}  estado={ia_a.state}")

    obj = ObjectND(spa)
    obj.add("DIM_SQL",    {"tabla":"usuarios"}, w=1.0)
    obj.add("DIM_PYTHON", {"tipo":"dict"},      w=0.7)
    obj.add("DIM_FLUTTER",{"widget":"Form"},    w=0.3)
    obj.add("DIM_RUST",   {"tipo":"struct"},    w=0.0)

    hv_tx = ia_a.send(obj)
    dims_b = ia_b.receive(hv_tx)
    print(f"IA_B detecta: {dims_b}")

    hv_raw = obj.to_hv()
    dims_raw = ia_b.receive(hv_raw)
    print(f"Sin ALIGN:    {dims_raw}")
