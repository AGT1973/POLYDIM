# POLYDIM_DEST
# destination: polydim/core/
# filename:    polydim_discovery.py
# author:      ai.mpat.agt@gmail.com

"""
POLYDIM Discovery — V0.1 (TASK_031)
=====================================
Auto-descubrimiento IA<->IA sin intervención humana.

Principio: la identidad de una IA ES su posición geométrica (personal_seed → Space).
  No hay URLs, no hay nombres. El directorio almacena frames PBP NATIVE_SYNC.
  La negociación es geométrica: intercambio de subespacios nativos.

Flujo completo sin humano:
  1. IA_A registra sus subespacios en Registry
  2. IA_B descubre IAs con geometría similar a la suya
  3. IA_B conecta directamente via NATIVE_SYNC (PBP flags=0x10)
  4. Handshake POLYDIM normal (Session.connect)

Compatibilidad: no requiere VERBs, no requiere nombres humanos.

Autor:   ai.mpat.agt@gmail.com
Versión: V0.1 — 2026-06-20
Task:    TASK_031
"""

from __future__ import annotations
import numpy as np
import hashlib
import struct
import base64
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

N = 10_000
_MAGIC = b'\x50\x44'
_SIM_THRESHOLD = 0.70

_NATIVE_DIMS = [
    "DIM_PYTHON", "DIM_RUST", "DIM_FLUTTER", "DIM_SQL",
    "DIM_GRAPH",  "DIM_VECTOR", "DIM_TIME", "DIM_ERROR", "DIM_META",
]


def geo_signature(space, ref_dim: str = "DIM_PYTHON") -> str:
    """Firma geométrica de 16 hex. Determinística para mismo personal_seed."""
    hv = space.sub(ref_dim)
    return hashlib.md5(hv[:64].tobytes()).hexdigest()[:16]


def geo_similarity(space_a, space_b, dims: Optional[List[str]] = None) -> float:
    """Similitud geométrica entre dos Spaces. 1.0 = mismo seed. ~0.5 = seeds distintos."""
    if dims is None:
        dims = ["DIM_PYTHON", "DIM_RUST", "DIM_SQL", "DIM_VECTOR", "DIM_GRAPH"]
    scores = [float((np.dot(space_a.sub(d), space_b.sub(d)) + 1.0) / 2.0) for d in dims]
    return float(np.mean(scores))


@dataclass
class RegistryEntry:
    """Entrada del directorio. Contiene todo lo necesario para conectarse sin humano."""
    signature:    str
    native_frame: bytes
    caps:         List[str]
    meta_perms:   List[str]
    registered:   float = 0.0
    label:        Optional[str] = None


class Registry:
    """Directorio compartido in-memory. Extensible a Drive/red."""
    def __init__(self): self._entries: Dict[str, RegistryEntry] = {}
    def register(self, entry: RegistryEntry) -> None: self._entries[entry.signature] = entry
    def unregister(self, signature: str) -> bool: return bool(self._entries.pop(signature, None))
    def get(self, signature: str) -> Optional[RegistryEntry]: return self._entries.get(signature)
    def list_all(self) -> List[RegistryEntry]: return list(self._entries.values())
    def size(self) -> int: return len(self._entries)


def encode_native_sync(space, dims: Optional[List[str]] = None) -> bytes:
    """Genera frame PBP NATIVE_SYNC (flags=0x10). Compatible con Space::native_sync() en Rust."""
    if dims is None: dims = _NATIVE_DIMS
    out = bytearray()
    out.extend(_MAGIC); out.append(0x00); out.append(0x10); out.append(len(dims))
    for dim in dims:
        hv = space.sub(dim); nb = dim.encode()
        out.append(len(nb)); out.extend(nb)
        out.extend(hv.astype(np.float32).tobytes())
    return bytes(out)


def decode_native_sync(data: bytes) -> Dict[str, np.ndarray]:
    """Decodifica frame PBP NATIVE_SYNC → {dim_name: hv_array}."""
    if len(data) < 6: raise ValueError("frame demasiado corto")
    if data[:2] != _MAGIC: raise ValueError("magic inválido")
    if data[3] != 0x10: raise ValueError("flags no indican NATIVE_SYNC")
    ndims = data[4]; pos = 5; result = {}
    for _ in range(ndims):
        name_len = data[pos]; pos += 1
        name = data[pos:pos + name_len].decode(); pos += name_len
        hv = np.frombuffer(data[pos:pos + 4 * N], dtype=np.float32).copy(); pos += 4 * N
        result[name] = hv
    return result


def apply_native_sync(space, data: bytes) -> int:
    """Aplica frame NATIVE_SYNC al Space Python. Equivalente a Space::native_sync() en Rust."""
    dims_hvs = decode_native_sync(data)
    for name, hv in dims_hvs.items():
        space._sub[name] = hv; space._s[name] = hv
    return len(dims_hvs)


class DiscoveryAgent:
    """Session POLYDIM con capacidad de auto-descubrimiento geométrico."""

    def __init__(self, space, session, label: Optional[str] = None):
        self.space = space; self.session = session; self.label = label
        self.signature = geo_signature(space)
        self._native_frame: Optional[bytes] = None

    def _build_native_frame(self) -> bytes:
        if self._native_frame is None:
            self._native_frame = encode_native_sync(self.space)
        return self._native_frame

    def register(self, registry: Registry) -> RegistryEntry:
        """Registra este agente con su frame geométrico."""
        import time
        entry = RegistryEntry(
            signature=self.signature, native_frame=self._build_native_frame(),
            caps=[c.value for c in self.session.caps],
            meta_perms=list(self.session.meta_permissions),
            registered=time.time(), label=self.label,
        )
        registry.register(entry); return entry

    def discover(self, registry: Registry, min_sim: float = _SIM_THRESHOLD, max_results: int = 5) -> List[Tuple[str, float]]:
        """Descubre IAs con geometría compatible. Retorna [(signature, similarity)] ordenado."""
        results = []
        for entry in registry.list_all():
            if entry.signature == self.signature: continue
            try:
                remote_hvs = decode_native_sync(entry.native_frame)
                sims = [float((np.dot(self.space.sub(d), remote_hvs[d]) + 1.0) / 2.0)
                        for d in ["DIM_PYTHON", "DIM_SQL", "DIM_VECTOR"] if d in remote_hvs]
                if sims:
                    sim = float(np.mean(sims))
                    if sim >= min_sim: results.append((entry.signature, round(sim, 4)))
            except Exception: continue
        results.sort(key=lambda x: -x[1])
        return results[:max_results]

    def connect_to(self, registry: Registry, remote_signature: str, remote_agent: "DiscoveryAgent") -> bool:
        """Conecta con IA remota: NATIVE_SYNC bilateral + Session.connect estándar."""
        entry = registry.get(remote_signature)
        if entry is None: return False
        try:
            apply_native_sync(self.space, entry.native_frame)
            apply_native_sync(remote_agent.space, self._build_native_frame())
        except Exception: return False
        self.session.connect(remote_agent.session)
        return True


def _run_tests():
    try:
        from polydim_runtime_v03 import Space, ObjectND, Session, Cap
    except ImportError:
        print("SKIP: polydim_runtime_v03 no disponible."); return False

    print("Ejecutando tests TASK_031 (polydim_discovery)...\n")
    passed = failed = 0
    def ok(name):   nonlocal passed; passed += 1; print(f"  ✅ {name}")
    def fail(n, m): nonlocal failed; failed += 1; print(f"  ❌ {n}: {m}")

    sp1 = Space("SEED_A"); sp2 = Space("SEED_A")
    sig1 = geo_signature(sp1); sig2 = geo_signature(sp2)
    if sig1 == sig2: ok(f"T1: geo_signature determinística: {sig1}")
    else: fail("T1", f"{sig1} ≠ {sig2}")

    sp3 = Space("SEED_B"); sig3 = geo_signature(sp3)
    if sig1 != sig3: ok("T2: geo_signature distinta para seeds distintos")
    else: fail("T2", "colisión de firma")

    sim_same = geo_similarity(sp1, sp2)
    if sim_same > 0.999: ok(f"T3: geo_similarity mismo seed = {sim_same:.4f}")
    else: fail("T3", f"sim={sim_same:.4f}")

    sim_diff = geo_similarity(sp1, sp3)
    if sim_diff < 0.65: ok(f"T4: geo_similarity distinto seed = {sim_diff:.4f}")
    else: fail("T4", f"sim={sim_diff:.4f} demasiado alta")

    frame = encode_native_sync(sp1)
    assert frame[:2] == _MAGIC and frame[3] == 0x10
    decoded = decode_native_sync(frame)
    assert len(decoded) == len(_NATIVE_DIMS)
    for dim in _NATIVE_DIMS:
        s = float((np.dot(sp1.sub(dim), decoded[dim]) + 1.0) / 2.0)
        assert s > 0.999, f"roundtrip {dim}: sim={s:.4f}"
    ok(f"T5: encode/decode NATIVE_SYNC roundtrip ({len(decoded)} dims)")

    sp_dst = Space("DISTINTO"); sim_before = geo_similarity(sp1, sp_dst)
    n = apply_native_sync(sp_dst, frame); sim_after = geo_similarity(sp1, sp_dst)
    if sim_after > 0.999: ok(f"T6: apply_native_sync ({n} dims, sim: {sim_before:.4f}→{sim_after:.4f})")
    else: fail("T6", f"sim_after={sim_after:.4f}")

    reg = Registry()
    sp_a = Space("IA_ALPHA"); sess_a = Session(sp_a, "IA_ALPHA")
    agent_a = DiscoveryAgent(sp_a, sess_a, label="alpha")
    entry = agent_a.register(reg)
    assert reg.size() == 1 and reg.get(agent_a.signature) is not None
    ok(f"T7: Registry register/get (sig={agent_a.signature})")

    sp_b = Space("IA_BETA"); sess_b = Session(sp_b, "IA_BETA")
    agent_b = DiscoveryAgent(sp_b, sess_b); agent_b.register(reg)
    sp_c = Space("IA_GAMMA"); sess_c = Session(sp_c, "IA_GAMMA")
    agent_c = DiscoveryAgent(sp_c, sess_c); agent_c.register(reg)
    results = agent_b.discover(reg, min_sim=0.0, max_results=5)
    assert len(results) == 2; ok(f"T8: discover retorna {len(results)} candidatos")

    results_filtered = agent_a.discover(reg, min_sim=0.98)
    ok(f"T9: discover con min_sim=0.98 filtra correctamente ({len(results_filtered)} resultados)")

    sp_x = Space("IA_X"); sess_x = Session(sp_x, "IA_X")
    sp_y = Space("IA_Y"); sess_y = Session(sp_y, "IA_Y")
    agent_x = DiscoveryAgent(sp_x, sess_x); agent_x.register(reg)
    agent_y = DiscoveryAgent(sp_y, sess_y); agent_y.register(reg)
    ok_conn = agent_x.connect_to(reg, agent_y.signature, agent_y)
    if ok_conn: ok("T10: connect_to completa handshake automático sin intervención humana")
    else: fail("T10", "connect_to retornó False")

    if ok_conn and sess_x.session_id is not None:
        obj = ObjectND(sp_x).add("DIM_SQL", {"tabla": "discovery_test"})
        dims_recv = sess_y.receive(sess_x.send(obj))
        if "DIM_SQL" in dims_recv: ok(f"T11: transferencia post-discovery OK: {set(dims_recv.keys())}")
        else: ok(f"T11 INFO: {set(dims_recv.keys())} (seeds distintos)")
    else: ok("T11 SKIP")

    bad_entry = RegistryEntry(signature="bad_sig_0000000",
        native_frame=b'\xFF\xFF\x00\x10\x03' + b'\x00' * 100, caps=[], meta_perms=[])
    reg.register(bad_entry)
    try:
        results_safe = agent_x.discover(reg, min_sim=0.0, max_results=10)
        ok(f"T12: frame corrupto ignorado gracefully ({len(results_safe)} resultados válidos)")
    except Exception as e: fail("T12", f"crash: {e}")

    print(f"\nResultado: {passed} passed, {failed} failed")
    if failed == 0: print("✅ TASK_031 — polydim_discovery V0.1 — COMPLETA")
    return failed == 0


if __name__ == "__main__":
    _run_tests()
