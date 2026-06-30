# POLYDIM_DEST
# destination: polydim/core/
# filename:    polydim_rng.py
# author:      ai.mpat.agt@gmail.com

"""
POLYDIM Canonical Random Generator — PCRG-1
============================================
Algoritmo determinístico, sin dependencias externas, implementable
idénticamente en Python, Rust, Dart, Go, o cualquier lenguaje.

POLYDIM no es una capa sobre Python/Rust/Flutter.
Los reemplaza: el objeto multi-dimensional ES el programa.
PCRG-1 garantiza que cualquier runtime POLYDIM produce
hipervectores idénticos para la misma clave.

Algoritmo (especificación canónica):
  1. FNV-1a 64-bit sobre key.encode('utf-8') → dos estados u64
  2. xorshift128+ para generación de bits
  3. Mapeo uniforme: (u64 + 0.5) / (2^64)  → f64 en (0, 1)
  4. Box-Muller → pares Gaussianos f32
  5. Normalización L2

Parámetros fijos (no cambiar — breaking change si se modifican):
  FNV_OFFSET = 14695981039346656037
  FNV_PRIME  = 1099511628211
  XS_A = 23  (shift left  en xorshift128+)
  XS_B = 17  (shift right en xorshift128+)
  XS_C = 26  (shift right en xorshift128+)
  STATE_SALT = 0xDEADBEEFCAFEBABE  (para el segundo estado)

Equivalencia Rust → ver polydim_core.rs :: mod pcrg

Autor:   ai.mpat.agt@gmail.com
Versión: V1.0 — 2026-06-17
"""

from __future__ import annotations
import math
import numpy as np

# ─────────────────────────────────────────────────────────────
# Parámetros canónicos PCRG-1 — NO MODIFICAR
# ─────────────────────────────────────────────────────────────

_FNV_OFFSET: int = 14695981039346656037
_FNV_PRIME:  int = 1099511628211
_U64_MASK:   int = 0xFFFFFFFF_FFFFFFFF
_U64_DENOM: float = float(0xFFFFFFFF_FFFFFFFF) + 1.0   # 2^64
_STATE_SALT: int = 0xDEADBEEF_CAFEBABE


# ─────────────────────────────────────────────────────────────
# Paso 1: semilla — FNV-1a 64-bit
# ─────────────────────────────────────────────────────────────

def _fnv1a_64(data: bytes, seed: int = _FNV_OFFSET) -> int:
    """FNV-1a 64-bit. Mismo algoritmo que en Rust/Dart/Go."""
    h = seed
    for b in data:
        h ^= b
        h = (h * _FNV_PRIME) & _U64_MASK
    return h


def pcrg_init(key: str) -> list[int]:
    """
    Inicializa el estado xorshift128+ desde un string.
    Retorna [s0, s1] — dos u64 no-nulos.
    """
    kb = key.encode('utf-8')
    s0 = _fnv1a_64(kb)
    s1 = _fnv1a_64(kb, (s0 ^ _STATE_SALT) & _U64_MASK)
    return [s0 if s0 else 1, s1 if s1 else 2]


# ─────────────────────────────────────────────────────────────
# Paso 2: generador — xorshift128+
# ─────────────────────────────────────────────────────────────

def _pcrg_next(state: list[int]) -> int:
    """
    xorshift128+ — mismos shifts que la especificación canónica.
    Modifica state in-place, retorna u64.
    """
    s1 = state[0]
    s0 = state[1]
    state[0] = s0
    s1 ^= (s1 << 23) & _U64_MASK   # XS_A = 23
    s1 ^= (s1 >> 17)                # XS_B = 17
    s1 ^= s0
    s1 ^= (s0 >> 26)                # XS_C = 26
    state[1] = s1
    return (state[0] + state[1]) & _U64_MASK


def _pcrg_uniform(state: list[int]) -> float:
    """Mapeo u64 → f64 en (0, 1) abierto. +0.5 evita 0.0 exacto."""
    return (_pcrg_next(state) + 0.5) / _U64_DENOM


# ─────────────────────────────────────────────────────────────
# Paso 3: Box-Muller → par Gaussiano
# ─────────────────────────────────────────────────────────────

def _gaussian_pair(state: list[int]) -> tuple[float, float]:
    u1 = _pcrg_uniform(state)
    u2 = _pcrg_uniform(state)
    r     = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    return r * math.cos(theta), r * math.sin(theta)


# ─────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────

def polydim_make_hv(key: str, n: int = 10_000) -> np.ndarray:
    """
    Genera un hipervector float32 normalizado.

    Determinístico: misma key + mismo n → mismo HV en cualquier runtime
    que implemente PCRG-1 (Python, Rust, Dart...).

    Parámetros:
        key  — string identificador (ej. "DIM_SQL", "IA_A:DIM_RUST")
        n    — dimensión del hipervector (default 10000)

    Retorna:
        np.ndarray float32 de forma (n,), norma = 1.0
    """
    state = pcrg_init(key)
    values: list[float] = []
    while len(values) < n:
        a, b = _gaussian_pair(state)
        values.append(a)
        if len(values) < n:
            values.append(b)
    arr = np.array(values, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 1e-10 else arr


def polydim_seed(key: str) -> int:
    """
    Retorna un seed u64 reproducible para operaciones auxiliares.
    Compatible con np.random.default_rng(seed) si se necesita.
    """
    return _fnv1a_64(key.encode('utf-8'))


# ─────────────────────────────────────────────────────────────
# Tests de referencia — valores esperados para validar Rust/Dart
# ─────────────────────────────────────────────────────────────

def reference_vectors() -> dict:
    """
    Vectores de referencia para verificar implementaciones en otros lenguajes.
    Los primeros 4 floats de cada HV deben coincidir exactamente.
    """
    cases = ["DIM_SQL", "DIM_RUST", "DIM_PYTHON", "DIM_FLUTTER", ""]
    return {
        key: polydim_make_hv(key, 10)[:4].tolist()
        for key in cases
    }


def _run_tests():
    print("Tests PCRG-1...\n")
    passed = failed = 0

    def ok(n): 
        nonlocal passed; passed += 1; print(f"  ✅ {n}")
    def fail(n, m): 
        nonlocal failed; failed += 1; print(f"  ❌ {n}: {m}")

    # T1: determinismo — misma key produce mismo HV
    hv1 = polydim_make_hv("DIM_SQL")
    hv2 = polydim_make_hv("DIM_SQL")
    if np.allclose(hv1, hv2):
        ok("T1: determinismo")
    else:
        fail("T1", "HVs distintos para misma key")

    # T2: normalización
    norm = float(np.linalg.norm(hv1))
    if abs(norm - 1.0) < 1e-5:
        ok(f"T2: norma=1.0 (actual={norm:.6f})")
    else:
        fail("T2", f"norma={norm}")

    # T3: keys distintas → HVs distintos (casi ortogonales)
    hv_sql  = polydim_make_hv("DIM_SQL")
    hv_rust = polydim_make_hv("DIM_RUST")
    sim = float((np.dot(hv_sql, hv_rust) + 1.0) / 2.0)
    if 0.47 < sim < 0.53:
        ok(f"T3: ortogonalidad keys distintas (sim={sim:.4f})")
    else:
        fail("T3", f"sim={sim:.4f} fuera de [0.47, 0.53]")

    # T4: personal_seed cambia HV
    hv_a = polydim_make_hv("IA_A:DIM_SQL")
    hv_b = polydim_make_hv("IA_B:DIM_SQL")
    sim2 = float((np.dot(hv_a, hv_b) + 1.0) / 2.0)
    if 0.47 < sim2 < 0.53:
        ok(f"T4: seeds distintos → HVs distintos (sim={sim2:.4f})")
    else:
        fail("T4", f"sim={sim2:.4f}")

    # T5: N variable
    hv_small = polydim_make_hv("TEST", n=100)
    if len(hv_small) == 100 and abs(np.linalg.norm(hv_small) - 1.0) < 1e-4:
        ok("T5: N variable funciona")
    else:
        fail("T5", f"len={len(hv_small)}")

    # T6: valores de referencia para Rust (imprimir para comparar)
    print("\n  Vectores de referencia (primeros 4 floats):")
    for key, vals in reference_vectors().items():
        label = repr(key) if key else '""'
        print(f"    {label:20} → {[round(v, 6) for v in vals]}")

    print(f"\nResultado: {passed} passed, {failed} failed\n")
    return failed == 0


if __name__ == "__main__":
    _run_tests()
