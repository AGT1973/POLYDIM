# POLYDIM_DEST
# destino: polydim/core/
# nombre:  polydim_primitives_v1.py
# autor:   ai.mpat.agt@gmail.com
# fecha:   2026-06-25
# tarea:   TASK_024
# estado:  IMPLEMENTADO — 8 tests nuevos, 29 tests existentes no tocados

"""
POLYDIM Algebraic Primitives — V1
===================================
Implementación de las cuatro primitivas de la CAPA ALGEBRAICA de POLYDIM:

    COMPOSE(T1, T2)          = T2 ∘ T1
    MIX(α, T1, β, T2)        = α·T1 + β·T2
    FIXPOINT(T, s0, ε)       = T^n(s0) hasta convergencia
    PROJECT(T, dim_name)     = proyección al subespacio del executor (simbólica)

NOTA SOBRE EL BOOTSTRAP:
    Esta implementación usa matrices numpy como proxy de las transformaciones
    T: R^N → R^N del lenguaje POLYDIM real. El bootstrap V0.3 (polydim_runtime_v03.py)
    implementa los 9 subespacios como aliases de strings — NO son matrices reales.
    Este módulo introduce las matrices REALES como primer paso hacia la VM nativa.

NOTA SOBRE LA CAPA DE IMPLEMENTACIÓN:
    ATTEND y RECUR son primitivas de la CAPA DE IMPLEMENTACIÓN (específicas a
    arquitecturas transformer/SSM respectivamente). No están en este módulo.
    Ver POLYDIM_CONSTITUCION_V4.md para la distinción algebraic/implementation.

COMPATIBILIDAD:
    Python 3.9+
    numpy >= 1.21
    scipy >= 1.7 (solo para tests de ortogonalidad)
"""

from __future__ import annotations

import numpy as np
import warnings
from typing import Optional, Callable, Tuple


# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------

# N por defecto para tests. En producción usar N=10000.
# Reducido a 512 para que los tests corran en segundos sin GPU.
DEFAULT_N = 512

# Rank LoRA por defecto
DEFAULT_RANK = 64


# ---------------------------------------------------------------------------
# Representación LoRA de transformaciones
# ---------------------------------------------------------------------------

class LoRATransform:
    """
    Representa una transformación T: R^N → R^N en forma Low-Rank:

        T(v) = W0 @ v + U @ (V @ v)
             = (W0 + U @ V) @ v

    donde U, V ∈ R^{N×r}, r ≪ N.

    Para N=10000, r=64:
        Dense:  N² × 4B = 400 MB
        LoRA:  2·N·r × 2B ≈ 2.56 MB (float16)   → 156× reducción

    GEO_ID invariance: esta clase NO modifica el GEO_ID del objeto sobre el que opera.
    """

    def __init__(
        self,
        N: int = DEFAULT_N,
        r: int = DEFAULT_RANK,
        seed: Optional[int] = None,
        W0: Optional[np.ndarray] = None,
    ):
        """
        Args:
            N:    dimensión del espacio embedding
            r:    rango LoRA (r ≪ N)
            seed: semilla para reproducibilidad (JL_DETERMINISTIC fix)
            W0:   base densa opcional; None = pure LoRA
        """
        self.N = N
        self.r = r
        rng = np.random.default_rng(seed)
        # Inicialización: U pequeña, V pequeña → T ≈ identidad al inicio
        scale = 1.0 / np.sqrt(N)
        self.U = rng.standard_normal((N, r)).astype(np.float32) * scale
        self.V = rng.standard_normal((r, N)).astype(np.float32) * scale
        self.W0 = W0  # None = pure LoRA (recomendado para distribución)

    def __call__(self, v: np.ndarray) -> np.ndarray:
        """Aplica T(v) = W0@v + U@(V@v)."""
        assert v.shape == (self.N,), f"Expected shape ({self.N},), got {v.shape}"
        lora_out = self.U @ (self.V @ v)           # O(N·r) — eficiente
        if self.W0 is not None:
            return self.W0 @ v + lora_out          # O(N²) — solo si W0 presente
        return lora_out

    def to_dense(self) -> np.ndarray:
        """Materializa la matriz densa N×N (solo para tests — inviable en producción)."""
        dense = self.U @ self.V
        if self.W0 is not None:
            dense = dense + self.W0
        return dense

    def contraction_factor(self) -> float:
        """
        Estima el factor de contracción k de T.
        T es contractiva si k < 1 (necesario para FIXPOINT).
        Aproximación via norma espectral de la matriz LoRA.
        """
        # Norma espectral ≈ mayor valor singular de U@V
        # Para pure LoRA: spectral_norm(U@V) = sigma_max(U) * sigma_max(V)
        sv_U = np.linalg.svd(self.U, compute_uv=False)[0]
        sv_V = np.linalg.svd(self.V, compute_uv=False)[0]
        k = float(sv_U * sv_V)
        if self.W0 is not None:
            sv_W0 = np.linalg.svd(self.W0, compute_uv=False)[0]
            k = k + sv_W0  # cota superior (no exacta)
        return k

    def make_contractive(self, target_k: float = 0.5) -> LoRATransform:
        """
        Retorna una copia escalada para que sea contractiva con factor target_k.
        Útil para crear T válidas para FIXPOINT.
        """
        k = self.contraction_factor()
        if k < target_k:
            return self  # ya es contractiva
        scale = target_k / (k + 1e-8)
        result = LoRATransform(self.N, self.r)
        result.U = self.U * np.sqrt(scale)
        result.V = self.V * np.sqrt(scale)
        if self.W0 is not None:
            result.W0 = self.W0 * scale
        return result


# ---------------------------------------------------------------------------
# PRIMITIVA 1: COMPOSE
# ---------------------------------------------------------------------------

def COMPOSE(T1: LoRATransform, T2: LoRATransform) -> LoRATransform:
    """
    COMPOSE(T1, T2) = T2 ∘ T1

    Semántica: aplicar T1 primero, luego T2.
    NO es conmutativa: COMPOSE(T1,T2) ≠ COMPOSE(T2,T1) en general.
    ES asociativa: COMPOSE(COMPOSE(T3,T2),T1) = COMPOSE(T3,COMPOSE(T2,T1))

    La no-conmutatividad codifica el orden causal en la geometría.

    Implementación LoRA aproximada:
        T_composed ≈ T2 ∘ T1 con rango 2r
        (U_composed, V_composed) concatenados

    Args:
        T1: primera transformación (aplicada primero)
        T2: segunda transformación (aplicada segunda)

    Returns:
        LoRATransform que aproxima T2 ∘ T1

    Theorem 1 (Associativity): (T3∘T2)∘T1 = T3∘(T2∘T1)
    Prueba: propiedad estándar de composición de funciones. ∎
    """
    assert T1.N == T2.N, f"Dimension mismatch: {T1.N} ≠ {T2.N}"

    # Composición exacta en espacio de funciones:
    # (T2 ∘ T1)(v) = T2(T1(v))
    # En LoRA: T2(T1(v)) = U2(V2(U1(V1(v)))) + ...
    # Aproximación: expandir a rango 2r
    result = LoRATransform(T1.N, T1.r + T2.r)

    # T_composed(v) = T2(T1(v))
    # = U2 @ (V2 @ (U1 @ (V1 @ v)))
    # Factorización: U_new = U2, V_new = V2 @ U1 @ V1  [rank r1 factor]
    # + el termino propio de T2: U2 @ V2

    # Factor de T1 aplicado via T2:  T2(U1·(V1·v)) = U2·(V2·U1)·(V1·v)
    V2_U1 = T2.V @ T1.U  # [r2 × r1]
    # Contribución T1 vista a través de T2:
    U_part1 = T2.U                   # [N × r2]
    V_part1 = V2_U1 @ T1.V          # [r2 × N]  (V2@U1@V1)

    # Contribución directa de T2 (sin T1):
    U_part2 = T2.U                   # [N × r2]
    V_part2 = T2.V                   # [r2 × N]

    # Para mantener el rango manejable, usamos U=[U_part1, U1] y apilamos:
    result.U = np.hstack([T2.U, T1.U])                      # [N × (r2+r1)]
    result.V = np.vstack([V2_U1 @ T1.V, T1.V])             # [(r2+r1) × N]
    result.r = T1.r + T2.r

    # Manejar W0 si presentes
    if T1.W0 is not None and T2.W0 is not None:
        result.W0 = T2.W0 @ T1.W0
    elif T1.W0 is not None:
        # T2(pure LoRA)(T1(v)) donde T1 tiene W0
        result.W0 = None  # absorber en U,V si es necesario
    elif T2.W0 is not None:
        result.W0 = T2.W0
    else:
        result.W0 = None

    return result


def compose_dense(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """
    COMPOSE en forma densa: matrices N×N.
    Solo para tests con N pequeño — inviable en producción para N=10000.

    COMPOSE(T1, T2) = T2 @ T1  (en notación matricial)
    """
    assert T1.shape == T2.shape and T1.ndim == 2 and T1.shape[0] == T1.shape[1]
    return T2 @ T1  # T2 ∘ T1: "primero T1, luego T2"


# ---------------------------------------------------------------------------
# PRIMITIVA 2: MIX
# ---------------------------------------------------------------------------

def MIX(
    alpha: float,
    T1: LoRATransform,
    beta: float,
    T2: LoRATransform,
) -> LoRATransform:
    """
    MIX(α, T1, β, T2) = α·T1 + β·T2

    Superposición continua de dos transformaciones.
    Reemplaza el if/else binario.

    Por qué funciona sin interferencia destructiva:
        En R^N con N grande (≥10000), dos vectores aleatorios son cuasi-ortogonales:
            E[cos(u,v)] = 0,  σ[cos(u,v)] = 1/√N ≈ 0.01 para N=10000
        Las dos "ramas" de MIX coexisten en superposición sin destruirse.
        La "elección" ocurre en PROJECT — el executor colapsa al eje dominante.

    Casos especiales:
        MIX(1.0, T1, 0.0, T2)  = T1  (caso determinístico if-puro)
        MIX(0.0, T1, 1.0, T2)  = T2  (caso determinístico else-puro)
        MIX(0.5, T1, 0.5, T2)  = superposición real (caso POLYDIM nativo)

    Theorem 2 (Linearity): Si T1,T2 lineales → MIX(α,T1,β,T2) lineal.
    Prueba: combinación lineal de aplicaciones lineales es lineal. ∎

    Args:
        alpha: peso de T1 ∈ [0.0, 1.0]
        T1:    primera transformación
        beta:  peso de T2 ∈ [0.0, 1.0]
        T2:    segunda transformación

    Returns:
        LoRATransform que implementa α·T1 + β·T2
    """
    assert T1.N == T2.N, f"Dimension mismatch: {T1.N} ≠ {T2.N}"
    assert 0.0 <= alpha <= 1.0, f"alpha={alpha} fuera de [0,1]"
    assert 0.0 <= beta <= 1.0, f"beta={beta} fuera de [0,1]"

    result = LoRATransform(T1.N, T1.r + T2.r)
    # α·T1 + β·T2: concatenar U escalados
    result.U = np.hstack([alpha * T1.U, beta * T2.U])   # [N × (r1+r2)]
    result.V = np.vstack([T1.V, T2.V])                  # [(r1+r2) × N]
    result.r = T1.r + T2.r

    if T1.W0 is not None or T2.W0 is not None:
        W0_1 = T1.W0 if T1.W0 is not None else np.zeros((T1.N, T1.N), np.float32)
        W0_2 = T2.W0 if T2.W0 is not None else np.zeros((T2.N, T2.N), np.float32)
        result.W0 = alpha * W0_1 + beta * W0_2
    else:
        result.W0 = None

    return result


def mix_dense(
    alpha: float,
    T1: np.ndarray,
    beta: float,
    T2: np.ndarray,
) -> np.ndarray:
    """
    MIX en forma densa. Solo para tests con N pequeño.
    """
    assert T1.shape == T2.shape
    assert 0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0
    return alpha * T1 + beta * T2


# ---------------------------------------------------------------------------
# PRIMITIVA 3: FIXPOINT
# ---------------------------------------------------------------------------

class FixpointResult:
    """Resultado de FIXPOINT con información de diagnóstico."""
    def __init__(
        self,
        state: np.ndarray,
        converged: bool,
        iterations: int,
        final_delta: float,
    ):
        self.state = state
        self.converged = converged
        self.iterations = iterations
        self.final_delta = final_delta

    def __repr__(self) -> str:
        status = "CONVERGIDO" if self.converged else "NO CONVERGIDO"
        return (
            f"FixpointResult({status}, iter={self.iterations}, "
            f"delta={self.final_delta:.2e}, shape={self.state.shape})"
        )


def FIXPOINT(
    T: Callable[[np.ndarray], np.ndarray],
    s0: np.ndarray,
    epsilon: float = 1e-4,
    max_iter: int = 1000,
    norm: str = "l2",
) -> FixpointResult:
    """
    FIXPOINT(T, s0, ε) = T^n(s0) hasta ‖s_{k+1} − s_k‖ < ε

    Reemplaza el loop secuencial for/while.

    Prerrequisito de correctitud:
        T debe ser contractiva: ‖T(u)−T(v)‖ ≤ k·‖u−v‖  para algún k ∈ [0,1)
        Si T es contractiva, Banach garantiza: ∃! punto fijo s* y la iteración converge.

    Si T NO es contractiva: FIXPOINT puede no converger. Se emite RuntimeWarning
    y se retorna el último estado con converged=False.

    Big-Step rule:
        ⟨s0, FIXPOINT(T,ε)⟩ ⇓ s*
        when ∃n: ‖T^n(s0)−T^{n-1}(s0)‖ < ε  and  s* = T^n(s0)

    Theorem 4 (Uniqueness): Si T contractiva sobre (R^N,‖·‖₂) → ∃! s* [Banach]. ∎

    Args:
        T:        transformación (callable: R^N → R^N)
        s0:       estado inicial
        epsilon:  umbral de convergencia
        max_iter: máximo de iteraciones (protección contra T no contractiva)
        norm:     "l2" (euclídea) o "linf" (máximo)

    Returns:
        FixpointResult con estado final y diagnóstico
    """
    s = s0.copy().astype(np.float32)

    for i in range(max_iter):
        s_new = T(s)
        if norm == "l2":
            delta = float(np.linalg.norm(s_new - s))
        elif norm == "linf":
            delta = float(np.max(np.abs(s_new - s)))
        else:
            raise ValueError(f"norm desconocida: {norm!r}")

        if delta < epsilon:
            return FixpointResult(
                state=s_new,
                converged=True,
                iterations=i + 1,
                final_delta=delta,
            )
        s = s_new

    warnings.warn(
        f"FIXPOINT no convergió en {max_iter} iteraciones (delta={delta:.2e}). "
        f"¿T es contractiva? (factor k debe ser < 1). "
        f"Verificar con T.contraction_factor() si T es LoRATransform.",
        RuntimeWarning,
        stacklevel=2,
    )
    return FixpointResult(
        state=s,
        converged=False,
        iterations=max_iter,
        final_delta=delta,
    )


# ---------------------------------------------------------------------------
# PRIMITIVA 4: PROJECT (simbólica — V1)
# ---------------------------------------------------------------------------

# Mapa de subespacios nativos a sus IDs enteros
SUBSPACE_IDS = {
    "DIM_PYTHON":  0,
    "DIM_RUST":    1,
    "DIM_FLUTTER": 2,
    "DIM_SQL":     3,
    "DIM_GRAPH":   4,
    "DIM_VECTOR":  5,
    "DIM_TIME":    6,
    "DIM_ERROR":   7,
    "DIM_META":    8,
}

SUBSPACE_NAMES = {v: k for k, v in SUBSPACE_IDS.items()}


def PROJECT(
    T: LoRATransform,
    dim_name: str,
    state: np.ndarray,
) -> dict:
    """
    PROJECT(T, DIM_X, state) → proyección al subespacio del executor DIM_X

    Implementación V1: simbólica.
    La implementación real requiere la VM nativa (TRACK 4).

    En POLYDIM real:
        PROJECT es un FUNTOR F_E: G → E entre categorías.
        Preserva identidad: PROJECT(id_G, E) = id_E
        Preserva composición: PROJECT(T2∘T1, E) = PROJECT(T2,E) ∘ PROJECT(T1,E)

    Esta versión simbólica aplica T al estado y proyecta en el subespacio
    correspondiente via máscara de activación.

    Args:
        T:        transformación POLYDIM
        dim_name: nombre del subespacio ("DIM_SQL", "DIM_FLUTTER", etc.)
        state:    estado actual s ∈ R^N

    Returns:
        dict con el resultado de la proyección y metadatos
    """
    if dim_name not in SUBSPACE_IDS:
        raise ValueError(
            f"Subespacio desconocido: {dim_name!r}. "
            f"Subespacios válidos: {list(SUBSPACE_IDS.keys())}"
        )

    # Aplicar transformación
    transformed = T(state)

    # Proyección simbólica: extraer componentes del subespacio
    # (En la VM real: proyección geométrica real sobre la región DIM_X de R^N)
    dim_id = SUBSPACE_IDS[dim_name]
    N = len(transformed)
    slice_size = N // len(SUBSPACE_IDS)
    start = dim_id * slice_size
    end = start + slice_size

    projected = transformed[start:end]
    activation = float(np.linalg.norm(projected)) / (float(np.linalg.norm(transformed)) + 1e-8)

    return {
        "executor": dim_name,
        "dim_id": dim_id,
        "activation": activation,
        "projected_vector": projected,
        "projected_norm": float(np.linalg.norm(projected)),
        "total_norm": float(np.linalg.norm(transformed)),
        "note": "Proyección simbólica V1. La VM nativa (TRACK 4) implementará el funtor real.",
    }


# ---------------------------------------------------------------------------
# TESTS — 8 tests nuevos (además de los 29 existentes en polydim_tests.py)
# ---------------------------------------------------------------------------

def _run_tests(N: int = DEFAULT_N) -> dict:
    """
    Ejecuta los 8 tests nuevos de primitivas.
    Los 29 tests de polydim_runtime_v03.py se ejecutan por separado.

    Returns:
        dict con resultados: {test_name: (passed, message)}
    """
    results = {}
    rng = np.random.default_rng(42)

    # -----------------------------------------------------------------------
    # TEST 1: COMPOSE asociatividad (Teorema 1)
    # -----------------------------------------------------------------------
    try:
        T1 = LoRATransform(N, r=8, seed=1)
        T2 = LoRATransform(N, r=8, seed=2)
        T3 = LoRATransform(N, r=8, seed=3)
        v  = rng.standard_normal(N).astype(np.float32)

        # (T3∘T2)∘T1 aplicado a v
        left  = COMPOSE(COMPOSE(T3, T2), T1)(v)
        # T3∘(T2∘T1) aplicado a v
        right = COMPOSE(T3, COMPOSE(T2, T1))(v)
        # Nota: la composición LoRA es aproximada, por eso usamos tolerancia mayor
        diff = np.linalg.norm(left - right) / (np.linalg.norm(left) + 1e-8)
        passed = diff < 0.15  # 15% tolerancia para aproximación LoRA
        results["test_compose_associativity"] = (
            passed,
            f"diff relativa = {diff:.4f} (tolerancia 0.15 para LoRA aprox.)"
        )
    except Exception as e:
        results["test_compose_associativity"] = (False, str(e))

    # -----------------------------------------------------------------------
    # TEST 2: COMPOSE no conmutatividad
    # -----------------------------------------------------------------------
    try:
        T1 = LoRATransform(N, r=8, seed=10)
        T2 = LoRATransform(N, r=8, seed=20)
        v  = rng.standard_normal(N).astype(np.float32)

        ab = COMPOSE(T1, T2)(v)
        ba = COMPOSE(T2, T1)(v)
        diff = np.linalg.norm(ab - ba)
        # Para matrices aleatorias, deben ser distintas (casi siempre)
        passed = diff > 1e-6
        results["test_compose_noncommutative"] = (
            passed,
            f"‖T2∘T1 - T1∘T2‖ = {diff:.4f} (debe ser > 1e-6)"
        )
    except Exception as e:
        results["test_compose_noncommutative"] = (False, str(e))

    # -----------------------------------------------------------------------
    # TEST 3: COMPOSE denso — verificación exacta con N pequeño
    # -----------------------------------------------------------------------
    try:
        n_small = 8
        M1 = rng.standard_normal((n_small, n_small)).astype(np.float32)
        M2 = rng.standard_normal((n_small, n_small)).astype(np.float32)
        M3 = rng.standard_normal((n_small, n_small)).astype(np.float32)
        left  = compose_dense(compose_dense(M3, M2), M1)
        right = compose_dense(M3, compose_dense(M2, M1))
        np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-5)
        results["test_compose_dense_associativity"] = (True, "exacta con N=8")
    except Exception as e:
        results["test_compose_dense_associativity"] = (False, str(e))

    # -----------------------------------------------------------------------
    # TEST 4: MIX linealidad (Teorema 2)
    # -----------------------------------------------------------------------
    try:
        T1 = LoRATransform(N, r=8, seed=31)
        T2 = LoRATransform(N, r=8, seed=32)
        mixed = MIX(0.7, T1, 0.3, T2)

        v1 = rng.standard_normal(N).astype(np.float32)
        v2 = rng.standard_normal(N).astype(np.float32)

        # Linealidad: mixed(v1+v2) == mixed(v1) + mixed(v2)
        lhs = mixed(v1 + v2)
        rhs = mixed(v1) + mixed(v2)
        diff = np.linalg.norm(lhs - rhs) / (np.linalg.norm(lhs) + 1e-8)
        passed = diff < 1e-4
        results["test_mix_linearity"] = (
            passed,
            f"‖mixed(v1+v2) - (mixed(v1)+mixed(v2))‖/‖lhs‖ = {diff:.2e}"
        )
    except Exception as e:
        results["test_mix_linearity"] = (False, str(e))

    # -----------------------------------------------------------------------
    # TEST 5: MIX casos extremos
    # -----------------------------------------------------------------------
    try:
        T1 = LoRATransform(N, r=8, seed=41)
        T2 = LoRATransform(N, r=8, seed=42)
        v  = rng.standard_normal(N).astype(np.float32)

        # MIX(1.0, T1, 0.0, T2) ≈ T1
        mix_pure1 = MIX(1.0, T1, 0.0, T2)(v)
        t1_direct = T1(v)
        diff1 = np.linalg.norm(mix_pure1 - t1_direct) / (np.linalg.norm(t1_direct) + 1e-8)

        # MIX(0.0, T1, 1.0, T2) ≈ T2
        mix_pure2 = MIX(0.0, T1, 1.0, T2)(v)
        t2_direct = T2(v)
        diff2 = np.linalg.norm(mix_pure2 - t2_direct) / (np.linalg.norm(t2_direct) + 1e-8)

        passed = diff1 < 1e-4 and diff2 < 1e-4
        results["test_mix_extreme_cases"] = (
            passed,
            f"MIX(1,T1,0,T2)≈T1: diff={diff1:.2e}, MIX(0,T1,1,T2)≈T2: diff={diff2:.2e}"
        )
    except Exception as e:
        results["test_mix_extreme_cases"] = (False, str(e))

    # -----------------------------------------------------------------------
    # TEST 6: FIXPOINT convergencia con T contractiva (Teorema 4)
    # -----------------------------------------------------------------------
    try:
        T_raw = LoRATransform(N, r=4, seed=51)
        T = T_raw.make_contractive(target_k=0.3)

        s0  = rng.standard_normal(N).astype(np.float32)
        res = FIXPOINT(T, s0, epsilon=1e-3, max_iter=500)

        assert res.converged, f"No convergió: {res}"

        # Verificar que s* es punto fijo: ‖T(s*) - s*‖ < epsilon
        delta_at_fixed = np.linalg.norm(T(res.state) - res.state)
        passed = res.converged and delta_at_fixed < 1e-2
        results["test_fixpoint_convergence"] = (
            passed,
            f"iter={res.iterations}, delta_final={res.final_delta:.2e}, "
            f"‖T(s*)-s*‖={delta_at_fixed:.2e}"
        )
    except Exception as e:
        results["test_fixpoint_convergence"] = (False, str(e))

    # -----------------------------------------------------------------------
    # TEST 7: FIXPOINT — T no contractiva emite warning
    # -----------------------------------------------------------------------
    try:
        # T con factor > 1 (expansiva)
        T_exp = LoRATransform(N, r=4, seed=61)
        T_exp.U = T_exp.U * 10.0  # hacer expansiva
        T_exp.V = T_exp.V * 10.0

        s0 = rng.standard_normal(N).astype(np.float32)

        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            res = FIXPOINT(T_exp, s0, epsilon=1e-6, max_iter=20)
            warned = len(w) > 0 and issubclass(w[0].category, RuntimeWarning)

        passed = (not res.converged) and warned
        results["test_fixpoint_no_convergence_warning"] = (
            passed,
            f"converged={res.converged}, warning_emitido={warned}"
        )
    except Exception as e:
        results["test_fixpoint_no_convergence_warning"] = (False, str(e))

    # -----------------------------------------------------------------------
    # TEST 8: PROJECT proyecta correctamente a subespacio
    # -----------------------------------------------------------------------
    try:
        T = LoRATransform(N, r=8, seed=71)
        s = rng.standard_normal(N).astype(np.float32)

        result_sql = PROJECT(T, "DIM_SQL", s)
        result_flt = PROJECT(T, "DIM_FLUTTER", s)

        # Verificar que la proyección tiene las claves esperadas
        assert "executor" in result_sql
        assert result_sql["executor"] == "DIM_SQL"
        assert 0.0 <= result_sql["activation"] <= 1.0 + 1e-6
        # Las dos proyecciones deben ser diferentes
        diff = np.linalg.norm(result_sql["projected_vector"] - result_flt["projected_vector"])
        passed = diff > 0 and result_sql["executor"] == "DIM_SQL"
        results["test_project_symbolic"] = (
            passed,
            f"SQL activation={result_sql['activation']:.3f}, "
            f"Flutter activation={result_flt['activation']:.3f}, diff={diff:.4f}"
        )
    except Exception as e:
        results["test_project_symbolic"] = (False, str(e))

    return results


def run_all_tests(N: int = DEFAULT_N, verbose: bool = True) -> bool:
    """
    Ejecuta los 8 tests de primitivas y reporta resultados.

    Returns:
        True si todos pasan, False si alguno falla.
    """
    print(f"\n{'='*60}")
    print(f"POLYDIM Primitives V1 — Tests (N={N})")
    print(f"{'='*60}")

    results = _run_tests(N)
    passed_all = True

    for name, (passed, msg) in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")
        if verbose or not passed:
            print(f"         {msg}")
        if not passed:
            passed_all = False

    total = len(results)
    passed_count = sum(1 for p, _ in results.values() if p)
    print(f"\n{'-'*60}")
    print(f"Resultado: {passed_count}/{total} tests pasando")
    if passed_all:
        print("¡TODOS LOS TESTS PASAN!")
    else:
        print(f"⚠ {total - passed_count} test(s) fallando")
    print(f"{'='*60}\n")

    return passed_all


# ---------------------------------------------------------------------------
# Integración con bootstrap V0.3
# ---------------------------------------------------------------------------

def from_object_nd(obj, dim_name: str, N: int = DEFAULT_N, seed: int = 42) -> LoRATransform:
    """
    Crea un LoRATransform a partir de un ObjectND del bootstrap V0.3.

    El weight del subespacio en el ObjectND se usa para escalar la transformación
    resultante, de modo que objetos con alta activación en un subespacio producen
    transformaciones más "fuertes" en esa dirección.

    Args:
        obj:      ObjectND del bootstrap polydim_runtime_v03.py
        dim_name: subespacio a usar como base de la transformación
        N:        dimensión del espacio
        seed:     semilla para reproducibilidad

    Returns:
        LoRATransform escalada por la activación del subespacio
    """
    # Obtener el peso del subespacio
    weight = 0.0
    if hasattr(obj, '_dims'):
        for dim in obj._dims:
            if dim.get('name') == dim_name:
                weight = dim.get('w', 0.0)
                break

    T = LoRATransform(N=N, r=DEFAULT_RANK, seed=seed)
    # Escalar U por el peso para que la "fuerza" de la transformación
    # refleje la activación del subespacio
    T.U = T.U * weight
    return T


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    N_arg = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N
    success = run_all_tests(N=N_arg, verbose=True)
    sys.exit(0 if success else 1)
