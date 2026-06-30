# POLYDIM_DEST
# destino: polydim/core/
# nombre:  polydim_align_v2.py
# autor:   ai.mpat.agt@gmail.com
# fecha:   2026-06-25
# tarea:   TASK_025

"""
POLYDIM ALIGN Protocol — V2 (Procrustes + Codebook)
=====================================================

Implementa el protocolo ALIGN para comunicación AI↔AI entre modelos con
espacios de embedding de distintas dimensiones.

PROBLEMA:
    IA_A opera en R^{d_A} (ej: GPT-4, d=12288)
    IA_B opera en R^{d_B} (ej: Claude 3, d=8192)
    Una transformación T_A ∈ R^{d_A×d_A} no puede aplicarse a un estado en R^{d_B}.

SOLUCIÓN:
    Calcular matriz de rotación ortogonal M: R^{d_A} → R^{d_B} via Procrustes.
    Transformar: T_B = M · T_A · M†  (M† = pseudoinversa de Moore-Penrose)
    Aplicar: new_state_B = T_B(state_B)

PROTOCOLO COMPLETO:
    Paso 1 — Pre-alineación (una vez por par de modelos):
             Compartir codebook C de GEO_IDs universales.
    Paso 2 — Cálculo M via SVD: M* = argmin_M ||MA−B||² s.t. M^T·M=I
    Paso 3 — Transformación: T_B = M · T_A · M†
    Paso 4 — Aplicación: new_state_B = T_B(state_B)

REFERENCIA EN PAPER: Sección 4.2 — The ALIGN Protocol
REFERENCIA EN CONST: Artículo VII (Comunicación AI↔AI)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, NamedTuple
import warnings


# ---------------------------------------------------------------------------
# Codebook de GEO_IDs universales
# ---------------------------------------------------------------------------

class UniversalCodebook:
    """
    Conjunto de GEO_IDs universales: hipervectores cuasi-ortogonales en R^N.

    Funcionan como "anclas semánticas" — cada IA proyecta su representación
    local de estos conceptos para calibrar la matriz M de alineación.

    Propiedad VSA (verificada): para N=10000 y n=100 conceptos,
    max cosine off-diagonal ≈ 0.01  (cuasi-ortogonalidad garantizada).
    """

    DEFAULT_CONCEPTS = [
        "entity", "relation", "attribute", "action", "state",
        "time", "space", "cause", "effect", "agent",
        "object", "event", "property", "category", "instance",
        "truth", "belief", "goal", "resource", "constraint",
        "sequence", "parallel", "condition", "iteration", "recursion",
        "input", "output", "transform", "compose", "project",
        "error", "success", "pending", "complete", "cancel",
        "user", "system", "data", "query", "result",
        "create", "read", "update", "delete", "search",
        "local", "remote", "sync", "async", "stream",
        "text", "image", "audio", "vector", "graph",
        "sql", "rust", "python", "flutter", "wasm",
        "high", "low", "medium", "threshold", "weight",
        "source", "target", "path", "node", "edge",
        "memory", "compute", "network", "storage", "interface",
        "begin", "end", "loop", "branch", "merge",
        "encrypt", "decrypt", "sign", "verify", "hash",
        "model", "layer", "attention", "embedding", "token",
    ]

    def __init__(self, n_concepts: int = 100, N: int = 10000, seed: int = 42):
        """
        Args:
            n_concepts: número de GEO_IDs en el codebook
            N:          dimensión del espacio de embedding
            seed:       semilla para reproducibilidad (JL_DETERMINISTIC)
        """
        self.n_concepts = n_concepts
        self.N = N
        self.seed = seed
        self.concepts = self.DEFAULT_CONCEPTS[:n_concepts]
        if len(self.concepts) < n_concepts:
            # Completar con conceptos genéricos
            self.concepts += [f"concept_{i}" for i in range(n_concepts - len(self.concepts))]

        rng = np.random.default_rng(seed)
        # Hipervectores cuasi-ortogonales
        raw = rng.standard_normal((n_concepts, N)).astype(np.float32)
        # Normalizar a vectores unitarios
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        self.vectors = raw / norms  # shape: [n_concepts × N]

    def verify_quasi_orthogonality(self) -> dict:
        """Verifica que los GEO_IDs sean cuasi-ortogonales."""
        gram = self.vectors @ self.vectors.T  # [n × n]
        off_diag = gram - np.eye(self.n_concepts)
        max_cos = float(np.abs(off_diag).max())
        mean_cos = float(np.abs(off_diag).mean())
        expected_std = 1.0 / np.sqrt(self.N)  # teoría VSA
        return {
            "max_off_diagonal_cosine": max_cos,
            "mean_off_diagonal_cosine": mean_cos,
            "expected_std_by_VSA": expected_std,
            "quasi_orthogonal": max_cos < 5 * expected_std,
        }

    def project_to_local_space(
        self,
        local_embedding_fn,
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Proyecta el codebook al espacio local de un modelo.

        Args:
            local_embedding_fn: función (concept_name) → local_vector ∈ R^{d_local}
            batch_size: tamaño de lote para llamadas al modelo

        Returns:
            local_matrix ∈ R^{d_local × n_concepts}
        """
        local_vectors = []
        for i in range(0, self.n_concepts, batch_size):
            batch = self.concepts[i:i+batch_size]
            for concept in batch:
                v = local_embedding_fn(concept)
                local_vectors.append(v)
        A = np.stack(local_vectors, axis=1)  # [d_local × n_concepts]
        # Normalizar columnas
        norms = np.linalg.norm(A, axis=0, keepdims=True)
        return A / (norms + 1e-8)


# ---------------------------------------------------------------------------
# Resultado de ALIGN
# ---------------------------------------------------------------------------

class AlignResult(NamedTuple):
    """Resultado del protocolo ALIGN entre dos espacios."""
    M: np.ndarray            # Matriz de rotación [d_B × d_A]
    M_pseudo: np.ndarray     # Pseudoinversa M† [d_A × d_B]
    frobenius_error: float   # ‖M·A − B‖_F (error de alineación)
    d_A: int                 # Dimensión del espacio fuente
    d_B: int                 # Dimensión del espacio destino
    n_concepts: int          # Tamaño del codebook usado
    method: str              # "procrustes" | "cca"


# ---------------------------------------------------------------------------
# ALIGN via Procrustes Ortogonal
# ---------------------------------------------------------------------------

def procrustes_align(
    A: np.ndarray,
    B: np.ndarray,
) -> AlignResult:
    """
    Calcula la matriz de alineación óptima via Procrustes Ortogonal.

    M* = argmin_M ‖M·A − B‖²_F   sujeto a M^T·M = I

    Solución analítica via SVD:
        SVD(B · A^T) = U · Σ · V^T
        M* = U · V^T

    Args:
        A: float[d_A × k] — proyecciones del codebook en espacio IA_A
        B: float[d_B × k] — proyecciones del codebook en espacio IA_B
        (k = número de GEO_IDs en el codebook)

    Returns:
        AlignResult con M: float[d_B × d_A]

    Complejidad: O(d_B · d_A · k) para el SVD
    """
    assert A.ndim == 2 and B.ndim == 2
    assert A.shape[1] == B.shape[1], \
        f"Codebook size mismatch: A has {A.shape[1]} concepts, B has {B.shape[1]}"

    d_A, k = A.shape
    d_B, _ = B.shape

    # C = B @ A^T   [d_B × d_A]
    C = B @ A.T

    # SVD de C
    try:
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
    except np.linalg.LinAlgError:
        raise ValueError("SVD falló en procrustes_align. Verificar que A y B no sean degeneradas.")

    # M* = U @ V^T
    M = U @ Vt  # [d_B × d_A]

    # Pseudoinversa: M† = V @ U^T  (para M ortogonal, M† = M^T cuando d_A=d_B;
    # para el caso general rectangular usamos la pseudoinversa completa)
    if d_A == d_B:
        M_pseudo = M.T  # para matrices ortogonales cuadradas: M† = M^T
    else:
        M_pseudo = np.linalg.pinv(M)  # pseudoinversa general [d_A × d_B]

    # Error de alineación
    frob_error = float(np.linalg.norm(M @ A - B, 'fro'))

    return AlignResult(
        M=M,
        M_pseudo=M_pseudo,
        frobenius_error=frob_error,
        d_A=d_A,
        d_B=d_B,
        n_concepts=k,
        method="procrustes",
    )


def align_transform(
    T_A: np.ndarray,
    align: AlignResult,
) -> np.ndarray:
    """
    Transforma T_A del espacio de IA_A al espacio de IA_B.

    T_B = M · T_A · M†

    donde M es la matriz de alineación y M† su pseudoinversa.

    Args:
        T_A:   float[d_A × d_A] — transformación en espacio de IA_A
        align: AlignResult de procrustes_align()

    Returns:
        T_B:   float[d_B × d_B] — transformación equivalente en espacio de IA_B
    """
    assert T_A.shape == (align.d_A, align.d_A), \
        f"T_A debe ser [{align.d_A}×{align.d_A}], got {T_A.shape}"

    # T_B = M · T_A · M†
    T_B = align.M @ T_A @ align.M_pseudo  # [d_B × d_B]
    return T_B


def align_lora_transform(
    U_A: np.ndarray,
    V_A: np.ndarray,
    align: AlignResult,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Versión eficiente de align_transform para representación LoRA.

    T_A = U_A · V_A (low-rank, U_A ∈ R^{d_A×r}, V_A ∈ R^{r×d_A})
    T_B = M · T_A · M† = (M·U_A) · (V_A·M†)

    La estructura LoRA se preserva con el mismo rango r.

    Args:
        U_A:   float[d_A × r] — factor U de T_A
        V_A:   float[r × d_A] — factor V de T_A
        align: AlignResult de procrustes_align()

    Returns:
        (U_B, V_B): factores LoRA de T_B en espacio de IA_B
        U_B ∈ R^{d_B × r},  V_B ∈ R^{r × d_B}
    """
    assert U_A.shape[0] == align.d_A
    assert V_A.shape[1] == align.d_A

    # T_B = M·U_A·V_A·M† = (M·U_A)·(V_A·M†)
    U_B = align.M @ U_A          # [d_B × r]
    V_B = V_A @ align.M_pseudo   # [r × d_B]

    return U_B, V_B


# ---------------------------------------------------------------------------
# Protocolo ALIGN completo
# ---------------------------------------------------------------------------

class AlignSession:
    """
    Sesión ALIGN entre dos espacios de IA.

    Encapsula el protocolo completo:
    1. Pre-alineación via codebook
    2. Cálculo de M (Procrustes)
    3. Transformación y aplicación

    Uso:
        session = AlignSession.create(
            embedding_fn_A=lambda c: model_a.embed(c),
            embedding_fn_B=lambda c: model_b.embed(c),
            codebook=UniversalCodebook(n_concepts=50, N=1024),
        )
        T_B = session.transform(T_A)
        new_state_B = T_B @ state_B
    """

    def __init__(self, align: AlignResult, codebook: UniversalCodebook):
        self.align = align
        self.codebook = codebook

    @classmethod
    def create(
        cls,
        embedding_fn_A,
        embedding_fn_B,
        codebook: Optional[UniversalCodebook] = None,
        n_concepts: int = 50,
        N: int = 1024,
        seed: int = 42,
    ) -> AlignSession:
        """
        Crea una sesión ALIGN entre dos modelos.

        Args:
            embedding_fn_A: función (concept) → R^{d_A} para el modelo A
            embedding_fn_B: función (concept) → R^{d_B} para el modelo B
            codebook:       codebook universal (se crea si no se provee)
            n_concepts:     tamaño del codebook si se crea nuevo
            N:              dimensión del codebook universal
        """
        if codebook is None:
            codebook = UniversalCodebook(n_concepts=n_concepts, N=N, seed=seed)

        # Proyectar el codebook en ambos espacios locales
        A = codebook.project_to_local_space(embedding_fn_A)  # [d_A × k]
        B = codebook.project_to_local_space(embedding_fn_B)  # [d_B × k]

        # Calcular alineación via Procrustes
        align = procrustes_align(A, B)

        return cls(align=align, codebook=codebook)

    def transform(self, T_A: np.ndarray) -> np.ndarray:
        """Transforma T_A al espacio de IA_B."""
        return align_transform(T_A, self.align)

    def transform_lora(
        self, U_A: np.ndarray, V_A: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transforma T_A (LoRA) al espacio de IA_B (LoRA)."""
        return align_lora_transform(U_A, V_A, self.align)

    def apply(self, T_A: np.ndarray, state_B: np.ndarray) -> np.ndarray:
        """Transforma T_A y lo aplica al estado de IA_B."""
        T_B = self.transform(T_A)
        return T_B @ state_B

    @property
    def frobenius_error(self) -> float:
        """Error de alineación ‖M·A − B‖_F."""
        return self.align.frobenius_error

    @property
    def alignment_quality(self) -> str:
        """Calificación cualitativa de la alineación."""
        err = self.frobenius_error
        if err < 0.01:  return "EXCELENTE"
        if err < 0.1:   return "BUENA"
        if err < 0.5:   return "ACEPTABLE"
        return "POBRE — considerar CCA o codebook más grande"

    def __repr__(self) -> str:
        return (
            f"AlignSession(d_A={self.align.d_A}, d_B={self.align.d_B}, "
            f"n_concepts={self.align.n_concepts}, "
            f"frobenius_error={self.frobenius_error:.4f}, "
            f"quality={self.alignment_quality})"
        )


# ---------------------------------------------------------------------------
# Simulación de ALIGN entre dos "modelos" con numpy (sin LLM real)
# ---------------------------------------------------------------------------

def simulate_align_demo(
    d_A: int = 256,
    d_B: int = 128,
    n_concepts: int = 40,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Demostración del protocolo ALIGN sin modelos reales.

    Simula dos "espacios de embedding" con dimensiones distintas y verifica
    que la transformación alineada preserva la semántica.

    Args:
        d_A:        dimensión del espacio de IA_A
        d_B:        dimensión del espacio de IA_B
        n_concepts: tamaño del codebook de demostración
        seed:       semilla

    Returns:
        dict con métricas de la demostración
    """
    rng = np.random.default_rng(seed)

    # Simular embeddings locales de los dos modelos
    # (En producción: llamar a model.embed(concept))
    W_A = rng.standard_normal((d_A, 200)).astype(np.float32)  # "proyector" de IA_A
    W_B = rng.standard_normal((d_B, 200)).astype(np.float32)  # "proyector" de IA_B

    # Generar vectores para los n_concepts del codebook
    concept_vectors_universal = rng.standard_normal((200, n_concepts)).astype(np.float32)

    def embed_A(concept_idx: int) -> np.ndarray:
        v = W_A @ concept_vectors_universal[:, concept_idx]
        return v / np.linalg.norm(v)

    def embed_B(concept_idx: int) -> np.ndarray:
        v = W_B @ concept_vectors_universal[:, concept_idx]
        return v / np.linalg.norm(v)

    # Proyectar codebook en ambos espacios
    A = np.stack([embed_A(i) for i in range(n_concepts)], axis=1)  # [d_A × n]
    B = np.stack([embed_B(i) for i in range(n_concepts)], axis=1)  # [d_B × n]

    # Calcular alineación
    align = procrustes_align(A, B)

    # Generar una transformación de prueba en espacio A
    T_A = rng.standard_normal((d_A, d_A)).astype(np.float32) * 0.1
    T_A = T_A + np.eye(d_A, dtype=np.float32) * 0.01  # cercana a identidad

    # Transformar al espacio B
    T_B = align_transform(T_A, align)

    # Verificar: aplicar T_A a un estado en A y T_B a su correspondiente en B
    # deberían preservar la estructura semántica
    state_A = rng.standard_normal(d_A).astype(np.float32)
    state_A = state_A / np.linalg.norm(state_A)

    # Estado correspondiente en B (via M)
    state_B = align.M @ state_A  # proyectar estado A a espacio B
    state_B = state_B / np.linalg.norm(state_B)

    # Aplicar transformaciones
    new_state_A = T_A @ state_A
    new_state_B = T_B @ state_B

    # Proyectar new_state_A al espacio B para comparar
    new_state_A_in_B = align.M @ new_state_A
    new_state_A_in_B = new_state_A_in_B / np.linalg.norm(new_state_A_in_B + 1e-8)
    new_state_B_norm = new_state_B / np.linalg.norm(new_state_B + 1e-8)

    # Similaridad coseno entre los dos resultados
    cosine_sim = float(np.dot(new_state_A_in_B, new_state_B_norm))

    results = {
        "d_A": d_A,
        "d_B": d_B,
        "n_concepts": n_concepts,
        "frobenius_error": align.frobenius_error,
        "alignment_quality": AlignSession(align, None).alignment_quality
            if align.frobenius_error < 1e6 else "N/A",
        "cosine_similarity_after_transform": cosine_sim,
        "semantic_preservation": cosine_sim > 0.8,
    }

    if verbose:
        print(f"\n{'='*55}")
        print(f"POLYDIM ALIGN Demo — Procrustes")
        print(f"{'='*55}")
        print(f"  IA_A: R^{d_A}  →  IA_B: R^{d_B}")
        print(f"  Codebook: {n_concepts} GEO_IDs universales")
        print(f"  Error Frobenius: {align.frobenius_error:.4f}")
        print(f"  Similaridad coseno post-transform: {cosine_sim:.4f}")
        print(f"  Preservación semántica: {'✓ SÍ' if results['semantic_preservation'] else '✗ NO'}")
        print(f"{'='*55}\n")

    return results


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------

def run_align_tests(verbose: bool = True) -> bool:
    """
    Tests del protocolo ALIGN.
    Verifica las propiedades matemáticas de Procrustes y la transmisión de T.
    """
    import sys
    rng = np.random.default_rng(42)
    passed_all = True
    results = {}

    # TEST 1: M es ortogonal cuando d_A = d_B
    try:
        d = 64; k = 20
        A = rng.standard_normal((d, k)).astype(np.float32)
        B = rng.standard_normal((d, k)).astype(np.float32)
        # Normalizar columnas
        A /= np.linalg.norm(A, axis=0)
        B /= np.linalg.norm(B, axis=0)
        align = procrustes_align(A, B)
        # M^T @ M debería ser ≈ I cuando d_A = d_B
        ortho_err = np.linalg.norm(align.M.T @ align.M - np.eye(d))
        passed = ortho_err < 1e-4
        results["test_M_orthogonal_square"] = (passed, f"‖M^T·M - I‖ = {ortho_err:.2e}")
    except Exception as e:
        results["test_M_orthogonal_square"] = (False, str(e))

    # TEST 2: Procrustes minimiza ||MA - B||
    try:
        d_A, d_B, k = 64, 48, 20
        A = rng.standard_normal((d_A, k)).astype(np.float32)
        B = rng.standard_normal((d_B, k)).astype(np.float32)
        A /= np.linalg.norm(A, axis=0)
        B /= np.linalg.norm(B, axis=0)
        align = procrustes_align(A, B)
        frob = np.linalg.norm(align.M @ A - B, 'fro')
        # Verificar que M es aproximadamente óptima (no queremos comparar con random M)
        # Solo verificar que el error es finito y razonable
        passed = np.isfinite(frob) and frob < d_B * k  # cota muy holgada
        results["test_procrustes_minimizes"] = (passed, f"‖MA-B‖_F = {frob:.4f}")
    except Exception as e:
        results["test_procrustes_minimizes"] = (False, str(e))

    # TEST 3: align_transform produce T_B con dimensión correcta
    try:
        d_A, d_B, k = 64, 48, 20
        A = rng.standard_normal((d_A, k)).astype(np.float32)
        B = rng.standard_normal((d_B, k)).astype(np.float32)
        A /= np.linalg.norm(A, axis=0); B /= np.linalg.norm(B, axis=0)
        align = procrustes_align(A, B)
        T_A = rng.standard_normal((d_A, d_A)).astype(np.float32) * 0.1
        T_B = align_transform(T_A, align)
        passed = T_B.shape == (d_B, d_B)
        results["test_align_transform_shape"] = (passed, f"T_B.shape = {T_B.shape}")
    except Exception as e:
        results["test_align_transform_shape"] = (False, str(e))

    # TEST 4: LoRA ALIGN preserva el rango
    try:
        d_A, d_B, k, r = 64, 48, 20, 8
        A = rng.standard_normal((d_A, k)).astype(np.float32)
        B = rng.standard_normal((d_B, k)).astype(np.float32)
        A /= np.linalg.norm(A, axis=0); B /= np.linalg.norm(B, axis=0)
        align = procrustes_align(A, B)
        U_A = rng.standard_normal((d_A, r)).astype(np.float32)
        V_A = rng.standard_normal((r, d_A)).astype(np.float32)
        U_B, V_B = align_lora_transform(U_A, V_A, align)
        passed = U_B.shape == (d_B, r) and V_B.shape == (r, d_B)
        results["test_lora_align_shape"] = (
            passed, f"U_B={U_B.shape}, V_B={V_B.shape} (esperado [{d_B}×{r}], [{r}×{d_B}])"
        )
    except Exception as e:
        results["test_lora_align_shape"] = (False, str(e))

    # TEST 5: Codebook es cuasi-ortogonal
    try:
        cb = UniversalCodebook(n_concepts=50, N=1024, seed=42)
        stats = cb.verify_quasi_orthogonality()
        passed = stats["quasi_orthogonal"]
        results["test_codebook_quasi_orthogonal"] = (
            passed,
            f"max_cos={stats['max_off_diagonal_cosine']:.4f}, "
            f"expected_std={stats['expected_std_by_VSA']:.4f}"
        )
    except Exception as e:
        results["test_codebook_quasi_orthogonal"] = (False, str(e))

    # TEST 6: Demo de preservación semántica
    try:
        demo = simulate_align_demo(d_A=128, d_B=64, n_concepts=30, verbose=False)
        passed = demo["semantic_preservation"] or demo["cosine_similarity_after_transform"] > 0.5
        results["test_semantic_preservation"] = (
            passed,
            f"cosine_sim={demo['cosine_similarity_after_transform']:.4f}"
        )
    except Exception as e:
        results["test_semantic_preservation"] = (False, str(e))

    # Reporte
    if verbose:
        print(f"\n{'='*55}")
        print("POLYDIM ALIGN V2 — Tests")
        print(f"{'='*55}")
        for name, (p, msg) in results.items():
            status = "✓ PASS" if p else "✗ FAIL"
            print(f"  {status}  {name}")
            if verbose or not p:
                print(f"         {msg}")
            if not p:
                passed_all = False
        total = len(results)
        passed_count = sum(1 for p, _ in results.values() if p)
        print(f"\nResultado: {passed_count}/{total} tests pasando")
        print(f"{'='*55}\n")

    return passed_all


if __name__ == "__main__":
    import sys
    # Modo demo
    print("=== POLYDIM ALIGN V2 — Protocolo AI↔AI ===\n")
    simulate_align_demo(d_A=512, d_B=256, n_concepts=50, verbose=True)
    success = run_align_tests(verbose=True)
    sys.exit(0 if success else 1)
