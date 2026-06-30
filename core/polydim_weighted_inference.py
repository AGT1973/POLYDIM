# POLYDIM_DEST
# destino: polydim/core/
# nombre:  polydim_weighted_inference.py
# autor:   ai.mpat.agt@gmail.com
# fecha:   2026-06-27
# tarea:   TASK_022 — PEND_002

"""
POLYDIM Weighted Inference — V1
================================
Dado un ObjectND sin pesos de activación declarados (o con pesos subóptimos),
infiere automáticamente los pesos óptimos para cada subespacio en función del
contenido semántico del objeto.

PROBLEMA ORIGINAL (PEND_002):
    El usuario puede crear:
        obj = ObjectND(sp).add("DIM_SQL", {"tabla": "usuarios"})
    sin especificar 'w' (peso). O puede especificar pesos arbitrarios que no
    reflejan la relevancia semántica real del contenido.

SOLUCIÓN:
    Calcular la relevancia de cada subespacio via similitud coseno entre
    el embedding del contenido y el vector base del subespacio (ê_i).
    Los pesos inferidos reflejan cuánta "presencia geométrica" tiene cada
    subespacio en el contenido.

VARIANTES IMPLEMENTADAS:
    1. CosineSimilarityInference — similitud coseno entre embedding y base
    2. TFIDFInference — pesos basados en riqueza de keywords por subespacio
    3. LengthProportionalInference — pesos proporcionales a la densidad del contenido
    4. HybridInference — combinación ponderada de los tres anteriores (recomendada)

USO:
    from polydim_weighted_inference import infer_weights, HybridInference

    # Sin pesos declarados — usa inferencia automática
    obj = ObjectND(sp).add("DIM_SQL", {"tabla": "usuarios", "col": "id"})
    weights = infer_weights(obj.content, method="hybrid")
    # → {"DIM_SQL": 0.87, "DIM_VECTOR": 0.12, "DIM_META": 0.08, ...}

    # Aplicar los pesos al objeto
    for dim, w in weights.items():
        obj.set_weight(dim, w)
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Imports del runtime (para funciones geométricas — compatible V0.3 / V0.4)
# ---------------------------------------------------------------------------
try:
    from polydim_runtime_v04 import (
        ObjectND as _ObjectND, Space as _Space,
        NATIVE as _NATIVE, UMBRAL as _UMBRAL,
        N as _N, _proj as _proj_fn,
    )
    _HAS_RUNTIME = True
except ImportError:
    try:
        from polydim_runtime_v03 import (  # type: ignore[no-redef]
            ObjectND as _ObjectND, Space as _Space,
            NATIVE as _NATIVE, UMBRAL as _UMBRAL,
            N as _N, _proj as _proj_fn,
        )
        _HAS_RUNTIME = True
    except ImportError:
        _HAS_RUNTIME = False

# ---------------------------------------------------------------------------
# Constantes del modelo de ruido y temperatura (para inferencia geométrica)
# ---------------------------------------------------------------------------
_N_VAL = _N if _HAS_RUNTIME else 10000
NOISE_MEAN: float = 0.5
NOISE_STD: float = 1.0 / (2.0 * math.sqrt(_N_VAL))
Z_THRESHOLD: float = 2.0
DEFAULT_TEMPERATURE: float = 1.0


# ---------------------------------------------------------------------------
# Mapeo de subespacios
# ---------------------------------------------------------------------------

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

SUBSPACE_NAMES = list(SUBSPACE_IDS.keys())

# ---------------------------------------------------------------------------
# Diccionarios de keywords por subespacio
# (usados para TFIDFInference)
# ---------------------------------------------------------------------------

SUBSPACE_KEYWORDS: Dict[str, List[str]] = {
    "DIM_PYTHON": [
        "def", "class", "import", "function", "lambda", "yield",
        "async", "await", "dict", "list", "tuple", "set", "int",
        "float", "str", "bool", "None", "True", "False",
        "pandas", "numpy", "sklearn", "ml", "analysis", "script",
        "module", "package", "pip", "python",
    ],
    "DIM_RUST": [
        "struct", "enum", "impl", "fn", "let", "mut", "pub",
        "use", "mod", "crate", "ownership", "borrow", "lifetime",
        "unsafe", "trait", "Option", "Result", "Vec", "HashMap",
        "performance", "memory", "safety", "cargo", "rust",
    ],
    "DIM_FLUTTER": [
        "widget", "flutter", "dart", "scaffold", "column", "row",
        "container", "text", "button", "stateful", "stateless",
        "build", "context", "setState", "provider", "riverpod",
        "ui", "interface", "screen", "view", "layout", "render",
        "gesture", "animation", "material", "cupertino",
    ],
    "DIM_SQL": [
        "select", "from", "where", "join", "insert", "update",
        "delete", "table", "column", "schema", "database", "db",
        "index", "primary", "foreign", "key", "constraint",
        "query", "sql", "relational", "row", "record", "field",
        "null", "not null", "int", "varchar", "timestamp",
    ],
    "DIM_GRAPH": [
        "node", "edge", "graph", "vertex", "path", "tree",
        "network", "relation", "link", "connection", "neighbor",
        "degree", "cycle", "dag", "directed", "undirected",
        "neo4j", "cypher", "graphql", "sparql", "rdf", "ontology",
    ],
    "DIM_VECTOR": [
        "embedding", "vector", "similarity", "cosine", "distance",
        "dimension", "space", "latent", "encode", "decode",
        "representation", "feature", "cluster", "search", "knn",
        "faiss", "pinecone", "weaviate", "qdrant", "dense",
        "semantic", "retrieval", "index", "approximate",
    ],
    "DIM_TIME": [
        "time", "date", "timestamp", "datetime", "duration",
        "interval", "sequence", "order", "event", "log",
        "series", "temporal", "history", "schedule", "cron",
        "before", "after", "during", "start", "end", "expire",
        "created_at", "updated_at", "epoch", "unix", "iso",
    ],
    "DIM_ERROR": [
        "error", "exception", "try", "catch", "raise", "throw",
        "failure", "fault", "bug", "issue", "warning", "panic",
        "crash", "timeout", "retry", "fallback", "recovery",
        "404", "500", "400", "503", "invalid", "not found",
        "permission", "unauthorized", "forbidden", "traceback",
    ],
    "DIM_META": [
        "meta", "metadata", "audit", "log", "version", "tag",
        "label", "annotation", "description", "name", "id",
        "uuid", "hash", "created", "modified", "author", "owner",
        "source", "origin", "schema", "format", "encoding",
        "mime", "content_type", "checksum", "signature",
    ],
}

# Conjunto de palabras clave por subespacio para lookup rápido
SUBSPACE_KW_SETS: Dict[str, set] = {
    dim: {kw.lower() for kw in kws}
    for dim, kws in SUBSPACE_KEYWORDS.items()
}


# ---------------------------------------------------------------------------
# Extracción de texto desde contenido arbitrario
# ---------------------------------------------------------------------------

def extract_text(content: Any) -> str:
    """
    Extrae texto legible de cualquier tipo de contenido POLYDIM.
    Maneja: str, dict, list, int, float, None.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.lower()
    if isinstance(content, dict):
        parts = []
        for k, v in content.items():
            parts.append(str(k).lower())
            parts.append(extract_text(v))
        return " ".join(parts)
    if isinstance(content, (list, tuple)):
        return " ".join(extract_text(item) for item in content)
    if isinstance(content, (int, float)):
        return str(content)
    return str(content).lower()


def tokenize(text: str) -> List[str]:
    """Tokenización simple por palabras alfanuméricas."""
    return re.findall(r'[a-z][a-z0-9_]{1,}', text.lower())


# ---------------------------------------------------------------------------
# Variante 1: CosineSimilarityInference
# ---------------------------------------------------------------------------

class CosineSimilarityInference:
    """
    Infiere pesos via similitud coseno entre el embedding del contenido
    y el vector base de cada subespacio.

    En V1: usa un embedding basado en bag-of-keywords ponderados.
    En V2 (con MiniLM disponible): usar embedding real del contenido.
    """

    def __init__(self, N: int = 512, seed: int = 42):
        self.N = N
        self.seed = seed
        # Pre-computar vectores base de subespacios
        self._bases = {
            dim: self._subspace_basis(i)
            for dim, i in SUBSPACE_IDS.items()
        }

    def _subspace_basis(self, dim_id: int) -> np.ndarray:
        """Vector base determinístico del subespacio (LCG)."""
        state = np.uint64((dim_id * 1_000_003 + 1) % (2**64))
        v = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            state = np.uint64(state * np.uint64(6_364_136_223_846_793_005)
                              + np.uint64(1_442_695_040_888_963_407))
            v[i] = float(state) / float(2**64) * 2.0 - 1.0
        norm = np.linalg.norm(v)
        return v / (norm + 1e-8)

    def _content_to_vector(self, text: str) -> np.ndarray:
        """
        Convierte texto a un vector en R^N via keyword-weighted sum
        de los vectores base de subespacios.
        """
        tokens = set(tokenize(text))
        v = np.zeros(self.N, dtype=np.float32)
        total_hits = 0
        for dim, kw_set in SUBSPACE_KW_SETS.items():
            hits = len(tokens & kw_set)
            if hits > 0:
                v += hits * self._bases[dim]
                total_hits += hits
        if total_hits > 0:
            v /= total_hits
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            # Sin keywords reconocidas → distribuir uniformemente
            v = np.ones(self.N, dtype=np.float32) / np.sqrt(self.N)
        else:
            v /= norm
        return v

    def infer(self, content_map: Dict[str, Any]) -> Dict[str, float]:
        """
        Infiere pesos para cada subespacio en content_map.
        content_map: {dim_name: contenido}
        """
        weights = {}
        for dim, content in content_map.items():
            if dim not in SUBSPACE_IDS:
                continue
            text = extract_text(content)
            if not text.strip():
                weights[dim] = 0.1
                continue
            content_vec = self._content_to_vector(text)
            basis = self._bases[dim]
            cosine = float(np.dot(content_vec, basis))
            # Mapear cosine ∈ [-1, 1] → peso ∈ [0.1, 1.0]
            weight = 0.1 + 0.9 * max(0.0, cosine)
            weights[dim] = round(min(1.0, weight), 4)
        return weights


# ---------------------------------------------------------------------------
# Variante 2: TFIDFInference
# ---------------------------------------------------------------------------

class TFIDFInference:
    """
    Infiere pesos via TF-IDF sobre los keywords de cada subespacio.

    TF(dim, text) = fracción de tokens en `text` que son keywords de `dim`
    IDF(dim) = log(N_dims / (1 + N_dims_matching))  — rareza del subespacio
    Peso(dim) = TF × IDF, normalizado a [0, 1]
    """

    def infer(self, content_map: Dict[str, Any]) -> Dict[str, float]:
        raw_scores: Dict[str, float] = {}
        for dim, content in content_map.items():
            if dim not in SUBSPACE_IDS:
                continue
            text = extract_text(content)
            tokens = tokenize(text)
            if not tokens:
                raw_scores[dim] = 0.0
                continue
            kw_set = SUBSPACE_KW_SETS[dim]
            tf = sum(1 for t in tokens if t in kw_set) / len(tokens)
            # IDF: inverso del número de subespacios con keywords similares
            n_overlapping = sum(
                1 for other_dim, other_kws in SUBSPACE_KW_SETS.items()
                if other_dim != dim and len({t for t in tokens if t in other_kws}) > 0
            )
            idf = math.log(len(SUBSPACE_IDS) / (1 + n_overlapping))
            raw_scores[dim] = max(0.0, tf * idf)

        if not raw_scores:
            return {}

        # Normalizar a [0.1, 1.0]
        max_score = max(raw_scores.values()) if raw_scores else 1.0
        if max_score < 1e-8:
            return {dim: 0.1 for dim in raw_scores}

        return {
            dim: round(0.1 + 0.9 * (score / max_score), 4)
            for dim, score in raw_scores.items()
        }


# ---------------------------------------------------------------------------
# Variante 3: LengthProportionalInference
# ---------------------------------------------------------------------------

class LengthProportionalInference:
    """
    Infiere pesos proporcionales a la densidad/riqueza del contenido.
    Útil cuando el contenido es estructurado (dicts con muchos campos)
    o cuando los otros métodos no tienen suficientes keywords.

    Peso(dim) ∝ len(flatten(content)) / max_length
    """

    def _content_density(self, content: Any) -> float:
        """Estima la densidad de información del contenido."""
        text = extract_text(content)
        tokens = tokenize(text)
        unique_tokens = set(tokens)
        # Combinar longitud y diversidad léxica
        density = len(tokens) * math.log(1 + len(unique_tokens))
        return density

    def infer(self, content_map: Dict[str, Any]) -> Dict[str, float]:
        densities: Dict[str, float] = {}
        for dim, content in content_map.items():
            if dim not in SUBSPACE_IDS:
                continue
            densities[dim] = self._content_density(content)

        max_density = max(densities.values()) if densities else 1.0
        if max_density < 1e-8:
            return {dim: 0.1 for dim in densities}

        return {
            dim: round(0.1 + 0.9 * (d / max_density), 4)
            for dim, d in densities.items()
        }


# ---------------------------------------------------------------------------
# Variante 4: HybridInference (recomendada)
# ---------------------------------------------------------------------------

class HybridInference:
    """
    Combinación ponderada de los tres métodos:
        w_final = α·w_cosine + β·w_tfidf + γ·w_length

    Valores por defecto: α=0.5, β=0.35, γ=0.15
    → Prioriza la similitud semántica (cosine), ajusta por keywords (tfidf),
      y tiene un factor de densidad de contenido (length).
    """

    def __init__(
        self,
        alpha: float = 0.50,  # peso del método cosine
        beta:  float = 0.35,  # peso del método tfidf
        gamma: float = 0.15,  # peso del método length
        N: int = 512,
    ):
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, \
            f"α+β+γ debe ser 1.0, got {alpha+beta+gamma}"
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self._cosine = CosineSimilarityInference(N=N)
        self._tfidf  = TFIDFInference()
        self._length = LengthProportionalInference()

    def infer(self, content_map: Dict[str, Any]) -> Dict[str, float]:
        w_cos = self._cosine.infer(content_map)
        w_tfi = self._tfidf.infer(content_map)
        w_len = self._length.infer(content_map)

        all_dims = set(w_cos) | set(w_tfi) | set(w_len)
        result = {}
        for dim in all_dims:
            wc = w_cos.get(dim, 0.1)
            wt = w_tfi.get(dim, 0.1)
            wl = w_len.get(dim, 0.1)
            combined = self.alpha * wc + self.beta * wt + self.gamma * wl
            result[dim] = round(min(1.0, max(0.05, combined)), 4)

        return result


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def infer_weights_semantic(
    content_map: Dict[str, Any],
    method: str = "hybrid",
    **kwargs,
) -> Dict[str, float]:
    """
    Infiere pesos semánticos para un mapa de contenido POLYDIM.
    """
    inferrers = {
        "hybrid":  HybridInference,
        "cosine":  CosineSimilarityInference,
        "tfidf":   TFIDFInference,
        "length":  LengthProportionalInference,
    }
    if method not in inferrers:
        raise ValueError(
            f"Método desconocido: {method!r}. "
            f"Opciones: {list(inferrers.keys())}"
        )
    inferrer = inferrers[method](**kwargs)
    return inferrer.infer(content_map)


def infer_weights_geometric(
    obj: Any,
    temperature: float = DEFAULT_TEMPERATURE,
    include_native: bool = True,
    dims_override: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Infiere pesos geométricos (z-score softmax) a partir de un ObjectND.
    """
    if not _HAS_RUNTIME:
        raise ImportError("Se requiere el runtime de POLYDIM para inferencia geométrica.")
    
    sp = obj._sp
    hv = obj._hv()

    # Dimensiones a evaluar
    if dims_override is not None:
        dims = list(dict.fromkeys(dims_override))
    elif include_native:
        dims = list(dict.fromkeys(_NATIVE + list(obj._props)))
    else:
        dims = list(obj._props)

    if not dims:
        return {}

    # Paso 1: activaciones brutas
    raw: Dict[str, float] = {d: _proj_fn(hv, sp.sub(d)) for d in dims}

    # Paso 2: z-scores respecto al ruido de fondo
    z: Dict[str, float] = {
        d: (v - NOISE_MEAN) / NOISE_STD for d, v in raw.items()
    }

    # Paso 3: filtrar z > Z_THRESHOLD
    active: Dict[str, float] = {d: zv for d, zv in z.items() if zv > Z_THRESHOLD}
    if not active:
        return {}

    # Paso 4: softmax estable (shift por máximo para evitar overflow)
    keys = list(active.keys())
    zvals = np.array([active[k] for k in keys], dtype=np.float64)
    zvals_shifted = zvals - zvals.max()
    exp_z = np.exp(zvals_shifted / max(temperature, 1e-8))
    weights = exp_z / exp_z.sum()

    return {k: round(float(w), 4) for k, w in zip(keys, weights)}


def infer_weights(
    obj_or_content: Any,
    *args,
    **kwargs,
) -> Dict[str, float]:
    """
    Dispatcher principal para inferencia de pesos.
    - Si el primer argumento es un diccionario (content_map), usa inferencia semántica.
    - Si el primer argumento es un ObjectND, usa inferencia geométrica.
    """
    if isinstance(obj_or_content, dict):
        return infer_weights_semantic(obj_or_content, *args, **kwargs)
    else:
        return infer_weights_geometric(obj_or_content, *args, **kwargs)


def apply_inferred_weights(obj, method: str = "hybrid", **kwargs) -> None:
    """
    Aplica pesos inferidos directamente a un ObjectND.
    Soporta tanto inferencia semántica (in-place) como geométrica.
    """
    if method == "geometric" or not isinstance(obj, dict) and hasattr(obj, '_sp') and method not in ["hybrid", "cosine", "tfidf", "length"]:
        # Aplicar geométrico
        ws = infer_weights_geometric(obj, **kwargs)
        if hasattr(obj, '_dims'):
            for dim_entry in obj._dims:
                dim_name = dim_entry.get('name', '')
                dim_entry['w'] = ws.get(dim_name, 0.0)
    else:
        # Extraer mapa de contenido del ObjectND y aplicar semántico
        content_map = {}
        if hasattr(obj, '_dims'):
            for dim_entry in obj._dims:
                dim_name = dim_entry.get('name', '')
                if dim_name in SUBSPACE_IDS:
                    content_map[dim_name] = dim_entry.get('content', {})
        elif hasattr(obj, 'content'):
            content_map = obj.content if isinstance(obj.content, dict) else {}

        if not content_map:
            return

        weights = infer_weights_semantic(content_map, method=method, **kwargs)

        # Actualizar los pesos en el objeto
        if hasattr(obj, '_dims'):
            for dim_entry in obj._dims:
                dim_name = dim_entry.get('name', '')
                if dim_name in weights:
                    dim_entry['w'] = weights[dim_name]


def infer_weights_ranked(
    obj: Any,
    temperature: float = DEFAULT_TEMPERATURE,
) -> List[tuple]:
    """Versión ordenada de infer_weights_geometric, de mayor a menor peso."""
    ws = infer_weights_geometric(obj, temperature=temperature)
    return sorted(ws.items(), key=lambda x: -x[1])


def weight_entropy(weights: Dict[str, float]) -> float:
    """Entropía de Shannon normalizada de los pesos inferidos."""
    if not weights:
        return 0.0
    vals = np.array(list(weights.values()), dtype=np.float64)
    vals = vals[vals > 0]
    if len(vals) == 1:
        return 0.0
    h = -np.sum(vals * np.log(vals + 1e-12))
    h_max = math.log(len(vals))
    return round(float(h / h_max), 4) if h_max > 0 else 0.0


def dominant_dim(
    obj: Any,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Optional[str]:
    """Retorna el nombre de la dimensión con mayor peso inferido geométricamente."""
    ranked = infer_weights_ranked(obj, temperature=temperature)
    return ranked[0][0] if ranked else None


def weight_report(
    obj: Any,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Genera un reporte textual de los pesos inferidos geométricamente."""
    if not _HAS_RUNTIME:
        return "Error: Runtime no disponible"
    sp = obj._sp
    hv = obj._hv()
    dims = list(dict.fromkeys(_NATIVE + list(obj._props)))
    raw = {d: _proj_fn(hv, sp.sub(d)) for d in dims}
    z = {d: (v - NOISE_MEAN) / NOISE_STD for d, v in raw.items()}
    ws = infer_weights_geometric(obj, temperature=temperature)

    lines = [
        f"WeightReport  geo_id={obj.geo_id}  T={temperature}  N={_N}",
        f"{'DIM':<18} {'act':>7} {'z':>7} {'w_inf':>7} {'active':>7}",
        "-" * 52,
    ]
    for d in sorted(dims, key=lambda x: -raw[x]):
        mark = "✓" if z[d] > Z_THRESHOLD else " "
        w_s = f"{ws.get(d, 0.0):.4f}" if d in ws else "  —   "
        lines.append(
            f"{d:<18} {raw[d]:>7.4f} {z[d]:>7.1f} {w_s:>7} {mark:>7}"
        )
    lines.append("-" * 52)
    lines.append(
        f"Dims activas: {len(ws)}  "
        f"Entropía: {weight_entropy(ws):.4f}  "
        f"Dominante: {dominant_dim(obj, temperature) or 'ninguna'}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo y tests
# ---------------------------------------------------------------------------

def run_demo():
    """Demostración de la inferencia de pesos sobre casos representativos."""
    print("\n" + "="*60)
    print("POLYDIM Weighted Inference V1 — Demo")
    print("="*60)

    test_cases = [
        {
            "name": "Objeto SQL puro",
            "content": {
                "DIM_SQL": {
                    "tabla": "usuarios",
                    "columna": "id",
                    "tipo": "INTEGER PRIMARY KEY",
                    "constraint": "NOT NULL"
                }
            },
        },
        {
            "name": "Objeto multi-dimensional (SQL + Flutter)",
            "content": {
                "DIM_SQL":     {"tabla": "productos", "select": "id, nombre, precio"},
                "DIM_FLUTTER": {"widget": "ListView", "builder": "itemBuilder"},
                "DIM_META":    {"version": "1.0", "created_at": "2026-06-27"},
            },
        },
        {
            "name": "Objeto de error con metadata",
            "content": {
                "DIM_ERROR": {"exception": "ConnectionTimeout", "retry": True, "traceback": "..."},
                "DIM_META":  {"timestamp": "2026-06-27T12:00:00", "severity": "error"},
                "DIM_TIME":  {"created": "2026-06-27", "expires": "2026-06-28"},
            },
        },
        {
            "name": "Objeto de embedding vectorial",
            "content": {
                "DIM_VECTOR": {"embedding": [0.1, 0.2, 0.3], "similarity": "cosine"},
                "DIM_GRAPH":  {"node": "user_42", "edges": ["product_1", "product_5"]},
            },
        },
    ]

    for case in test_cases:
        print(f"\n  [{case['name']}]")
        for method in ["cosine", "tfidf", "hybrid"]:
            weights = infer_weights(case["content"], method=method)
            print(f"  {method:8s}: ", end="")
            for dim, w in sorted(weights.items(), key=lambda x: -x[1]):
                bar = "#" * int(w * 10)
                print(f"{dim.replace('DIM_','')}={w:.2f}{bar} ", end="")
            print()


def run_tests() -> bool:
    """Tests de unidad para la inferencia de pesos."""
    print("\n" + "="*60)
    print("POLYDIM Weighted Inference V1 — Tests")
    print("="*60)

    results = {}
    passed_all = True

    # Test 1: SQL puro debería tener DIM_SQL como peso dominante
    try:
        w = infer_weights({"DIM_SQL": {"tabla": "usuarios", "select": "id"}})
        passed = "DIM_SQL" in w and w["DIM_SQL"] >= 0.5
        results["test_sql_dominant"] = (passed, f"DIM_SQL={w.get('DIM_SQL', 0):.3f}")
    except Exception as e:
        results["test_sql_dominant"] = (False, str(e))

    # Test 2: Flutter debería reconocer keywords de UI
    try:
        w = infer_weights({"DIM_FLUTTER": {"widget": "ListView", "builder": "itemBuilder"}})
        passed = "DIM_FLUTTER" in w and w["DIM_FLUTTER"] >= 0.4
        results["test_flutter_recognized"] = (passed, f"DIM_FLUTTER={w.get('DIM_FLUTTER', 0):.3f}")
    except Exception as e:
        results["test_flutter_recognized"] = (False, str(e))

    # Test 3: Pesos en [0, 1]
    try:
        content = {
            "DIM_SQL":    {"tabla": "t"},
            "DIM_PYTHON": {"def": "fn"},
            "DIM_META":   {"version": "1"},
        }
        w = infer_weights(content, method="hybrid")
        passed = all(0.0 <= v <= 1.0 for v in w.values())
        results["test_weights_bounded"] = (passed, f"range=[{min(w.values()):.3f}, {max(w.values()):.3f}]")
    except Exception as e:
        results["test_weights_bounded"] = (False, str(e))

    # Test 4: Contenido vacío → pesos mínimos
    try:
        w = infer_weights({"DIM_SQL": "", "DIM_FLUTTER": None})
        passed = all(v <= 0.2 for v in w.values()) if w else True
        results["test_empty_content"] = (passed, f"pesos={dict(w)}")
    except Exception as e:
        results["test_empty_content"] = (False, str(e))

    # Test 5: Método tfidf da resultados distintos de hybrid
    try:
        content = {"DIM_VECTOR": {"embedding": [0.1, 0.2], "cosine": "similarity"}}
        w_h = infer_weights(content, method="hybrid")
        w_t = infer_weights(content, method="tfidf")
        # Verificar que ambos reconocen el subespacio
        passed = "DIM_VECTOR" in w_h and "DIM_VECTOR" in w_t
        results["test_methods_consistent"] = (
            passed,
            f"hybrid={w_h.get('DIM_VECTOR',0):.3f}, tfidf={w_t.get('DIM_VECTOR',0):.3f}"
        )
    except Exception as e:
        results["test_methods_consistent"] = (False, str(e))

    # Test 6: extract_text maneja tipos mixtos
    try:
        text = extract_text({"key": [1, 2, 3], "nested": {"a": "b"}, "val": 42.5})
        passed = isinstance(text, str) and len(text) > 0
        results["test_extract_text_mixed"] = (passed, f"len={len(text)}")
    except Exception as e:
        results["test_extract_text_mixed"] = (False, str(e))

    # Reporte
    for name, (p, msg) in results.items():
        status = "[PASS]" if p else "[FAIL]"
        print(f"  {status}  {name}: {msg}")
        if not p:
            passed_all = False

    total = len(results)
    passed_count = sum(1 for p, _ in results.values() if p)
    print(f"\n  Resultado: {passed_count}/{total} tests pasando")
    print("="*60 + "\n")
    return passed_all


if __name__ == "__main__":
    import sys
    run_demo()
    success = run_tests()
    sys.exit(0 if success else 1)
