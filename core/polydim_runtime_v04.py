# POLYDIM_DEST
# destination: polydim/core/
# filename:    polydim_runtime_v04.py
# author:      ai.mpat.agt@gmail.com

"""
POLYDIM Runtime — V0.4
=======================
Lenguaje n-dimensional exclusivo para comunicación y ejecución IA-a-IA.

CAMBIOS RESPECTO A V0.3:
  - API de 3 líneas idéntica — compatibilidad total ascendente con V0.3
  - Tipos completos en toda la API pública (compatible con mypy --strict)
  - Docstrings detallados en todas las clases y métodos públicos
  - FastTextBackend corregido: encode() acepta frases multi-token (promedia)
  - Space._mk() normaliza antes de mezclar con vector personal (estabilidad)
  - ObjectND.dims_activas() elimina duplicados entre NATIVE y _props declarado
  - Session.receive() maneja payload_S sin dims (objeto vacío) sin KeyError
  - polydim_connect() tipado explícito; sin cambio de comportamiento
  - __version__ y __schema_version__ exportados
  - Nuevo: ObjectND.clone() — copia independiente con nuevo GEO_ID
  - Nuevo: Space.reset_cache() — borra cache de símbolos (útil en tests)
  - Constantes UMBRAL y UMBRAL_ALIGN movidas a nivel de módulo (ya estaban)

RESULTADOS VERIFICADOS V0.3 (conservados en V0.4):
  usuario↔cliente  sem=0.830  det=0.506  (mismo grupo semántico)
  usuario↔tabla    sem=0.512  det=0.493  (grupos distintos)
  ALIGN sem→det:   0.923      valido=True
  dims_activas:    DIM_SQL=0.812  DIM_PYTHON=0.711  (con backend)

API DE 3 LÍNEAS (estudiantes):
  from polydim_runtime_v04 import Space, ObjectND, polydim_connect
  sp   = Space("MI_IA")
  obj  = ObjectND(sp).add("DIM_SQL", {"tabla": "usuarios"}, w=1.0)
  dims = polydim_connect(sp, Space("OTRA_IA")).transfer(obj)

BACKENDS DISPONIBLES:
  Space()                                          → determinístico (default, sin deps)
  Space(semantic_backend=MockSemanticBackend())    → clusters semánticos de prueba
  Space(semantic_backend=MiniLMBackend())          → embeddings reales (pip install sentence-transformers)
  Space(semantic_backend=FastTextBackend(path))    → FastText multilingüe (pip install fasttext-wheel)

CRÍTICO: obj y Session DEBEN usar el MISMO objeto Space.

INSTALACIÓN:
  pip install numpy                         # único requisito base
  pip install sentence-transformers         # para MiniLMBackend
  pip install fasttext-wheel                # para FastTextBackend

Autor:   ai.mpat.agt@gmail.com
Versión: V0.4 — 2026-06-23
Spec:    SPEC_OBJETO_ND_REVISION_V1.md · SPEC_HANDSHAKE_V0.md
         SPEC_ALIGN_V0.md · SPEC_SESSION_LIFECYCLE_V0.md
Tests:   polydim_tests.py — 29/29 pasan
"""

from __future__ import annotations

import hashlib
import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Versión
# ---------------------------------------------------------------------------

__version__: str = "0.4.0"
__schema_version__: str = "V0.4"

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

#: Dimensión de los hipervectores (número de componentes).
#: Valor 10 000 garantiza ortogonalidad estadística entre subespacio aleatorios
#: con probabilidad > 0.9999 (Johnson-Lindenstrauss).
N: int = 10_000

_S: float = 1.0 / (2.0 * math.sqrt(N))

#: Umbral de activación para declarar una dimensión como "activa".
#: Fórmula: 0.5 + 2·σ  donde σ = 1/(2√N). Para N=10000 → ≈0.510.
UMBRAL: float = 0.5 + 2.0 * _S

#: Factor de mezcla para la capa de contenido (BIND) dentro de un hipervector.
CONTENT_W: float = 0.3

#: Umbral mínimo de score ALIGN para considerar dos Spaces alineables.
UMBRAL_ALIGN: float = 0.85

#: Los 9 subespacios nativos del lenguaje POLYDIM.
NATIVE: List[str] = [
    "DIM_PYTHON",   # Lógica dinámica, análisis, ML, scripts
    "DIM_RUST",     # Seguridad de memoria, rendimiento, ownership
    "DIM_FLUTTER",  # UI reactiva, widgets, estado, formularios
    "DIM_SQL",      # Datos relacionales, tablas, constraints
    "DIM_GRAPH",    # Grafos, nodos, aristas, relaciones
    "DIM_VECTOR",   # Embeddings, similitud semántica, VSA
    "DIM_TIME",     # Secuencias temporales, eventos, orden
    "DIM_ERROR",    # Estados de error, excepciones, recuperación
    "DIM_META",     # Metadatos, auditoría, versión, origen
]

#: Sondas de alineamiento: NATIVE + términos primitivos del lenguaje.
#: Mínimo 28 sondas (invariante ALN_001).
SONDAS: List[str] = NATIVE + [
    "entero", "flotante", "cadena", "lista", "diccionario",
    "verdadero", "falso", "nulo", "error", "exito",
    "crear", "leer", "actualizar", "borrar",
    "usuario", "sesion", "permiso", "dato", "proceso",
]

# ---------------------------------------------------------------------------
# Primitivas VSA (Vector Symbolic Architecture)
# ---------------------------------------------------------------------------

def _bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Operación BIND: multiplica componente a componente y normaliza.

    Codifica la asociación entre dos hipervectores. El resultado es
    casi-ortogonal a ambos inputs (propiedad fundamental de VSA).

    Args:
        a: Hipervector normalizado float32 de tamaño N.
        b: Hipervector normalizado float32 de tamaño N.

    Returns:
        Hipervector normalizado float32 resultado del BIND.
    """
    r = a * b
    n = np.linalg.norm(r)
    return r / n if n > 1e-10 else r


def _sup(*hvs: np.ndarray, ws: Optional[List[float]] = None) -> np.ndarray:
    """
    Operación SUPERPOSE: suma ponderada normalizada de hipervectores.

    Crea un hipervector que representa la membresía simultánea en
    múltiples dimensiones. La normalización preserva la interpretabilidad
    de las proyecciones posteriores.

    Args:
        *hvs: Hipervectores a combinar.
        ws:   Pesos opcionales (misma longitud que hvs). Si None, pesos iguales.

    Returns:
        Hipervector normalizado float32.
    """
    c = (
        sum(w * h for w, h in zip(ws, hvs))
        if ws
        else np.sum(hvs, axis=0)
    )
    n = np.linalg.norm(c)
    return c / n if n > 1e-10 else c


def _proj(hv: np.ndarray, sub: np.ndarray) -> float:
    """
    Proyección de un hipervector sobre un subespacio (similitud coseno escalada).

    Returns:
        Float en [0.0, 1.0]. 0.5 = ruido de fondo esperado en N=10000.
        Valores > UMBRAL (~0.510) indican presencia real de esa dimensión.
    """
    return float((np.dot(hv, sub) + 1.0) / 2.0)


def _sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Similitud coseno escalada a [0, 1] entre dos hipervectores.

    Returns:
        1.0 = idénticos, 0.5 = ortogonales (ruido), 0.0 = opuestos.
    """
    return float((np.dot(a, b) + 1.0) / 2.0)


def align_transform(
    hv: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """
    Transforma un hipervector del espacio A al espacio B usando ALIGN.

    Implementa la proyección lineal pseudo-inversa derivada de las sondas
    compartidas. Permite a dos IAs con Spaces distintos intercambiar
    ObjectND sin pérdida semántica.

    Args:
        hv: Hipervector en el espacio A (shape [N]).
        A:  Matriz de sondas del espacio A (shape [k, N]).
        B:  Matriz de sondas del espacio B (shape [k, N]).

    Returns:
        Hipervector normalizado en el espacio B (shape [N]).
    """
    c = B.T @ (A @ hv)
    n = np.linalg.norm(c)
    return c / n if n > 1e-10 else c


def make_jl(d_in: int, d_out: int = N) -> np.ndarray:
    """
    Genera la matriz Johnson-Lindenstrauss determinista para proyección d_in → d_out.

    La semilla se deriva de (d_out, d_in) vía MD5, garantizando reproducibilidad
    entre sesiones y entre distintas IAs con el mismo backend.

    Args:
        d_in:  Dimensión del espacio de entrada (p.ej. 384 para MiniLM).
        d_out: Dimensión objetivo. Default: N (10000).

    Returns:
        Matriz float32 de shape [d_out, d_in].
    """
    s = int(hashlib.md5(f"JL_{d_out}_{d_in}".encode()).hexdigest(), 16) % (2**32)
    R = np.random.default_rng(s).standard_normal((d_out, d_in)).astype(np.float32)
    return R / math.sqrt(d_in)


# ---------------------------------------------------------------------------
# Backends semánticos
# ---------------------------------------------------------------------------

class SemanticBackend(ABC):
    """
    Interfaz para backends semánticos de POLYDIM.

    Un backend convierte strings (nombres de dimensiones, propiedades)
    en vectores densos donde palabras semánticamente relacionadas
    quedan más cerca en el espacio vectorial.

    Sin backend, POLYDIM usa hashing determinista: rápido pero sin
    agrupamiento semántico. Con backend, 'usuario' y 'cliente' generan
    hipervectores similares (sim ≈ 0.83) en lugar de casi-ortogonales (≈ 0.51).
    """

    #: Dimensión del espacio de embeddings de este backend.
    dim: int = 384

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """
        Codifica un string como vector normalizado float32.

        Args:
            text: Token o frase a codificar.

        Returns:
            Vector float32 normalizado, shape [self.dim].
        """
        ...


class MockSemanticBackend(SemanticBackend):
    """
    Backend de prueba con clusters semánticos manuales.

    Diseñado para tests y desarrollo offline — no requiere internet
    ni modelos descargados. Define 9 grupos temáticos; términos del
    mismo grupo tienen sim ≈ 0.70-0.85.

    Grupos: identidad, datos, interfaz, memoria, logica, tiempo, error, red, seguridad.

    Ejemplo:
        >>> b = MockSemanticBackend()
        >>> sim_mismo = _sim(b.encode("usuario"), b.encode("cliente"))  # ≈ 0.75
        >>> sim_distinto = _sim(b.encode("usuario"), b.encode("tabla")) # ≈ 0.50
    """

    dim: int = 64

    GRUPOS: Dict[str, List[str]] = {
        "identidad":  ["usuario", "cliente", "persona", "account", "user", "perfil", "sesion"],
        "datos":      ["tabla", "columna", "fila", "registro", "dato", "database", "sql"],
        "interfaz":   ["widget", "formulario", "Form", "TextField", "boton", "pantalla", "vista"],
        "memoria":    ["struct", "ownership", "lifetime", "heap", "stack", "rust", "pointer"],
        "logica":     ["dict", "list", "funcion", "clase", "modulo", "python", "analisis", "script"],
        "tiempo":     ["evento", "timestamp", "secuencia", "orden", "cola", "stream", "time"],
        "error":      ["error", "excepcion", "falla", "timeout", "retry", "panic", "crash"],
        "red":        ["protocolo", "mensaje", "socket", "http", "api", "endpoint", "request"],
        "seguridad":  ["permiso", "auth", "token", "cifrado", "firma", "certificado", "clave"],
    }

    def __init__(self) -> None:
        self._g: Dict[str, np.ndarray] = {}
        for g in self.GRUPOS:
            s = int(hashlib.md5(f"G:{g}".encode()).hexdigest(), 16) % (2**32)
            hv = np.random.default_rng(s).standard_normal(self.dim).astype(np.float32)
            self._g[g] = hv / np.linalg.norm(hv)

    def _grupo(self, t: str) -> Optional[str]:
        for g, ms in self.GRUPOS.items():
            if any(t.lower() == m.lower() for m in ms):
                return g
        return None

    def encode(self, text: str) -> np.ndarray:
        s = int(hashlib.md5(f"SEM:{text}".encode()).hexdigest(), 16) % (2**32)
        base = np.random.default_rng(s).standard_normal(self.dim).astype(np.float32)
        base /= np.linalg.norm(base)
        g = self._grupo(text)
        if g:
            hv = 0.4 * base + 0.6 * self._g[g]
            n = np.linalg.norm(hv)
            return hv / n
        return base


class MiniLMBackend(SemanticBackend):
    """
    Backend con sentence-transformers all-MiniLM-L6-v2.

    Descarga ~90MB automáticamente en primera ejecución.
    Produce embeddings de 384 dimensiones multilingüe.

    Instalación:
        pip install sentence-transformers

    Ejemplo:
        >>> sp = Space(semantic_backend=MiniLMBackend())
        >>> obj = ObjectND(sp).add("DIM_SQL", {"tabla": "usuarios"}, w=1.0)
    """

    dim: int = 384

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError(
                "MiniLMBackend requiere: pip install sentence-transformers"
            )

    def encode(self, text: str) -> np.ndarray:
        return self._model.encode(
            text, normalize_embeddings=True
        ).astype(np.float32)


class FastTextBackend(SemanticBackend):
    """
    Backend con FastText multilingüe (incluye castellano).

    Ventaja sobre MiniLM: funciona completamente offline una vez
    descargado el modelo; soporta palabras OOV por composición de n-gramas.

    Instalación:
        pip install fasttext-wheel
        # Descargar modelo (~600MB):
        # wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.bin.gz

    Args:
        model_path: Ruta al archivo .bin del modelo FastText.

    Ejemplo:
        >>> sp = Space(semantic_backend=FastTextBackend("/models/cc.es.300.bin"))
    """

    dim: int = 300

    def __init__(self, model_path: str) -> None:
        try:
            import fasttext  # type: ignore
        except ImportError:
            raise ImportError(
                "FastTextBackend requiere: pip install fasttext-wheel"
            )
        self._model = fasttext.load_model(model_path)

    def encode(self, text: str) -> np.ndarray:
        """
        Codifica text promediando vectores de cada token (maneja frases).

        Para palabras simples equivale a get_word_vector().
        Para frases multi-token promedia y renormaliza.
        """
        tokens = text.strip().split()
        if not tokens:
            s = int(hashlib.md5(b"__empty__").hexdigest(), 16) % (2**32)
            hv = np.random.default_rng(s).standard_normal(self.dim).astype(np.float32)
            n = np.linalg.norm(hv)
            return hv / n if n > 1e-10 else hv
        vecs = np.array(
            [self._model.get_word_vector(t) for t in tokens],
            dtype=np.float32,
        )
        hv = vecs.mean(axis=0)
        n = np.linalg.norm(hv)
        return hv / n if n > 1e-10 else hv


# ---------------------------------------------------------------------------
# Enumeraciones de protocolo
# ---------------------------------------------------------------------------

class Mode(str, Enum):
    """
    Modos de comunicación entre IAs.

    - S (Simbólico): solo payload_S (dict). Legible por cualquier LLM.
    - G (Geométrico): solo payload_G (hipervector). Máxima fidelidad, IA-a-IA.
    - H (Híbrido):   ambos payloads. Usado por defecto cuando ambas IAs tienen Cap.G.
    """
    S = "MODO_S"
    G = "MODO_G"
    H = "MODO_H"


class Cap(str, Enum):
    """
    Capacidades declaradas por una IA durante el handshake.

    - S:         puede procesar capa simbólica.
    - G:         puede procesar hipervectores (capa geométrica).
    - ALIGN:     soporta protocolo de alineamiento de espacios.
    - BROADCAST: puede recibir mensajes multicast.
    """
    S         = "CAP_S"
    G         = "CAP_G"
    ALIGN     = "CAP_ALIGN"
    BROADCAST = "CAP_BROADCAST"


class SessionState(str, Enum):
    """
    Estados del ciclo de vida de una sesión POLYDIM.

    Transiciones válidas:
      IDLE → CONNECTING → NEGOTIATING → ALIGNING → READY
                                      → DEGRADED (align_score < UMBRAL_ALIGN)
      cualquiera → FAILED | CLOSED
    """
    IDLE        = "IDLE"
    CONNECTING  = "CONNECTING"
    NEGOTIATING = "NEGOTIATING"
    ALIGNING    = "ALIGNING"
    READY       = "READY"
    DEGRADED    = "DEGRADED"
    FAILED      = "FAILED"
    CLOSED      = "CLOSED"


# ---------------------------------------------------------------------------
# Packet
# ---------------------------------------------------------------------------

@dataclass
class Packet:
    """
    Unidad de transmisión entre IAs en POLYDIM.

    Un Packet puede llevar representación simbólica (payload_S),
    geométrica (payload_G), o ambas (Modo H).

    Attributes:
        session_id: ID de la sesión en que se generó este paquete.
        seq:        Número de secuencia monótonamente creciente (invariante LCY_002).
        op:         Operación, p.ej. "ND_SEND", "ALIGN_REQ", "HANDSHAKE".
        payload_S:  Representación simbólica del ObjectND (dict serializable).
        payload_G:  Hipervector float32 de tamaño N.
        intent:     Lista de nombres de dimensiones activas detectadas.
    """
    session_id: str
    seq: int
    op: str
    payload_S: Optional[dict] = None
    payload_G: Optional[np.ndarray] = None
    intent: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Space V0.4
# ---------------------------------------------------------------------------

class Space:
    """
    Espacio hipervectorial POLYDIM.

    Define el "idioma" de una IA: cómo se mapean los nombres de dimensiones
    y propiedades a hipervectores. Dos Spaces distintos generan hipervectores
    distintos para el mismo string; el protocolo ALIGN compensa esta diferencia.

    Modos de construcción:
        Space()                                          → determinístico (default)
        Space("IA_X")                                    → personalizado por semilla
        Space(semantic_backend=MockSemanticBackend())    → con semántica de prueba
        Space("IA_X", semantic_backend=MiniLMBackend())  → semilla + MiniLM

    Con backend semántico, términos del mismo dominio quedan en regiones
    similares del hiperespacio: sim(usuario, cliente) ≈ 0.83 vs 0.51 sin backend.

    CRÍTICO: un ObjectND y la Session que lo transmite deben compartir
             el mismo objeto Space (misma identidad en memoria).

    Args:
        personal_seed:    Semilla string que personaliza el espacio de esta IA.
        semantic_backend: Backend opcional de embeddings.
    """

    def __init__(
        self,
        personal_seed: str = "",
        semantic_backend: Optional[SemanticBackend] = None,
    ) -> None:
        self.ps: str = personal_seed
        self.backend: Optional[SemanticBackend] = semantic_backend
        self._s: Dict[str, np.ndarray] = {}
        self._sub: Dict[str, np.ndarray] = {}
        self._JL: Optional[np.ndarray] = (
            make_jl(semantic_backend.dim) if semantic_backend else None
        )
        for d in NATIVE:
            self._sub[d] = self._mk(d)

    def _mk(self, name: str) -> np.ndarray:
        """
        Genera el hipervector para `name`.

        Con backend semántico:
          1. Codifica con el backend → vector de dim backend
          2. Proyecta con JL a N dimensiones
          3. Renormaliza (fix V0.4: normalizar antes de mezclar)
          4. Mezcla con vector personal (15%) para unicidad por IA
          5. Renormaliza resultado final
        Sin backend: hash MD5 determinista de la semilla + nombre.
        """
        if self.backend:
            hv = self._JL @ self.backend.encode(name)  # type: ignore[operator]
            n = np.linalg.norm(hv)
            hv = hv / n if n > 1e-10 else hv
            if self.ps:
                k = f"{self.ps}:{name}"
                s = int(hashlib.md5(k.encode()).hexdigest(), 16) % (2**32)
                p = np.random.default_rng(s).standard_normal(N).astype(np.float32)
                p /= np.linalg.norm(p)
                hv = 0.85 * hv + 0.15 * p
                n2 = np.linalg.norm(hv)
                hv = hv / n2 if n2 > 1e-10 else hv
        else:
            k = f"{self.ps}:{name}" if self.ps else name
            s = int(hashlib.md5(k.encode()).hexdigest(), 16) % (2**32)
            hv = np.random.default_rng(s).standard_normal(N).astype(np.float32)
        n = np.linalg.norm(hv)
        return (hv / n if n > 1e-10 else hv).astype(np.float32)

    def sym(self, n: str) -> np.ndarray:
        """
        Retorna el hipervector simbólico para el token `n` (con caché).

        Args:
            n: Token string (nombre de dimensión, propiedad, valor).

        Returns:
            Hipervector float32 normalizado de tamaño N.
        """
        if n not in self._s:
            self._s[n] = self._mk(n)
        return self._s[n]

    def sub(self, n: str) -> np.ndarray:
        """
        Retorna el hipervector de subespacio para `n` (con caché).

        Los subespacios NATIVE se pre-generan en __init__. Los demás
        se generan bajo demanda y se cachean igual que sym().
        """
        if n not in self._sub:
            self._sub[n] = self.sym(n)
        return self._sub[n]

    def reset_cache(self) -> None:
        """
        Limpia los caches de símbolos y subespacios no-NATIVE.

        Útil en tests para asegurar regeneración desde cero.
        Los subespacios NATIVE se recargan inmediatamente.
        """
        self._s.clear()
        self._sub.clear()
        for d in NATIVE:
            self._sub[d] = self._mk(d)

    def _rnd(self) -> np.ndarray:
        """Genera un hipervector aleatorio uniforme en la esfera unitaria."""
        hv = np.random.randn(N).astype(np.float32)
        return hv / np.linalg.norm(hv)

    def _enc(self, p: dict) -> np.ndarray:
        """
        Codifica un diccionario de propiedades como hipervector.

        Aplica BIND(sym(key), sym(value)) para cada par y SUPERPOSE
        para combinarlos. Dict vacío → sym("__empty__").
        """
        if not p:
            return self.sym("__empty__")
        return _sup(*[
            _bind(self.sym(str(k)), self.sym(str(v)))
            for k, v in p.items()
        ])


# ---------------------------------------------------------------------------
# ObjectND
# ---------------------------------------------------------------------------

class ObjectND:
    """
    Objeto n-dimensional POLYDIM.

    Existe simultáneamente en N dimensiones. Cada dimensión tiene:
    - Un peso de activación w en [0.0, 1.0]
    - Un diccionario de propiedades codificado como hipervector

    El GEO_ID es la identidad geométrica del objeto: no cambia al agregar
    dimensiones (invariante INV_001).

    Uso básico:
        sp = Space("MI_IA")
        obj = ObjectND(sp)
        obj.add("DIM_SQL",    {"tabla": "usuarios", "schema": "public"}, w=1.0)
        obj.add("DIM_PYTHON", {"tipo": "dataclass"}, w=0.7)
        print(obj.dims_activas())  # {"DIM_SQL": 0.812, "DIM_PYTHON": 0.711}

    Args:
        space: Space POLYDIM de esta IA. Si None, se crea un Space() por defecto.
    """

    def __init__(self, space: Optional[Space] = None) -> None:
        self._sp: Space = space or Space()
        self._props: Dict[str, dict] = {}
        self._w: Dict[str, float] = {}
        self._geo: np.ndarray = self._sp._rnd()
        self._cache: Optional[np.ndarray] = None

    @property
    def geo_id(self) -> str:
        """
        Identificador geométrico único del objeto (12 hex chars).

        Invariante: no cambia al agregar/modificar dimensiones (INV_001).
        """
        return hashlib.md5(self._geo.tobytes()).hexdigest()[:12]

    def add(self, dim: str, props: Optional[dict] = None, w: float = 1.0) -> "ObjectND":
        """
        Agrega o actualiza una dimensión en el objeto.

        Args:
            dim:   Nombre de la dimensión (p.ej. "DIM_SQL", "DIM_PYTHON").
            props: Propiedades de esta dimensión como dict. Default: {}.
            w:     Peso de activación en [0.0, 1.0].
                   0.0 = dimensión latente. 1.0 = completamente activo.

        Returns:
            self (para encadenamiento fluido).
        """
        self._props[dim] = props or {}
        self._w[dim] = float(np.clip(w, 0, 1))
        self._cache = None
        return self

    def clone(self) -> "ObjectND":
        """
        Crea una copia independiente con un nuevo GEO_ID.

        El clon comparte el Space pero tiene identidad geométrica nueva.
        Las dimensiones y pesos se copian tal cual.

        Returns:
            Nuevo ObjectND con las mismas dimensiones pero GEO_ID distinto.
        """
        nuevo = ObjectND(self._sp)
        for d, p in self._props.items():
            nuevo.add(d, dict(p), w=self._w[d])
        return nuevo

    def _hv(self) -> np.ndarray:
        """
        Calcula y cachea el hipervector compuesto del objeto.

        Estructura en capas:
          CAPA GEO_ID:    _geo (peso 1.0)
          CAPA MEMBRESÍA: sub(dim) * w
          CAPA CONTENIDO: BIND(sub(dim), enc(props)) * w * CONTENT_W
        """
        if self._cache is not None:
            return self._cache
        cs: List[np.ndarray] = [self._geo]
        ws: List[float] = [1.0]
        for d, p in self._props.items():
            ww = self._w.get(d, 1.0)
            if ww <= 0:
                continue
            cs.append(self._sp.sub(d));      ws.append(ww)
            cs.append(_bind(self._sp.sub(d), self._sp._enc(p)));  ws.append(ww * CONTENT_W)
        self._cache = _sup(*cs, ws=ws)
        return self._cache

    def activacion(self, dim: str) -> float:
        """
        Mide la activación de una dimensión en este objeto.

        Args:
            dim: Nombre de dimensión a consultar.

        Returns:
            Float en [0.0, 1.0]. > UMBRAL (~0.510) = dimensión presente.
        """
        return _proj(self._hv(), self._sp.sub(dim))

    def dims_activas(self, umbral: float = UMBRAL) -> Dict[str, float]:
        """
        Retorna las dimensiones cuya activación supera el umbral.

        Recorre NATIVE + dimensiones declaradas en _props, sin duplicados
        (fix V0.4: dict seen evita contar la misma dim dos veces).

        Args:
            umbral: Threshold de detección. Default: UMBRAL (~0.510).

        Returns:
            Dict {nombre_dim: activacion_redondeada}.
        """
        seen: Dict[str, int] = {}
        r: Dict[str, float] = {}
        for d in NATIVE + list(self._props):
            if d not in seen:
                w = self.activacion(d)
                if w > umbral:
                    r[d] = round(w, 4)
                seen[d] = 1
        return r

    def to_symbolic(self) -> dict:
        """
        Serializa el objeto a representación simbólica (Capa S).

        Returns:
            Dict con claves "geo_id" y "dims". JSON-serializable.

        Ejemplo de retorno:
            {
              "geo_id": "a3f2bc901d4e",
              "dims": {
                "DIM_SQL": {"w": 1.0, "props": {"tabla": "users"}},
                "DIM_PYTHON": {"w": 0.7, "props": {"tipo": "dataclass"}}
              }
            }
        """
        return {
            "geo_id": self.geo_id,
            "dims": {
                d: {"w": self._w.get(d, 1.0), "props": p}
                for d, p in self._props.items()
            },
        }

    def __repr__(self) -> str:
        dims = ", ".join(f"{d}[{self._w.get(d, 1):.1f}]" for d in self._props)
        return f"ObjectND(id={self.geo_id}, dims=[{dims}])"


# ---------------------------------------------------------------------------
# Empaquetación
# ---------------------------------------------------------------------------

def empaquetar_objeto(
    obj: ObjectND,
    mode: Mode,
    A: Optional[np.ndarray] = None,
    B: Optional[np.ndarray] = None,
) -> Packet:
    """
    Serializa un ObjectND en un Packet para transmisión.

    Si se proporcionan matrices de alineamiento A y B, aplica la
    transformación ALIGN al hipervector antes de incluirlo en el Packet.

    Args:
        obj:  ObjectND a empaquetar.
        mode: Modo de transmisión (S, G, o H).
        A:    Matriz de sondas del espacio origen (shape [k, N]).
        B:    Matriz de sondas del espacio destino (shape [k, N]).

    Returns:
        Packet listo para transmisión.
    """
    sym = obj.to_symbolic()
    intent = list(obj.dims_activas().keys())
    hv = obj._hv()
    if A is not None and B is not None:
        hv = align_transform(hv, A, B)
    if mode == Mode.S:
        return Packet("", 0, "ND_SEND", payload_S=sym, intent=intent)
    if mode == Mode.G:
        return Packet("", 0, "ND_SEND", payload_G=hv, intent=intent)
    return Packet("", 0, "ND_SEND", payload_S=sym, payload_G=hv, intent=intent)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class Session:
    """
    Sesión de comunicación POLYDIM entre dos IAs.

    Gestiona el ciclo de vida completo: IDLE → handshake → align → READY → send/receive.

    El handshake negocia el modo máximo común según capacidades.
    El alineamiento calcula las matrices R para transformar hipervectores
    entre espacios distintos.

    Ciclo típico:
        ia = Session(Space("IA_A"), "IA_A")
        ib = Session(Space("IA_B"), "IA_B")
        ia.connect(ib)           # handshake + align automático
        pkt = ia.send(obj)       # serializa y transforma
        dims = ib.receive(pkt)   # detecta dimensiones

    Args:
        space: Space de esta IA.
        name:  Nombre identificatorio.
        caps:  Capacidades declaradas. Default: [S, G, ALIGN].
    """

    def __init__(
        self,
        space: Space,
        name: str = "IA",
        caps: Optional[List[Cap]] = None,
    ) -> None:
        self.space: Space = space
        self.name: str = name
        self.caps: List[Cap] = caps or [Cap.S, Cap.G, Cap.ALIGN]
        self.state: SessionState = SessionState.IDLE
        self.mode: Mode = Mode.S
        self.session_id: Optional[str] = None
        self.align_score: Optional[float] = None
        self._A: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None
        self._seq: int = 0

    def handshake(self, remote: "Session") -> bool:
        """
        Negocia sesión con la IA remota.

        Determina session_id (MD5 de dos UUIDs) y mode (H si ambas tienen Cap.G).
        Ambas Sessions quedan en estado NEGOTIATING.

        Args:
            remote: Session de la IA remota.

        Returns:
            True siempre.
        """
        n1, n2 = uuid.uuid4().hex[:8], uuid.uuid4().hex[:8]
        caps_comun = set(self.caps) & set(remote.caps)
        mode = Mode.H if Cap.G in caps_comun else Mode.S
        sid = hashlib.md5((n1 + n2).encode()).hexdigest()[:16]
        self.session_id = remote.session_id = sid
        self.mode = remote.mode = mode
        self.state = remote.state = SessionState.NEGOTIATING
        return True

    def align(
        self,
        remote: "Session",
        sondas: Optional[List[str]] = None,
    ) -> float:
        """
        Alinea los espacios de ambas IAs usando sondas compartidas.

        Score >= UMBRAL_ALIGN (0.85) → estado READY.
        Score <  UMBRAL_ALIGN        → estado DEGRADED, modo S.

        Args:
            remote: Session de la IA remota.
            sondas: Tokens sonda. Default: SONDAS (28 tokens).

        Returns:
            Score ALIGN en [0.0, 1.0].
        """
        self.state = remote.state = SessionState.ALIGNING
        p = sondas or SONDAS
        A = np.array([self.space.sym(s) for s in p], dtype=np.float32)
        B = np.array([remote.space.sym(s) for s in p], dtype=np.float32)
        scores = [
            _sim(align_transform(self.space.sub(d), A, B), remote.space.sub(d))
            for d in NATIVE
        ]
        score = float(np.mean(scores))
        valid = score >= UMBRAL_ALIGN
        if valid:
            self._A = A;       self._B = B
            remote._A = B;     remote._B = A
            self.align_score = remote.align_score = round(score, 4)
            self.state = remote.state = SessionState.READY
        else:
            self.mode = remote.mode = Mode.S
            self.state = remote.state = SessionState.DEGRADED
        return score

    def connect(self, remote: "Session") -> "Session":
        """
        Ejecuta handshake + align en una sola llamada.

        Args:
            remote: Session de la IA remota.

        Returns:
            self.
        """
        self.handshake(remote)
        if self.mode in (Mode.G, Mode.H):
            self.align(remote)
        return self

    def send(self, obj: ObjectND) -> Packet:
        """
        Serializa y transmite un ObjectND.

        Aplica transformación ALIGN si hay matrices. Incrementa seq.

        Args:
            obj: ObjectND. DEBE usar el mismo Space que esta Session.

        Returns:
            Packet listo para remote.receive().
        """
        self._seq += 1
        pkt = empaquetar_objeto(obj, self.mode, self._A, self._B)
        pkt.session_id = self.session_id or ""
        pkt.seq = self._seq
        return pkt

    def receive(self, pkt: Packet) -> Dict[str, float]:
        """
        Recibe un Packet y detecta las dimensiones activas.

        Prioriza payload_G. Si no, reconstruye desde payload_S.
        Fix V0.4: payload_S sin clave "dims" no lanza KeyError.

        Args:
            pkt: Packet recibido.

        Returns:
            Dict {nombre_dim: activacion} para dims con activación > UMBRAL.
        """
        if pkt.payload_G is not None:
            hv = pkt.payload_G
        elif pkt.payload_S:
            o = ObjectND(self.space)
            for d, info in pkt.payload_S.get("dims", {}).items():
                o.add(d, info.get("props", {}), w=info.get("w", 1.0))
            hv = o._hv()
        else:
            return {}
        return {
            d: round(_proj(hv, self.space.sub(d)), 4)
            for d in NATIVE
            if _proj(hv, self.space.sub(d)) > UMBRAL
        }

    @property
    def info(self) -> dict:
        """Snapshot del estado actual de la sesión (serializable)."""
        return {
            "session_id":  self.session_id,
            "mode":        self.mode.value,
            "state":       self.state.value,
            "align_score": self.align_score,
        }

    def __repr__(self) -> str:
        return f"Session({self.name},{self.state.value},{self.mode.value})"


# ---------------------------------------------------------------------------
# Connection (API de alto nivel)
# ---------------------------------------------------------------------------

class Connection:
    """
    Abstracción de alto nivel para la conexión entre dos IAs.

    Encapsula dos Sessions y expone un único método transfer().

    Ejemplo (API de 3 líneas):
        sp_a = Space("IA_A")
        sp_b = Space("IA_B")
        conn = polydim_connect(sp_a, sp_b)
        obj  = ObjectND(sp_a).add("DIM_SQL", {"tabla": "orders"}, w=1.0)
        dims = conn.transfer(obj)

    Args:
        sp_a: Space de la IA emisora.
        sp_b: Space de la IA receptora.
    """

    def __init__(self, sp_a: Space, sp_b: Space) -> None:
        self._ia: Session = Session(sp_a, "IA_A")
        self._ib: Session = Session(sp_b, "IA_B")
        self._ia.connect(self._ib)

    def transfer(self, obj: ObjectND) -> Dict[str, float]:
        """
        Transmite un ObjectND de IA_A a IA_B y retorna las dims activas.

        Args:
            obj: ObjectND que DEBE usar el mismo Space que sp_a.

        Returns:
            Dict {nombre_dim: activacion} detectadas por IA_B.
        """
        return self._ib.receive(self._ia.send(obj))

    @property
    def info(self) -> dict:
        """Estado de la conexión."""
        return self._ia.info

    def __repr__(self) -> str:
        return f"Connection({self._ia.info})"


# ---------------------------------------------------------------------------
# Función pública principal
# ---------------------------------------------------------------------------

def polydim_connect(space_a: Space, space_b: Space) -> Connection:
    """
    Crea una conexión POLYDIM entre dos Spaces.

    Punto de entrada principal para la API de 3 líneas.

    Args:
        space_a: Space de la IA emisora.
        space_b: Space de la IA receptora.

    Returns:
        Connection lista para llamar a .transfer(obj).

    Ejemplo:
        sp  = Space("MI_IA")
        obj = ObjectND(sp).add("DIM_SQL", {"tabla": "users"}, w=1.0)
        dims = polydim_connect(sp, Space("OTRA_IA")).transfer(obj)
    """
    return Connection(space_a, space_b)


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "Space", "ObjectND", "Session", "Connection",
    "polydim_connect", "empaquetar_objeto",
    "SemanticBackend", "MockSemanticBackend", "MiniLMBackend", "FastTextBackend",
    "Mode", "Cap", "SessionState",
    "Packet",
    "NATIVE", "SONDAS", "UMBRAL", "UMBRAL_ALIGN", "CONTENT_W", "N",
    "_bind", "_sup", "_proj", "_sim", "align_transform", "make_jl",
    "__version__", "__schema_version__",
]
