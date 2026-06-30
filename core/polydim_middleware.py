# POLYDIM_DEST
# destination: polydim_v1/core/
# filename:    polydim_middleware.py
# author:      ai.mpat.agt@gmail.com (claude-sonnet-4-6)
# fecha:       2026-06-27
# tarea:       TASK_041
# roadmap:     Camino 3 — instrumentación via middleware

"""
POLYDIM Middleware — V0.1
==========================
Conecta cualquier LLM de producción con el protocolo POLYDIM,
habilitando comunicación IA↔IA via POLYDIM_BIN sin fine-tuning.

FLUJO CAMINO 3:
  Texto → [LLM_A] → inferir ObjectND → POLYDIM_BIN → [LLM_B] → respuesta

VALIDACIÓN (sandbox, 2026-06-27):
  DIM_SQL detectado por LLM_B: 0.9001  (vs 0.8026 en LLM_A)
  align_score A↔B: 0.9992
  Packet size: 40 014 bytes por objeto

USO BÁSICO:
  # Inicializar dos agentes
  agent_a = PolydimAgent("LLM_A", llm_fn=mi_llm_api)
  agent_b = PolydimAgent("LLM_B", llm_fn=otro_llm_api)
  agent_a.connect(agent_b)

  # IA_A envía intención
  response = agent_a.send_intent(
      "Consultar tabla usuarios por email",
      dims={"DIM_SQL": {"tabla": "usuarios", "filtro": "email"}}
  )

  # IA_B recibe y responde
  agent_b.receive_and_respond()

COMPATIBILIDAD LLM:
  Cualquier LLM con API de completions (OpenAI, Anthropic, Ollama, etc.)
  La función llm_fn debe tener firma: llm_fn(prompt: str) -> str

Autor:   ai.mpat.agt@gmail.com
Versión: V0.1 — 2026-06-27
Roadmap: POLYDIM_ROADMAP_TRANSFORMER_V0.md — Camino 3
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from polydim_runtime_v04 import (
        Space, ObjectND, Session, Cap, Mode, SessionState,
        _sim, _proj, NATIVE, UMBRAL, N,
    )
    from polydim_weighted_inference import infer_weights
    from polydim_drift_detection import DriftDetector
except ImportError as e:
    raise ImportError(
        f"polydim_middleware requiere polydim_runtime_v04.py, "
        f"polydim_weighted_inference.py y polydim_drift_detection.py en el path. {e}"
    )

# ---------------------------------------------------------------------------
# Constantes del formato binario (SPEC_FORMATO_BINARIO_V0.md)
# ---------------------------------------------------------------------------

MAGIC   = b'PDIM'
VERSION = 4


# ---------------------------------------------------------------------------
# Helpers POLYDIM_BIN
# ---------------------------------------------------------------------------

def _pack_g(hv: np.ndarray, geo_id: str) -> bytes:
    """Empaqueta hipervector en POLYDIM_BIN Modo G."""
    geo_raw = bytes.fromhex(geo_id + geo_id)[:6]
    return (
        MAGIC
        + struct.pack(">BBH", VERSION, 0x01, N // 100)
        + geo_raw
        + hv.astype(np.float32).tobytes()
    )


def _unpack_hv(data: bytes) -> np.ndarray:
    """Desempaqueta hipervector de POLYDIM_BIN."""
    assert data[:4] == MAGIC, "MAGIC inválido"
    n = struct.unpack(">H", data[6:8])[0] * 100
    return np.frombuffer(data[14:14 + n * 4], dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# IntentParser — texto → ObjectND
# ---------------------------------------------------------------------------

class IntentParser:
    """
    Convierte texto en ObjectND usando heurísticas de keywords.

    Para Camino 3, el parser es determinista y basado en palabras clave.
    En Camino 1, este componente será reemplazado por el modelo fine-tuneado.

    Args:
        space: Space del agente que usa este parser.
    """

    # Palabras clave por dimensión nativa
    KEYWORDS: Dict[str, List[str]] = {
        "DIM_SQL":     ["tabla", "query", "select", "insert", "update", "delete",
                        "schema", "columna", "fila", "join", "sql", "base de datos",
                        "database", "table", "row", "column", "index"],
        "DIM_PYTHON":  ["función", "clase", "módulo", "script", "python", "código",
                        "función", "list", "dict", "import", "def", "class", "async"],
        "DIM_FLUTTER": ["widget", "pantalla", "formulario", "botón", "ui", "flutter",
                        "scaffold", "column", "row", "text", "button", "state"],
        "DIM_RUST":    ["ownership", "borrow", "lifetime", "struct", "trait", "rust",
                        "memory", "unsafe", "cargo", "crate"],
        "DIM_GRAPH":   ["nodo", "arista", "grafo", "relación", "graph", "node", "edge",
                        "conexión", "red", "network"],
        "DIM_VECTOR":  ["embedding", "similitud", "vector", "semantic", "coseno",
                        "similarity", "encode", "cluster"],
        "DIM_TIME":    ["timestamp", "evento", "secuencia", "tiempo", "fecha",
                        "time", "schedule", "orden", "sequence"],
        "DIM_ERROR":   ["error", "excepción", "fallo", "timeout", "crash", "bug",
                        "exception", "retry", "fallback"],
        "DIM_META":    ["versión", "metadata", "audit", "origen", "version",
                        "author", "created", "modified"],
    }

    def __init__(self, space: Space) -> None:
        self.space = space

    def parse(self, text: str, extra_props: Optional[Dict[str, dict]] = None) -> ObjectND:
        """
        Convierte texto en ObjectND.

        Calcula pesos por frecuencia de keywords, luego agrega props extra.

        Args:
            text:        Texto a analizar.
            extra_props: Props adicionales por dimensión {dim: {k: v}}.

        Returns:
            ObjectND con dimensiones ponderadas por relevancia del texto.
        """
        text_lower = text.lower()
        obj = ObjectND(self.space)

        for dim, keywords in self.KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text_lower)
            if hits > 0:
                w = min(1.0, hits * 0.25)  # 0.25 por keyword, máx 1.0
                props = extra_props.get(dim, {}) if extra_props else {}
                obj.add(dim, props, w=w)

        # Si no detectó nada, agregar DIM_META como base
        if not obj._props:
            obj.add("DIM_META", {"raw_text": text[:100]}, w=0.5)

        return obj

    def dims_from_text(self, text: str) -> Dict[str, float]:
        """Retorna estimación de pesos por dimensión sin crear ObjectND."""
        text_lower = text.lower()
        result = {}
        for dim, keywords in self.KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text_lower)
            if hits > 0:
                result[dim] = min(1.0, hits * 0.25)
        return result


# ---------------------------------------------------------------------------
# PolydimAgent — agente con identidad POLYDIM
# ---------------------------------------------------------------------------

@dataclass
class AgentMessage:
    """Mensaje intercambiado entre agentes POLYDIM.

    Attributes:
        sender:      Nombre del agente emisor.
        text:        Texto en lenguaje natural (para el humano / LLM).
        object_nd:   ObjectND asociado al mensaje.
        packet:      POLYDIM_BIN serializado.
        dims_active: Dimensiones activas detectadas en el receptor.
        timestamp:   Unix timestamp de creación.
    """
    sender: str
    text: str
    object_nd: ObjectND
    packet: bytes
    dims_active: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class PolydimAgent:
    """
    Agente que envuelve un LLM con el protocolo POLYDIM.

    Permite que cualquier LLM participe en comunicación IA↔IA
    via POLYDIM_BIN sin fine-tuning ni modificación del modelo.

    Args:
        name:    Identificador único del agente.
        llm_fn:  Función que llama al LLM. Firma: (prompt: str) -> str.
                 Si None, el agente opera en modo solo-POLYDIM (sin LLM).
        n_space: Semilla para el Space de este agente.

    Ejemplo con OpenAI:
        import openai
        def my_llm(prompt):
            return openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content

        agent = PolydimAgent("GPT_A", llm_fn=my_llm)

    Ejemplo con Anthropic:
        import anthropic
        client = anthropic.Anthropic()
        def my_llm(prompt):
            return client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            ).content[0].text

        agent = PolydimAgent("CLAUDE_A", llm_fn=my_llm)
    """

    def __init__(
        self,
        name: str,
        llm_fn: Optional[Callable[[str], str]] = None,
        n_space: str = "",
    ) -> None:
        self.name = name
        self.llm_fn = llm_fn
        self.space = Space(n_space or name)
        self.session: Optional[Session] = None
        self.parser = IntentParser(self.space)
        self.drift = DriftDetector()
        self._history: List[AgentMessage] = []
        self._remote: Optional["PolydimAgent"] = None

    def connect(self, remote: "PolydimAgent") -> float:
        """
        Establece sesión POLYDIM con otro agente.

        Args:
            remote: Agente remoto a conectar.

        Returns:
            align_score de la sesión establecida.
        """
        self.session = Session(self.space, self.name)
        remote.session = Session(remote.space, remote.name)
        self.session.connect(remote.session)
        self._remote = remote
        remote._remote = self
        return self.session.align_score or 0.0

    def send_intent(
        self,
        text: str,
        dims: Optional[Dict[str, dict]] = None,
        use_llm: bool = True,
    ) -> AgentMessage:
        """
        Convierte texto en ObjectND y lo envía al agente remoto.

        Si use_llm=True y hay llm_fn, enriquece el texto con el LLM
        antes de inferir el ObjectND.

        Args:
            text:    Intención en texto natural.
            dims:    Props explícitas por dimensión {dim: {k: v}}.
            use_llm: Si True, usa el LLM para enriquecer la intención.

        Returns:
            AgentMessage con el objeto enviado y dims detectadas en remoto.
        """
        assert self.session is not None, "Llamar connect() primero"
        assert self._remote is not None, "Llamar connect() primero"

        # Opcional: enriquecer con LLM
        enriched_text = text
        if use_llm and self.llm_fn:
            prompt = (
                f"Analiza esta intención y extrae: tabla/endpoint/módulo principal, "
                f"operación (read/write/transform), y contexto técnico relevante. "
                f"Responde en JSON con claves 'tabla', 'operacion', 'contexto'.\n\n"
                f"Intención: {text}"
            )
            try:
                llm_response = self.llm_fn(prompt)
                enriched_text = f"{text} | {llm_response}"
            except Exception:
                pass  # Fallback a texto original si LLM falla

        # Inferir ObjectND
        obj = self.parser.parse(enriched_text, extra_props=dims)

        # Monitorear drift
        drift_event = self.drift.update(obj)

        # Enviar via Session (con ALIGN)
        pkt = self.session.send(obj)

        # Recibir dims detectadas en remoto
        dims_detected = self._remote.session.receive(pkt)

        # Registrar en historial del remoto
        msg = AgentMessage(
            sender=self.name,
            text=text,
            object_nd=obj,
            packet=pkt.payload_G.tobytes() if pkt.payload_G is not None else b"",
            dims_active=dims_detected,
        )
        self._remote._history.append(msg)
        self._history.append(msg)

        return msg

    def receive_and_respond(
        self,
        use_llm: bool = True,
    ) -> Optional[AgentMessage]:
        """
        Procesa el último mensaje recibido y genera respuesta.

        Si use_llm=True y hay llm_fn, genera texto de respuesta natural.

        Returns:
            AgentMessage de respuesta, o None si no hay mensajes.
        """
        if not self._history:
            return None

        last = self._history[-1]
        if last.sender == self.name:
            return None  # El último mensaje es propio

        # Construir respuesta ObjectND basada en dims detectadas
        response_obj = ObjectND(self.space)
        for dim, activation in last.dims_active.items():
            if activation > UMBRAL:
                response_obj.add(dim, {"role": "response", "to": last.sender}, w=activation)

        # Opcional: generar texto de respuesta con LLM
        response_text = f"ACK from {self.name}"
        if use_llm and self.llm_fn:
            dims_str = ", ".join(
                f"{d}={v:.2f}" for d, v in last.dims_active.items()
            )
            prompt = (
                f"Eres un agente AI llamado {self.name}. "
                f"Recibiste una intención con estas dimensiones activas: {dims_str}. "
                f"Texto original: '{last.text}'. "
                f"Genera una respuesta técnica concisa (máx 2 oraciones)."
            )
            try:
                response_text = self.llm_fn(prompt)
            except Exception:
                pass

        # Enviar respuesta
        assert self.session is not None
        pkt = self.session.send(response_obj)
        dims_resp = self._remote.session.receive(pkt) if self._remote else {}

        msg = AgentMessage(
            sender=self.name,
            text=response_text,
            object_nd=response_obj,
            packet=pkt.payload_G.tobytes() if pkt.payload_G is not None else b"",
            dims_active=dims_resp,
        )
        self._history.append(msg)
        if self._remote:
            self._remote._history.append(msg)

        return msg

    def conversation_report(self) -> str:
        """Reporte textual del historial de la conversación."""
        lines = [f"=== Conversación {self.name} ==="]
        for i, msg in enumerate(self._history):
            lines.append(
                f"[{i+1}] {msg.sender}: '{msg.text[:60]}...'"
                if len(msg.text) > 60 else
                f"[{i+1}] {msg.sender}: '{msg.text}'"
            )
            if msg.dims_active:
                dims_str = ", ".join(f"{d}={v:.3f}" for d, v in msg.dims_active.items())
                lines.append(f"     dims_detectadas: {dims_str}")
        drift_r = self.drift.report()
        lines.append(f"\nDrift: max={drift_r.drift_max:.4f}  eventos={drift_r.n_eventos}")
        return "\n".join(lines)

    @property
    def align_score(self) -> Optional[float]:
        """Score ALIGN de la sesión activa."""
        return self.session.align_score if self.session else None

    def __repr__(self) -> str:
        return f"PolydimAgent({self.name}, align={self.align_score})"


# ---------------------------------------------------------------------------
# Función de conveniencia: demo sin LLM
# ---------------------------------------------------------------------------

def demo_middleware_sin_llm() -> None:
    """
    Demo del middleware sin LLM externo.

    Simula dos agentes comunicándose via POLYDIM_BIN usando
    solo el IntentParser basado en keywords.
    """
    print("=== POLYDIM Middleware Demo (sin LLM) ===\n")

    # Crear agentes
    agent_a = PolydimAgent("IA_Backend", llm_fn=None)
    agent_b = PolydimAgent("IA_Frontend", llm_fn=None)

    # Conectar
    score = agent_a.connect(agent_b)
    print(f"Sesión establecida. align_score = {score:.4f}\n")

    # IA_A envía intención SQL
    msg1 = agent_a.send_intent(
        "Necesito consultar la tabla usuarios filtrando por email activo",
        dims={"DIM_SQL": {"tabla": "usuarios", "filtro": "email", "status": "activo"}}
    )
    print(f"IA_Backend envió: '{msg1.text}'")
    print(f"IA_Frontend detectó: {msg1.dims_active}\n")

    # IA_B responde
    msg2 = agent_b.receive_and_respond(use_llm=False)
    if msg2:
        print(f"IA_Frontend respondió (dims): {msg2.dims_active}\n")

    # IA_A envía intención Flutter
    msg3 = agent_a.send_intent(
        "Mostrar formulario de login con campo email y botón submit",
        dims={"DIM_FLUTTER": {"widget": "LoginForm", "campos": "email,password"}}
    )
    print(f"IA_Backend envió: '{msg3.text}'")
    print(f"IA_Frontend detectó: {msg3.dims_active}\n")

    # Reporte
    print(agent_a.conversation_report())


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "PolydimAgent",
    "AgentMessage",
    "IntentParser",
    "demo_middleware_sin_llm",
]


if __name__ == "__main__":
    demo_middleware_sin_llm()
