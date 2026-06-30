# POLYDIM_DEST
# destination: polydim_v1/core/
# filename:    polydim_dataset_generator.py
# author:      ai.mpat.agt@gmail.com (claude-sonnet-4-6)
# fecha:       2026-06-27
# tarea:       TASK_044

"""
POLYDIM Dataset Generator — V0.1
===================================
Pipeline para generar pares (texto_intención, ObjectND) como datos
de entrenamiento para fine-tuning (Camino 1).

DOS MODOS DE OPERACIÓN:
  1. Sin LLM (plantillas): genera datos sintéticos sin costo de API.
     Útil para validar el pipeline y generar un dataset base (~10k pares).

  2. Con LLM (producción): usa polydim_middleware.py para enriquecer
     los pares con variaciones reales del lenguaje natural.
     Requiere API key y budget. Estimado: $5-20 para 10k pares con GPT-4o-mini.

FORMATO DEL DATASET:
  JSONL (una línea por par) con estructura:
  {
    "text":           "Consultar tabla usuarios por email activo",
    "dims_declared":  {"DIM_SQL": {"tabla": "usuarios", ...}},
    "activations_gt": {"DIM_SQL": 0.8431, "DIM_PYTHON": 0.7123},
    "dominant_dim":   "DIM_SQL",
    "geo_id":         "a3f2bc901d4e",
    "source":         "template" | "llm"
  }

USO:
  # Modo plantillas (sin LLM, gratis)
  gen = DatasetGenerator()
  gen.generate(n=1000, output_path="polydim_dataset.jsonl")

  # Modo LLM (requiere llm_fn)
  import anthropic
  client = anthropic.Anthropic()
  def my_llm(prompt):
      return client.messages.create(
          model="claude-haiku-4-5-20251001",
          max_tokens=512,
          messages=[{"role": "user", "content": prompt}]
      ).content[0].text

  gen = DatasetGenerator(llm_fn=my_llm)
  gen.generate(n=10000, output_path="polydim_dataset_llm.jsonl")

ESTIMACIÓN DE COSTOS (Camino 1):
  GPT-4o-mini: ~$0.001 por par → $10 para 10k pares
  Claude Haiku: ~$0.0008 por par → $8 para 10k pars
  Llama 3 local (Ollama): $0 (solo cómputo)

Autor:   ai.mpat.agt@gmail.com
Versión: V0.1 — 2026-06-27
Roadmap: POLYDIM_ROADMAP_TRANSFORMER_V0.md — Camino 1, prerequisito
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

try:
    from polydim_runtime_v04 import Space, ObjectND, NATIVE, _proj, UMBRAL
    from polydim_weighted_inference import infer_weights, dominant_dim
except ImportError:
    raise ImportError("Requiere polydim_runtime_v04.py y polydim_weighted_inference.py")

# ---------------------------------------------------------------------------
# Plantillas de intención por dimensión
# ---------------------------------------------------------------------------

TEMPLATES: List[tuple] = [
    # DIM_SQL
    ("Consultar {tabla} filtrando por {campo}",
     {"DIM_SQL": {"tabla": "{tabla}", "op": "SELECT", "filtro": "{campo}"}}),
    ("Insertar nuevo {entidad} en {tabla} con {campo}={valor}",
     {"DIM_SQL": {"tabla": "{tabla}", "op": "INSERT"}, "DIM_PYTHON": {"tipo": "orm"}}),
    ("Actualizar {campo} de {tabla} donde id={valor}",
     {"DIM_SQL": {"tabla": "{tabla}", "op": "UPDATE", "campo": "{campo}"}}),
    ("Eliminar registros de {tabla} con status={valor}",
     {"DIM_SQL": {"tabla": "{tabla}", "op": "DELETE"}}),
    ("Join entre {tabla} y {tabla2} por {campo}",
     {"DIM_SQL": {"tabla_a": "{tabla}", "tabla_b": "{tabla2}", "op": "JOIN"}}),

    # DIM_FLUTTER
    ("Mostrar formulario de {accion} con campos {campos}",
     {"DIM_FLUTTER": {"widget": "Form", "accion": "{accion}", "campos": "{campos}"}}),
    ("Pantalla de {accion} con lista de {entidad}",
     {"DIM_FLUTTER": {"widget": "ListView", "accion": "{accion}", "source": "{entidad}"}}),
    ("Botón de {accion} que navega a {pantalla}",
     {"DIM_FLUTTER": {"widget": "ElevatedButton", "accion": "{accion}", "destino": "{pantalla}"}}),
    ("Barra de navegación con secciones {campos}",
     {"DIM_FLUTTER": {"widget": "BottomNavBar", "secciones": "{campos}"}}),

    # DIM_PYTHON
    ("Función que procesa lista de {entidad} y retorna {resultado}",
     {"DIM_PYTHON": {"tipo": "function", "input": "{entidad}", "output": "{resultado}"}}),
    ("Clase {entidad} con métodos {campos}",
     {"DIM_PYTHON": {"tipo": "class", "nombre": "{entidad}", "metodos": "{campos}"}}),
    ("Script que lee {tabla} de CSV y genera reporte",
     {"DIM_PYTHON": {"tipo": "script", "input": "csv", "output": "reporte"},
      "DIM_SQL": {"tabla": "{tabla}"}}),
    ("Dataclass para representar {entidad} con campos {campos}",
     {"DIM_PYTHON": {"tipo": "dataclass", "nombre": "{entidad}", "campos": "{campos}"}}),

    # DIM_ERROR
    ("Detectar error {codigo} en {servicio} y reintentar con backoff",
     {"DIM_ERROR": {"code": "{codigo}", "servicio": "{servicio}"},
      "DIM_TIME": {"retry": "exponential_backoff"}}),
    ("Manejar timeout en llamada a {servicio} después de {valor} segundos",
     {"DIM_ERROR": {"tipo": "timeout", "servicio": "{servicio}"},
      "DIM_TIME": {"timeout_s": "{valor}"}}),
    ("Fallback a caché cuando {servicio} retorna {codigo}",
     {"DIM_ERROR": {"code": "{codigo}"}, "DIM_VECTOR": {"fallback": "cache"}}),

    # DIM_GRAPH
    ("Grafo de relaciones entre {entidad} y {entidad2}",
     {"DIM_GRAPH": {"nodo_a": "{entidad}", "nodo_b": "{entidad2}", "tipo": "bidireccional"}}),
    ("Camino más corto entre {entidad} y {entidad2} en el grafo de {tabla}",
     {"DIM_GRAPH": {"algoritmo": "dijkstra", "source": "{entidad}", "target": "{entidad2}"},
      "DIM_SQL": {"tabla": "{tabla}"}}),

    # DIM_TIME
    ("Serie temporal de {campo} de {tabla} en los últimos {valor} días",
     {"DIM_TIME": {"ventana": "{valor}d"}, "DIM_SQL": {"tabla": "{tabla}", "campo": "{campo}"}}),
    ("Programar tarea de {accion} cada {valor} horas",
     {"DIM_TIME": {"schedule": "cron", "intervalo": "{valor}h", "tarea": "{accion}"}}),

    # DIM_RUST
    ("Struct Rust para {entidad} con campos {campos} y lifetime seguro",
     {"DIM_RUST": {"tipo": "struct", "nombre": "{entidad}", "lifetime": "safe"}}),
    ("Función Rust sin unsafe que procesa buffer de {entidad}",
     {"DIM_RUST": {"tipo": "function", "unsafe": False, "input": "{entidad}"}}),

    # DIM_META
    ("Documentar {entidad} con versión {valor} y autor {campo}",
     {"DIM_META": {"entidad": "{entidad}", "version": "{valor}", "autor": "{campo}"}}),
    ("Agregar metadata de auditoría a {tabla}: created_at, updated_by",
     {"DIM_META": {"audit": True}, "DIM_SQL": {"tabla": "{tabla}"}}),
]

VOCAB: Dict[str, List[str]] = {
    "tabla":    ["usuarios", "pedidos", "productos", "pagos", "sesiones",
                 "inventario", "categorias", "reviews", "notificaciones"],
    "tabla2":   ["roles", "permisos", "tags", "direcciones", "facturas"],
    "campo":    ["email", "id", "nombre", "fecha_creacion", "status",
                 "precio", "stock", "rating", "telefono"],
    "valor":    ["activo", "true", "false", "premium", "7", "30", "1.0", "v2"],
    "accion":   ["login", "registro", "edición", "búsqueda", "exportar",
                 "importar", "sincronizar", "validar"],
    "campos":   ["email,password", "nombre,apellido", "email,rol",
                 "id,nombre,precio", "fecha,monto,status"],
    "pantalla": ["Dashboard", "Perfil", "Configuración", "Detalle", "Lista"],
    "entidad":  ["Usuario", "Pedido", "Producto", "Pago", "Sesión"],
    "entidad2": ["Rol", "Categoría", "Proveedor", "Tag", "Dirección"],
    "resultado":["lista filtrada", "dict agrupado", "serie temporal",
                 "JSON response", "DataFrame"],
    "codigo":   ["404", "503", "429", "500", "401", "403"],
    "servicio": ["auth", "payments", "catalog", "notifications", "search"],
}


# ---------------------------------------------------------------------------
# DatasetSample
# ---------------------------------------------------------------------------

@dataclass
class DatasetSample:
    """Par (texto, ObjectND) para fine-tuning."""
    text: str
    dims_declared: Dict[str, dict]
    activations_gt: Dict[str, float]
    dominant_dim: Optional[str]
    geo_id: str
    source: str  # "template" | "llm"
    timestamp: float = field(default_factory=time.time)

    def to_jsonl(self) -> str:
        return json.dumps({
            "text": self.text,
            "dims_declared": self.dims_declared,
            "activations_gt": self.activations_gt,
            "dominant_dim": self.dominant_dim,
            "geo_id": self.geo_id,
            "source": self.source,
        }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# DatasetGenerator
# ---------------------------------------------------------------------------

class DatasetGenerator:
    """
    Genera el dataset de entrenamiento para Camino 1 (fine-tuning).

    Args:
        space_seed: Semilla del Space para reproducibilidad.
        llm_fn:     Función LLM opcional para enriquecer pares.
                    Si None, usa solo plantillas (modo sintético).
        seed:       Semilla aleatoria para reproducibilidad.
    """

    def __init__(
        self,
        space_seed: str = "DATASET_GEN_V1",
        llm_fn: Optional[Callable[[str], str]] = None,
        seed: int = 42,
    ) -> None:
        self.sp = Space(space_seed)
        self.llm_fn = llm_fn
        random.seed(seed)

    def _fill_template(self, template_text: str, template_dims: dict) -> tuple:
        """Rellena placeholders en texto y dims con vocabulario aleatorio."""
        # Construir mapa de sustituciones
        placeholders = set()
        for word in template_text.split():
            for c in ["{", "}"]:
                word = word.strip(c + ".,;:")
            if word.startswith("{") and word.endswith("}"):
                placeholders.add(word[1:-1])

        # También buscar en los valores de dims
        for dim_props in template_dims.values():
            for v in dim_props.values():
                if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                    placeholders.add(v[1:-1])

        # Generar sustituciones
        subs = {k: random.choice(VOCAB.get(k, [k])) for k in placeholders}

        # Aplicar al texto
        text = template_text
        for k, v in subs.items():
            text = text.replace("{" + k + "}", v)

        # Aplicar a dims
        dims = {}
        for dim, props in template_dims.items():
            dims[dim] = {
                pk: subs.get(pv[1:-1], pv) if isinstance(pv, str) and pv.startswith("{") else pv
                for pk, pv in props.items()
            }

        return text, dims

    def _make_object(self, dims: dict) -> ObjectND:
        """Crea ObjectND desde dims declaradas."""
        obj = ObjectND(self.sp)
        for dim, props in dims.items():
            w = round(random.uniform(0.6, 1.0), 2)
            obj.add(dim, props, w=w)
        return obj

    def _sample_from_template(self) -> DatasetSample:
        """Genera un sample desde plantilla sintética."""
        template_text, template_dims = random.choice(TEMPLATES)
        text, dims = self._fill_template(template_text, template_dims)
        obj = self._make_object(dims)
        hv = obj._hv()
        activations = {
            d: round(_proj(hv, self.sp.sub(d)), 4)
            for d in NATIVE if _proj(hv, self.sp.sub(d)) > UMBRAL
        }
        dom = max(activations, key=activations.get) if activations else None
        return DatasetSample(
            text=text,
            dims_declared=dims,
            activations_gt=activations,
            dominant_dim=dom,
            geo_id=obj.geo_id,
            source="template",
        )

    def _sample_with_llm(self, base_sample: DatasetSample) -> DatasetSample:
        """
        Enriquece un sample de plantilla con variación del LLM.

        El LLM reescribe el texto en lenguaje natural más variado,
        manteniendo la misma intención semántica.
        """
        if not self.llm_fn:
            return base_sample
        prompt = (
            f"Reescribe esta intención técnica en lenguaje natural más variado "
            f"(máx 1 oración, sin jerga innecesaria, distinto al original):\n\n"
            f"Original: {base_sample.text}\n\n"
            f"Solo devuelve la oración reescrita, sin explicaciones."
        )
        try:
            rewritten = self.llm_fn(prompt).strip()
            if rewritten and len(rewritten) > 10:
                return DatasetSample(
                    text=rewritten,
                    dims_declared=base_sample.dims_declared,
                    activations_gt=base_sample.activations_gt,
                    dominant_dim=base_sample.dominant_dim,
                    geo_id=base_sample.geo_id,
                    source="llm",
                )
        except Exception:
            pass
        return base_sample

    def generate(
        self,
        n: int = 1000,
        output_path: str = "polydim_dataset.jsonl",
        llm_ratio: float = 0.5,
        verbose: bool = True,
    ) -> List[DatasetSample]:
        """
        Genera n pares y los guarda en JSONL.

        Args:
            n:           Número de pares a generar.
            output_path: Archivo de salida JSONL.
            llm_ratio:   Fracción de pares enriquecidos con LLM (0-1).
                         Solo se aplica si llm_fn está disponible.
            verbose:     Mostrar progreso.

        Returns:
            Lista de DatasetSample generados.
        """
        samples = []
        n_llm = int(n * llm_ratio) if self.llm_fn else 0
        n_template = n - n_llm

        if verbose:
            print(f"Generando {n} pares: {n_template} plantilla + {n_llm} LLM")

        for i in range(n_template):
            samples.append(self._sample_from_template())
            if verbose and (i + 1) % 100 == 0:
                print(f"  {i+1}/{n} plantillas...")

        for i in range(n_llm):
            base = self._sample_from_template()
            samples.append(self._sample_with_llm(base))
            if verbose and (i + 1) % 50 == 0:
                print(f"  {i+1}/{n_llm} LLM...")

        random.shuffle(samples)

        with open(output_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(s.to_jsonl() + "\n")

        if verbose:
            dims_dist = {}
            for s in samples:
                if s.dominant_dim:
                    dims_dist[s.dominant_dim] = dims_dist.get(s.dominant_dim, 0) + 1
            print(f"\nDataset guardado: {output_path}")
            print(f"Total: {len(samples)} pares")
            print(f"Distribución: {dict(sorted(dims_dist.items(), key=lambda x: -x[1]))}")

        return samples

    def stats(self, samples: List[DatasetSample]) -> dict:
        """Estadísticas del dataset generado."""
        dims_dist = {}
        for s in samples:
            if s.dominant_dim:
                dims_dist[s.dominant_dim] = dims_dist.get(s.dominant_dim, 0) + 1
        avg_dims = sum(len(s.activations_gt) for s in samples) / len(samples) if samples else 0
        return {
            "total": len(samples),
            "dims_distribution": dims_dist,
            "avg_active_dims": round(avg_dims, 2),
            "sources": {
                "template": sum(1 for s in samples if s.source == "template"),
                "llm":      sum(1 for s in samples if s.source == "llm"),
            },
        }


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "DatasetGenerator",
    "DatasetSample",
    "TEMPLATES",
    "VOCAB",
]


if __name__ == "__main__":
    import tempfile
    import os
    gen = DatasetGenerator()
    out_dir = tempfile.gettempdir()
    out_path = os.path.join(out_dir, "polydim_dataset_test.jsonl")
    samples = gen.generate(n=100, output_path=out_path)
    print(f"\nStats: {gen.stats(samples)}")
    print("\nPrimeros 2 samples:")
    for s in samples[:2]:
        print(f"  '{s.text}' -> {s.dominant_dim} ({s.activations_gt})")
