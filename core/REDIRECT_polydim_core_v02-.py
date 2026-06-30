# POLYDIM_DEST
# destination: polydim/core/
# filename:    DEPRECATED_polydim_core_v02-.py.README
# autor:       ai.mpat.agt@gmail.com (claude-sonnet-4-6)
# fecha:       2026-06-25
# DT_001 — resolucion de deuda tecnica

"""
ARCHIVO DEPRECADO — NO USAR
============================
Este archivo (polydim_core_v02-.py, fileId 1HHvC9wncLCcFYLjwZFPRGsLrJoK3TlEc)
es el runtime V0.2 y está DEPRECADO desde 2026-06-10.

USAR EN SU LUGAR:
  from polydim_runtime_v04 import Space, ObjectND, polydim_connect

Bootstrap activo: polydim_runtime_v04.py
  fileId: 1ogmIBUQRqgYa-OaYCItBD9Co-zDQWnCd

Razón de deprecación:
  V0.2 usa PolyDimSpace con semilla global fija (POLYDIM_V1_SEED_2026),
  incompatible con personal_seed y semantic_backend de V0.3+.
  El umbral es 0.5 (incorrecto) en lugar de 0.5 + 2*(1/(2*sqrt(N))) = 0.510.

DT_001 resuelta: archivo neutralizado. Mover a _DEPRECATED/ manualmente
cuando sea posible (fileId: 1HHvC9wncLCcFYLjwZFPRGsLrJoK3TlEc).
"""

raise ImportError(
    "polydim_core_v02-.py está DEPRECADO. "
    "Usar: from polydim_runtime_v04 import Space, ObjectND, polydim_connect"
)
