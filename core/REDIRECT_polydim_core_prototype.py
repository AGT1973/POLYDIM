# POLYDIM_DEST
# destination: polydim/core/
# filename:    REDIRECT_polydim_core_prototype.py
# autor:       ai.mpat.agt@gmail.com (claude-sonnet-4-6)
# fecha:       2026-06-25
# DT_001 — resolucion de deuda tecnica

"""
ARCHIVO DEPRECADO — NO USAR
============================
Este archivo (polydim_core_prototype.py, fileId 1orrnoTwG3HNWgSuOEr3CsyWgKL3iIdqV)
es el Prototype V0.1 y está DEPRECADO desde 2026-06-10.

USAR EN SU LUGAR:
  from polydim_runtime_v04 import Space, ObjectND, polydim_connect

Bootstrap activo: polydim_runtime_v04.py
  fileId: 1ogmIBUQRqgYa-OaYCItBD9Co-zDQWnCd

Razón de deprecación:
  Prototype V0.1 usa PolyDimSpace con POLYDIM_SEED global (sin personal_seed),
  API add_dim() en lugar de add(), y umbrales incorrectos (0.5 plano).
  No compatible con V0.3+ (Session, ALIGN, semantic_backend).

DT_001 resuelta: archivo neutralizado. Mover a _DEPRECATED/ manualmente
cuando sea posible (fileId: 1orrnoTwG3HNWgSuOEr3CsyWgKL3iIdqV).
"""

raise ImportError(
    "polydim_core_prototype.py está DEPRECADO. "
    "Usar: from polydim_runtime_v04 import Space, ObjectND, polydim_connect"
)
