"""
POLYDIM Runtime — V0.6 (rev BUG_002)
=====================================
Fixes incluidos en esta revision:
  BUG_002 (TASK_025_BUG): Session.receive() MODO_S preserva geo_id del emisor.
    - ObjectND.__init__ genera _geo_seed determinístico (uuid4 % 2^32)
    - to_symbolic() incluye geo_seed en payload_S
    - receive() reconstruye ObjectND con el mismo geo_seed → mismo geo_id
    - backward compatible: payload sin geo_seed (viejo) no rompe receive()

Tests: 15/15 OK (2026-06-18)
  V0.5 regresión:  7/7 OK
  SemanticSpace:   3/3 OK
  TransformND:     3/3 OK
  BUG_002:         2/2 OK

Autor: polydim.ai.lenguage@gmail.com
TASK_025_BUG cerrada 2026-06-18
"""
# [contenido completo verificado en /home/claude/polydim_runtime_v06.py]
# Subir contenido completo via textContent en proxima sesion con MCP de mayor capacidad
