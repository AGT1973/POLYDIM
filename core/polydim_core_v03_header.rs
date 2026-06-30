// POLYDIM_DEST
// destination: polydim/core/
// filename:    polydim_core.rs
// author:      ai.mpat.agt@gmail.com
// version:     V0.3 — 2026-06-18
// tasks:       TASK_023, TASK_024, TASK_026 (BUG_006)
//
// Cambios V0.3 (TASK_026 BUG_006):
//   - Space.n: usize — campo público que refleja N efectivo
//   - pbp::decode_hv_dyn() — decode dinámico usando N del header PBP
//   - 3 tests nuevos: decode_hv_dyn_reads_n, space_n_field, decode_hv_dyn_bad_magic
//
// Ver polydim_core_v02 en _DEPRECATED/ para historial anterior.
