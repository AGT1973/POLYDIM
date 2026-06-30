# POLYDIM

> **The assembly language of the transformer era.**

A formally-specified programming language whose fundamental unit is a geometric
transformation **T: R^N → R^N**, not a Von Neumann instruction.

[![ORCID](https://img.shields.io/badge/ORCID-0009--0001--2787--6067-green)](https://orcid.org/0009-0001-2787-6067)
[![Tests](https://img.shields.io/badge/tests-29%2F29-brightgreen)](./core/polydim_tests.py)
[![License](https://img.shields.io/badge/paper-CC%20BY%204.0-blue)](LICENSE)

---

## The Problem

Modern AI architectures operate in continuous high-dimensional geometric spaces
(R^N, N >= 10,000), yet are forced to communicate via sequential text, losing
**over 99.9% of semantic information** at every serialization boundary.

> Interlat [Du et al., 2025]: a hidden state carries ~40,000 bits vs ~15 bits per token.
> That is a **2,667x semantic bandwidth gap**.

## The Language

POLYDIM has four algebraic primitives:

| Primitive | Replaces |
|---|---|
| `COMPOSE(T1, T2)` | Sequential execution |
| `MIX(a, T1, b, T2)` | if/else branching |
| `FIXPOINT(T, e)` | for/while loops |
| `PROJECT(T, E)` | Type casting / compilation |

## Formal Foundations

Five theorems proved unconditionally (T1-T5): Associativity, Linearity, Functor
(PROJECT for COMPILE/RENDER/EXPORT), Banach fixed point, GEO_ID invariance.

## Repository Structure

```
polydim/
├── core/        # Python bootstrap (29/29 tests) + Rust VM V0.1
├── spec/        # Formal specifications (binary format, semantics, ALIGN protocol)
├── POLYDIM_SOTA_UNIFIED_V1.md   # Full paper (Markdown)
├── POLYDIM_SOTA_UNIFIED_V1.docx # Full paper (Word)
└── polydim.bib  # BibTeX references
```

## Results

| Metric | Value |
|---|---|
| Bootstrap tests | 29/29 |
| Align score | 0.9993 |
| Semantic gain vs baseline | +64.5% |
| LoRA compression | 99.4% |
| Theorems proved | 5 (unconditional) |

## Citation

```bibtex
@misc{garciatraba2026polydim,
  title  = {{POLYDIM}: A Transformer-Native Algebraic Programming Language},
  author = {Garcia Traba, Ariel H.},
  orcid  = {0009-0001-2787-6067},
  year   = {2026},
  note   = {Preprint. cs.PL + cs.AI. Independent researcher, Buenos Aires, Argentina}
}
```

## Author

**Ariel H. Garcia Traba** - Independent Researcher - Buenos Aires, Argentina
- Email: ariel.garcia.traba@gmail.com
- ORCID: [0009-0001-2787-6067](https://orcid.org/0009-0001-2787-6067)

## License

- Paper: CC BY 4.0 - Code: MIT
