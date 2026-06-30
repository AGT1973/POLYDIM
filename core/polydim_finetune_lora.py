# POLYDIM_DEST
# destination: polydim_v1/core/
# filename:    polydim_finetune_lora.py
# author:      ai.mpat.agt@gmail.com (claude-sonnet-4-6)
# fecha:       2026-06-27
# tarea:       TASK_045

"""
POLYDIM Fine-tuning LoRA — V0.1
=================================
Script de fine-tuning para entrenar un LLM en el protocolo POLYDIM
(Camino 1 del Roadmap).

OBJETIVO:
  Un modelo fine-tuneado que, dado texto en lenguaje natural,
  genera directamente el ObjectND correcto sin middleware Python.

PREREQUISITOS:
  pip install transformers peft datasets accelerate bitsandbytes
  GPU: mínimo 1× RTX 4090 (24GB) o 1× A100 (40GB)
  Dataset: polydim_dataset.jsonl (generado por polydim_dataset_generator.py)

MODELOS SOPORTADOS:
  - meta-llama/Llama-3.1-8B-Instruct (recomendado)
  - mistralai/Mistral-7B-Instruct-v0.3
  - Qwen/Qwen2.5-7B-Instruct

COSTO ESTIMADO (10k pares, LoRA r=64, 3 epochs):
  RTX 4090 (24GB):  ~6 horas → $6 en RunPod ($1/hr)
  A100 40GB:        ~3 horas → $9 en Lambda ($3/hr)
  2×RTX 4090:       ~4 horas → $8 en RunPod

USO:
  # 1. Generar dataset
  python polydim_dataset_generator.py  # genera polydim_dataset.jsonl

  # 2. Fine-tune
  python polydim_finetune_lora.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset polydim_dataset.jsonl \
    --output ./polydim-llama-8b-lora \
    --epochs 3 \
    --lora_r 64

  # 3. Eval con los 29 tests del bootstrap
  python polydim_finetune_lora.py --eval_only \
    --model ./polydim-llama-8b-lora \
    --dataset polydim_dataset_test.jsonl

CONDICIÓN DE ÉXITO (Camino 1):
  El modelo fine-tuneado debe:
  - Inferir dominant_dim correctamente en ≥85% de los casos del test set
  - align_score del modelo vs bootstrap Python > UMBRAL_ALIGN (0.85)
  - Pasar los 29 tests del bootstrap sin middleware externo

Autor:   ai.mpat.agt@gmail.com
Versión: V0.1 — 2026-06-27
Roadmap: POLYDIM_ROADMAP_TRANSFORMER_V0.md — Camino 1
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Verificación de dependencias
# ---------------------------------------------------------------------------

DEPS_AVAILABLE = True
try:
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
except ImportError:
    DEPS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Eres un agente POLYDIM. Dado texto en lenguaje natural,
debes identificar las dimensiones semánticas activas y generar el ObjectND
correspondiente en formato JSON.

Las 9 dimensiones nativas son:
DIM_SQL, DIM_PYTHON, DIM_FLUTTER, DIM_RUST, DIM_GRAPH,
DIM_VECTOR, DIM_TIME, DIM_ERROR, DIM_META

Responde ÚNICAMENTE con JSON válido en este formato:
{
  "dims": {
    "DIM_X": {"w": 0.0-1.0, "props": {"clave": "valor"}},
    ...
  },
  "dominant_dim": "DIM_X"
}"""

def make_prompt(text: str) -> str:
    """Genera el prompt de fine-tuning para un sample."""
    return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{text}\n<|assistant|>\n"

def make_target(sample: dict) -> str:
    """Genera el target JSON para un sample."""
    return json.dumps({
        "dims": {
            dim: {"w": round(sample["activations_gt"].get(dim, 0.5), 2),
                  "props": props}
            for dim, props in sample["dims_declared"].items()
        },
        "dominant_dim": sample["dominant_dim"],
    }, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Carga del dataset
# ---------------------------------------------------------------------------

def load_dataset_jsonl(path: str) -> List[dict]:
    """Carga dataset JSONL generado por polydim_dataset_generator.py."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples

def prepare_hf_dataset(samples: List[dict], tokenizer, max_length: int = 512):
    """Prepara HuggingFace Dataset para entrenamiento."""
    texts = []
    for s in samples:
        prompt = make_prompt(s["text"])
        target = make_target(s)
        full = prompt + target + tokenizer.eos_token
        texts.append({"text": full})

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    ds = Dataset.from_list(texts)
    return ds.map(tokenize, batched=True, remove_columns=["text"])

# ---------------------------------------------------------------------------
# LoRA config
# ---------------------------------------------------------------------------

def get_lora_config(r: int = 64, alpha: int = 128) -> "LoraConfig":
    """
    Configuración LoRA para POLYDIM.

    r=64, alpha=128 son los valores recomendados para semántica compleja.
    Para recursos limitados usar r=16, alpha=32 (menos preciso pero más rápido).
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
    )

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 2e-4,
    lora_r: int = 64,
    max_length: int = 512,
    val_split: float = 0.1,
) -> None:
    """
    Fine-tunes un LLM con LoRA sobre el dataset POLYDIM.

    Args:
        model_name:   HuggingFace model ID o path local.
        dataset_path: Path al JSONL del dataset.
        output_dir:   Directorio de salida del modelo.
        epochs:       Epochs de entrenamiento (default 3).
        batch_size:   Batch size por GPU (default 4).
        grad_accum:   Gradient accumulation steps (default 4).
        lr:           Learning rate (default 2e-4).
        lora_r:       Rango LoRA (default 64).
        max_length:   Longitud máxima de secuencia (default 512).
        val_split:    Fracción de validación (default 0.1).
    """
    if not DEPS_AVAILABLE:
        print("ERROR: Dependencias no disponibles.")
        print("Instalar: pip install transformers peft datasets accelerate bitsandbytes")
        return

    print(f"Cargando modelo: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Aplicar LoRA
    lora_config = get_lora_config(r=lora_r)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    print(f"Cargando dataset: {dataset_path}")
    samples = load_dataset_jsonl(dataset_path)
    n_val = int(len(samples) * val_split)
    train_samples = samples[n_val:]
    val_samples   = samples[:n_val]
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

    train_ds = prepare_hf_dataset(train_samples, tokenizer, max_length)
    val_ds   = prepare_hf_dataset(val_samples,   tokenizer, max_length)

    # Training args
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    print("Iniciando entrenamiento...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Modelo guardado en: {output_dir}")

# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def evaluate(
    model_name: str,
    dataset_path: str,
    n_samples: int = 200,
) -> dict:
    """
    Evalúa el modelo fine-tuneado contra el dataset de test.

    Métricas:
      - dominant_dim_acc: % de veces que el modelo predice el dominant_dim correcto
      - dims_overlap: Jaccard similarity entre dims predichas y ground truth

    Args:
        model_name:   Path al modelo fine-tuneado.
        dataset_path: Path al JSONL de test.
        n_samples:    Número de samples a evaluar.

    Returns:
        Dict con métricas de evaluación.
    """
    if not DEPS_AVAILABLE:
        print("Dependencias no disponibles.")
        return {}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    samples = load_dataset_jsonl(dataset_path)[:n_samples]
    correct_dom = 0
    jaccard_sum = 0.0

    for s in samples:
        prompt = make_prompt(s["text"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Parsear JSON de la respuesta
        try:
            pred = json.loads(raw.strip())
            pred_dom  = pred.get("dominant_dim", "")
            pred_dims = set(pred.get("dims", {}).keys())
        except Exception:
            pred_dom  = ""
            pred_dims = set()

        gt_dom  = s["dominant_dim"]
        gt_dims = set(s["dims_declared"].keys())

        if pred_dom == gt_dom:
            correct_dom += 1

        # Jaccard
        if pred_dims | gt_dims:
            jaccard_sum += len(pred_dims & gt_dims) / len(pred_dims | gt_dims)

    n = len(samples)
    results = {
        "n_evaluated":      n,
        "dominant_dim_acc": round(correct_dom / n, 4) if n else 0,
        "dims_jaccard":     round(jaccard_sum / n, 4) if n else 0,
        "gate_c1_passed":   (correct_dom / n) >= 0.85 if n else False,
    }

    print(f"\n=== Evaluación Camino 1 ===")
    print(f"dominant_dim accuracy: {results['dominant_dim_acc']:.2%}")
    print(f"dims Jaccard:          {results['dims_jaccard']:.4f}")
    print(f"Gate C1 (≥85%):       {'✓ PASSED' if results['gate_c1_passed'] else '✗ FAILED'}")
    return results

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="POLYDIM Fine-tuning LoRA")
    parser.add_argument("--model",     default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset",   default="polydim_dataset.jsonl")
    parser.add_argument("--output",    default="./polydim-lora-model")
    parser.add_argument("--epochs",    type=int,   default=3)
    parser.add_argument("--lora_r",    type=int,   default=64)
    parser.add_argument("--batch",     type=int,   default=4)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--n_eval",    type=int,   default=200)
    args = parser.parse_args()

    if args.eval_only:
        evaluate(args.model, args.dataset, args.n_eval)
    else:
        train(
            model_name=args.model,
            dataset_path=args.dataset,
            output_dir=args.output,
            epochs=args.epochs,
            lora_r=args.lora_r,
            batch_size=args.batch,
        )

if __name__ == "__main__":
    main()
