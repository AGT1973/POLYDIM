# POLYDIM_DEST
# destination: polydim_v1/core/
# filename:    polydim_colab_t4.py
# author:      ai.mpat.agt@gmail.com (claude-sonnet-4-6)
# fecha:       2026-06-27
# tarea:       TASK_046 — variante Google Colab T4

"""
POLYDIM Fine-tuning — Google Colab T4
======================================
Variante de polydim_finetune_lora.py optimizada para T4 (15GB VRAM).

EJECUTAR EN COLAB:
  1. Abrir nuevo notebook en colab.research.google.com
  2. Runtime → Change runtime type → T4 GPU
  3. Pegar y ejecutar celda por celda

DIFERENCIAS vs polydim_finetune_lora.py:
  - Cuantización 4-bit (load_in_4bit=True) → modelo entra en 15GB
  - Modelo: Llama 3.2 3B (recomendado) o Mistral 7B con 4-bit
  - Batch size reducido (2 en lugar de 4)
  - Checkpoints automáticos en Google Drive
  - Resume automático si Colab desconecta

TIEMPO ESTIMADO T4:
  Dataset generación:  ~2 min
  Fine-tuning 3B/3ep:  ~4-6 horas  (Colab gratuito puede cortar a 4h)
  Fine-tuning 7B/3ep:  ~8-10 horas (requiere Colab Pro o checkpoints)
  Evaluación:          ~10 min

COSTO:
  Colab gratuito:  $0 (pero con límite de tiempo)
  Colab Pro:       $10/mes (sin límite de tiempo)
  Colab Pro+:      $50/mes (GPU garantizada + más VRAM)

Autor:   ai.mpat.agt@gmail.com
Versión: V0.1 — 2026-06-27
"""

# =============================================================================
# CELDA 1 — Instalación
# =============================================================================
CELDA_1 = """
!pip install -q transformers peft datasets accelerate bitsandbytes
!pip install -q numpy

# Clonar/copiar los archivos del runtime POLYDIM desde Drive
# (asumiendo que Drive está montado)
import os
if not os.path.exists('polydim_runtime_v04.py'):
    # Descargar desde Drive usando el fileId
    !gdown 1ogmIBUQRqgYa-OaYCItBD9Co-zDQWnCd -O polydim_runtime_v04.py
    !gdown 1YKSnR9hb-IPIJ2Bs9KjlBEk7LM3vHq8C -O polydim_weighted_inference.py
    !gdown 15m5Sov5tyrlwXKJ6EoP-zqJ0MDjl3WmJ -O polydim_dataset_generator.py
    !gdown 1B2I4MZdbuUL45eCuitbOsoEejgcRNXOG -O polydim_finetune_lora.py
print("✓ Instalación completa")
"""

# =============================================================================
# CELDA 2 — Montar Drive y verificar GPU
# =============================================================================
CELDA_2 = """
from google.colab import drive
drive.mount('/content/drive')

import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

OUTPUT_DIR = "/content/drive/MyDrive/polydim_camino1"
import os; os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Checkpoints: {OUTPUT_DIR}")
"""

# =============================================================================
# CELDA 3 — Generar dataset
# =============================================================================
CELDA_3 = """
from polydim_dataset_generator import DatasetGenerator

gen = DatasetGenerator(seed=42)
samples = gen.generate(
    n=5000,              # 5k pares para T4 (10k si hay tiempo)
    output_path="polydim_dataset.jsonl",
    verbose=True,
)
print(f"\\nStats: {gen.stats(samples)}")
"""

# =============================================================================
# CELDA 4 — Fine-tuning con cuantización 4-bit para T4
# =============================================================================
CELDA_4 = """
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import json
from datasets import Dataset

# ── Configuración ──────────────────────────────────────────────────────────
MODEL_NAME  = "meta-llama/Llama-3.2-3B-Instruct"  # 3B cabe bien en T4
# MODEL_NAME  = "mistralai/Mistral-7B-Instruct-v0.3"  # alternativa 7B 4-bit
LORA_R      = 32          # r=32 para T4 (r=64 requiere más VRAM)
LORA_ALPHA  = 64
EPOCHS      = 3
BATCH_SIZE  = 2           # batch pequeño para T4
GRAD_ACCUM  = 8           # equivale a batch efectivo de 16
LR          = 2e-4
MAX_LEN     = 384         # reducido para T4
CHECKPOINT  = OUTPUT_DIR  # directorio en Drive para checkpoints

# ── Cargar tokenizer ───────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Cargar modelo con cuantización 4-bit ───────────────────────────────────
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# ── LoRA ───────────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    inference_mode=False,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Dataset ────────────────────────────────────────────────────────────────
SYSTEM = (
    "Eres un agente POLYDIM. Dado texto, identifica dimensiones semánticas "
    "activas y genera el ObjectND en JSON.\\n"
    "Dimensiones: DIM_SQL, DIM_PYTHON, DIM_FLUTTER, DIM_RUST, DIM_GRAPH, "
    "DIM_VECTOR, DIM_TIME, DIM_ERROR, DIM_META\\n"
    "Responde SOLO con JSON: {\\\"dims\\\": {...}, \\\"dominant_dim\\\": \\\"DIM_X\\\"}"
)

def format_sample(s):
    prompt = f"<|system|>\\n{SYSTEM}\\n<|user|>\\n{s['text']}\\n<|assistant|>\\n"
    target = json.dumps({
        "dims": {
            d: {"w": round(s["activations_gt"].get(d, 0.5), 2), "props": p}
            for d, p in s["dims_declared"].items()
        },
        "dominant_dim": s["dominant_dim"],
    })
    return prompt + target + tokenizer.eos_token

samples = []
with open("polydim_dataset.jsonl") as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

n_val = int(len(samples) * 0.1)
texts_train = [format_sample(s) for s in samples[n_val:]]
texts_val   = [format_sample(s) for s in samples[:n_val]]

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN, padding=False)

ds_train = Dataset.from_list([{"text": t} for t in texts_train]).map(tokenize, batched=True, remove_columns=["text"])
ds_val   = Dataset.from_list([{"text": t} for t in texts_val]).map(tokenize, batched=True, remove_columns=["text"])
print(f"Train: {len(ds_train)}  Val: {len(ds_val)}")

# ── Training args ──────────────────────────────────────────────────────────
# Detectar si hay checkpoint previo para resume automático
import glob
checkpoints = glob.glob(f"{CHECKPOINT}/checkpoint-*")
resume_from = sorted(checkpoints)[-1] if checkpoints else None
if resume_from:
    print(f"Resumiendo desde: {resume_from}")

args = TrainingArguments(
    output_dir=CHECKPOINT,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    fp16=True,
    logging_steps=20,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,           # guardar frecuentemente en Drive
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    report_to="none",
    load_best_model_at_end=True,
    optim="paged_adamw_8bit",  # optimizador eficiente para 4-bit
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# ── Entrenar ───────────────────────────────────────────────────────────────
trainer.train(resume_from_checkpoint=resume_from)
trainer.save_model(CHECKPOINT + "/final")
tokenizer.save_pretrained(CHECKPOINT + "/final")
print(f"\\n✓ Modelo guardado en Drive: {CHECKPOINT}/final")
"""

# =============================================================================
# CELDA 5 — Evaluación
# =============================================================================
CELDA_5 = """
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_BASE = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_LORA = OUTPUT_DIR + "/final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_BASE, torch_dtype=torch.float16, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, MODEL_LORA)
model.eval()

# Cargar test set (primeras 100 líneas)
samples = []
with open("polydim_dataset.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 100: break
        if line.strip():
            samples.append(json.loads(line))

SYSTEM = (
    "Eres un agente POLYDIM. Dado texto, identifica dimensiones semánticas "
    "activas y genera el ObjectND en JSON.\\n"
    "Responde SOLO con JSON: {\\\"dims\\\": {...}, \\\"dominant_dim\\\": \\\"DIM_X\\\"}"
)

correct = 0
jaccard_sum = 0.0

for s in samples:
    prompt = f"<|system|>\\n{SYSTEM}\\n<|user|>\\n{s['text']}\\n<|assistant|>\\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, temperature=0.1, do_sample=False)
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        pred = json.loads(raw.strip())
        pred_dom  = pred.get("dominant_dim", "")
        pred_dims = set(pred.get("dims", {}).keys())
    except:
        pred_dom  = ""
        pred_dims = set()

    gt_dom  = s["dominant_dim"]
    gt_dims = set(s["dims_declared"].keys())

    if pred_dom == gt_dom: correct += 1
    if pred_dims | gt_dims:
        jaccard_sum += len(pred_dims & gt_dims) / len(pred_dims | gt_dims)

n = len(samples)
acc   = correct / n
jacc  = jaccard_sum / n
print(f"\\n=== Evaluación Camino 1 (T4) ===")
print(f"dominant_dim accuracy: {acc:.2%}")
print(f"dims Jaccard:          {jacc:.4f}")
print(f"Gate C1 (≥85%):       {'✓ PASSED' if acc >= 0.85 else '✗ FAILED — más epochs o más datos'}")
"""

# =============================================================================
# Imprimir instrucciones
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("POLYDIM Camino 1 — Google Colab T4")
    print("=" * 60)
    print()
    print("Instrucciones:")
    print("1. Abrir colab.research.google.com → nuevo notebook")
    print("2. Runtime → Change runtime type → T4 GPU")
    print("3. Crear 5 celdas con el contenido de CELDA_1 a CELDA_5")
    print("4. Ejecutar en orden")
    print()
    print("Modelo recomendado: meta-llama/Llama-3.2-3B-Instruct")
    print("  (requiere aceptar licencia en huggingface.co/meta-llama)")
    print()
    print("Alternativa sin licencia: mistralai/Mistral-7B-Instruct-v0.3")
    print()
    print("Tiempo estimado T4 gratuito:")
    print("  3B model, 5k pares, 3 epochs: ~3-4 horas")
    print("  7B model, 5k pares, 3 epochs: ~6-8 horas (necesita Pro)")
    print()
    print("Celdas:")
    for i, celda in enumerate([CELDA_1, CELDA_2, CELDA_3, CELDA_4, CELDA_5], 1):
        print(f"\n{'='*50}")
        print(f"CELDA {i}:")
        print(celda.strip())
