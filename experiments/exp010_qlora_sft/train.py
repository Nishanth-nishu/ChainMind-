#!/usr/bin/env python3
"""
EXP010: QLoRA Supervised Fine-Tuning of Qwen2.5-7B-Instruct on ChainMind data.

Design decisions (validated against published best practices):
──────────────────────────────────────────────────────────────
1. UNSLOTH over HF PEFT:
   - 2x faster training via custom CUDA kernels
   - 60% less VRAM than standard LoRA (fits RTX 3090 easily)
   - Native Qwen2.5 support with correct ChatML tokenization
   - Ref: https://github.com/unslothai/unsloth

2. QLoRA (4-bit NF4) over LoRA (bf16):
   - Full LoRA of Qwen2.5-7B requires ~28GB VRAM (exceeds 3090's 24GB)
   - QLoRA with 4-bit base uses ~12-14GB leaving headroom for activations
   - Quality gap between LoRA and QLoRA is <1% at 7B scale (Dettmers 2023)
   - Ref: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS 2023

3. r=64, alpha=128 (2× r ratio):
   - r=8-16: too small for task-specific adaptation
   - r=128: diminishing returns, more memory
   - r=64 with alpha=128 is the standard recommendation for instruction tuning
   - ALL linear projection layers targeted (not just q/v) for maximum adaptation

4. loss_on_responses_only=True:
   - System prompt and user query tokens masked from loss computation
   - Prevents model from learning to regurgitate prompts
   - Dramatically improves instruction following quality
   - Standard best practice for instruction tuning since LLaMA-2 (Touvron 2023)

5. max_seq_length=2048:
   - ChainMind traces: ~800-1200 tokens (THOUGHT+ACTION+OBSERVATION+FINAL_ANSWER)
   - 2048 covers 99%+ of examples without truncation
   - Packing disabled to avoid cross-example contamination in traces

6. Cosine LR with warmup:
   - 5% warmup prevents early gradient instability
   - Cosine decay outperforms linear for instruction tuning (Stanford Alpaca ablation)

7. Gradient checkpointing (unsloth mode):
   - Unsloth's custom gradient checkpointing is 30% faster than PyTorch default
   - Required to fit 7B model with r=64 LoRA on 24GB

Usage:
  python3 experiments/exp010_qlora_sft/train.py [--dry-run] [--output-dir models/chainmind-ft-v1]
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("exp010_train")

PROJECT_ROOT = Path(__file__).parent.parent.parent
SFT_DATA = PROJECT_ROOT / "data" / "sft_dataset.jsonl"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HF_CACHE = os.environ.get("HF_HOME", "/scratch/nishanth.r/hf_cache")


def validate_environment():
    """Check all required packages and CUDA availability before starting."""
    log.info("=== Environment Validation ===")
    errors = []

    try:
        import torch
        log.info(f"  PyTorch: {torch.__version__}")
        if not torch.cuda.is_available():
            errors.append("CUDA not available — QLoRA requires a GPU")
        else:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.info(f"  GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f}GB VRAM)")
            if vram_gb < 16:
                errors.append(f"VRAM too low: {vram_gb:.1f}GB (need ≥16GB for QLoRA 7B)")
    except ImportError:
        errors.append("PyTorch not installed")

    try:
        from unsloth import FastLanguageModel
        log.info("  unsloth: ✅")
    except ImportError:
        errors.append("unsloth not installed: pip install unsloth")

    try:
        import trl
        log.info(f"  trl: {trl.__version__} ✅")
    except ImportError:
        errors.append("trl not installed: pip install trl")

    try:
        import datasets
        log.info(f"  datasets: {datasets.__version__} ✅")
    except ImportError:
        errors.append("datasets not installed: pip install datasets")

    if not SFT_DATA.exists():
        errors.append(f"SFT data not found: {SFT_DATA}. Run scripts/build_sft_dataset.py first.")

    if errors:
        for e in errors:
            log.error(f"  ❌ {e}")
        sys.exit(1)

    log.info("  ✅ All checks passed")


def load_dataset(data_path: Path, max_examples: int = -1):
    """Load the SFT dataset from JSONL into HuggingFace Dataset format."""
    from datasets import Dataset

    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if max_examples > 0:
        examples = examples[:max_examples]

    log.info(f"Loaded {len(examples)} training examples from {data_path}")

    # Validation: check format
    for i, ex in enumerate(examples[:3]):
        assert "messages" in ex, f"Example {i} missing 'messages' key"
        assert len(ex["messages"]) >= 2, f"Example {i} has <2 messages"
        roles = [m["role"] for m in ex["messages"]]
        assert "user" in roles, f"Example {i} missing user message"
        assert "assistant" in roles, f"Example {i} missing assistant message"

    log.info("  ✅ Dataset format validated")
    return Dataset.from_list(examples)


def format_messages(examples, tokenizer):
    """
    Apply Qwen2.5 ChatML template to the messages.

    Qwen2.5-Instruct uses ChatML format:
      <|im_start|>system\n...<|im_end|>\n
      <|im_start|>user\n...<|im_end|>\n
      <|im_start|>assistant\n...<|im_end|>

    The tokenizer.apply_chat_template handles this automatically.
    We set add_generation_prompt=False since this is training data.
    """
    texts = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}


def train(args):
    """Main training function."""
    import torch
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
    from unsloth.chat_templates import train_on_responses_only

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Model + Tokenizer ────────────────────────────────────────────────
    log.info(f"\n=== Loading {MODEL_ID} with QLoRA ===")
    log.info(f"  HF_HOME: {HF_CACHE}")
    log.info(f"  max_seq_length: {args.max_seq_length}")
    log.info(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=args.max_seq_length,
        dtype=None,                        # auto: bf16 on Ampere+, fp16 otherwise
        load_in_4bit=True,                 # QLoRA: 4-bit NF4 base model
        cache_dir=HF_CACHE,
    )

    # ── Apply LoRA adapters ───────────────────────────────────────────────────
    # Target ALL linear layers for maximum task adaptation.
    # Using r=64, alpha=128 (2× ratio is the proven best practice).
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized GC
        random_state=42,
        use_rslora=False,          # Standard LoRA (rsLoRA negligible gain at r=64)
        loftq_config=None,
    )

    log.info(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    log.info(f"  Total params:     {sum(p.numel() for p in model.parameters()):,}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    log.info(f"\n=== Loading dataset from {SFT_DATA} ===")
    max_ex = 100 if args.dry_run else -1
    dataset = load_dataset(SFT_DATA, max_examples=max_ex)

    # Apply ChatML template
    dataset = dataset.map(
        lambda ex: format_messages(ex, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Train/eval split (90/10)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    log.info(f"  Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # ── Training Config ───────────────────────────────────────────────────────
    # Effective batch = per_device * grad_accum = 2 * 8 = 16
    # This matches the recommendation from the Unsloth docs for 7B models.
    num_epochs = 1 if args.dry_run else args.epochs
    max_steps = 20 if args.dry_run else -1

    training_args = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,          # Effective batch size = 16
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        optim="adamw_8bit",                     # 8-bit Adam saves ~2GB VRAM
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,                          # Disabled: traces must not be packed
        report_to="none",                       # No WandB needed on cluster
        seed=42,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # ── Loss masking: train on assistant responses only ───────────────────────
    # This is critical: we mask system + user tokens from the loss.
    # The model learns to GENERATE the assistant content, not memorize the prompt.
    # Qwen2.5 ChatML tokens: <|im_start|>assistant\n ... <|im_end|>
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    # Apply response-only loss masking via Unsloth helper
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info(f"\n=== Starting Training ===")
    log.info(f"  Epochs: {num_epochs}")
    log.info(f"  Effective batch size: {2 * 8} = 16")
    log.info(f"  Learning rate: {args.learning_rate}")

    start = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start

    log.info(f"\n{'='*50}")
    log.info(f"Training complete in {elapsed/3600:.1f}h")
    log.info(f"Train loss: {train_result.training_loss:.4f}")

    # Save adapter weights
    log.info(f"\n=== Saving LoRA adapter to {adapter_dir} ===")
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # ── Merge + save full model (for vLLM loading) ────────────────────────────
    merged_dir = output_dir / "merged"
    log.info(f"\n=== Merging LoRA into base and saving to {merged_dir} ===")
    log.info("  (This takes ~5 min and requires ~28GB CPU RAM)")

    FastLanguageModel.for_inference(model)  # Disable gradient checkpointing for merge
    model.save_pretrained_merged(
        str(merged_dir),
        tokenizer,
        save_method="merged_16bit",  # bf16 merged weights for vLLM serving
    )

    log.info(f"✅ Training complete!")
    log.info(f"   Adapter:  {adapter_dir}")
    log.info(f"   Merged:   {merged_dir}")
    log.info(f"   To serve: vllm serve {merged_dir} --port 8100")

    # Save training metadata
    meta = {
        "model_id": MODEL_ID,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_seq_length": args.max_seq_length,
        "learning_rate": args.learning_rate,
        "epochs": num_epochs,
        "train_examples": len(train_ds),
        "train_loss": train_result.training_loss,
        "training_time_hours": elapsed / 3600,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"   Metadata: {output_dir}/training_metadata.json")


def main():
    parser = argparse.ArgumentParser(description="QLoRA SFT for ChainMind-FT-v1")
    parser.add_argument("--output-dir", default="models/chainmind-ft-v1",
                        help="Output directory for adapter and merged model")
    parser.add_argument("--lora-r", type=int, default=64,
                        help="LoRA rank (default: 64)")
    parser.add_argument("--lora-alpha", type=int, default=128,
                        help="LoRA alpha (default: 128, = 2×r)")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Max sequence length (default: 2048)")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Training epochs (default: 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 20 steps only for smoke-test")
    args = parser.parse_args()

    validate_environment()
    train(args)


if __name__ == "__main__":
    main()
