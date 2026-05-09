# 📸 Visual Tour — Post-Training & Enterprise AI Portfolio

A picture-first walk through all 28 repositories. Each section shows the
key visualization(s) and a short note on what the project demonstrates.
For per-repo details (math, architecture, results), follow the repo links.

## How to read this page
**GIFs vs static figures.** Animated GIFs show *training dynamics* —
convergence, optimization trajectories, stage transitions. Static PNGs
show *final state* — ablation results, architecture comparisons,
calibration curves.

## Table of contents
- [Flagship: Three-Stage Post-Training](#flagship-three-stage-post-training)
- [Preference Alignment](#preference-alignment)
- [Reinforcement Learning](#reinforcement-learning)
- [SFT & Fine-Tuning](#sft--fine-tuning)
- [Synthetic Data & Data Quality](#synthetic-data--data-quality)
- [Agentic Systems & RAG](#agentic-systems--rag)
- [Inference & Edge Deployment](#inference--edge-deployment)
- [MLOps & Observability](#mlops--observability)
- [Optimization & Competitive Math](#optimization--competitive-math)
- [Multimodal](#multimodal)
- [Responsible AI](#responsible-ai)

---

## Flagship: Three-Stage Post-Training

### lfm-enterprise-text-posttraining

![three-stage-training](https://raw.githubusercontent.com/fikreab-s/lfm-enterprise-text-posttraining/main/viz/three_stage_training.gif)

Complete three-stage post-training pipeline: **SFT → DPO (β=5.0, length-normalized) → SLERP Merge (t=0.5)**. The animation shows progressive improvement across stages — instruction following jumps +30pp from SFT, win rate climbs +17pp from DPO, and SLERP merge recovers general capability without sacrificing domain gains.

![three-stage-ablation](https://raw.githubusercontent.com/fikreab-s/lfm-enterprise-text-posttraining/main/viz/three_stage_ablation.png)

The ablation chart: each stage contributes measurably. General capability (tinyBench) dips slightly during alignment but recovers with merge.

![beta-sweep](https://raw.githubusercontent.com/fikreab-s/lfm-enterprise-text-posttraining/main/viz/beta_sweep_heatmap.png)

β sensitivity heatmap. The highlighted column shows β=5.0 as the sweet spot — high win rate (72%), low length bias (8%), and strong domain accuracy (84%). Lower β values cause length exploitation; higher values degrade coherence.

![slerp-interpolation](https://raw.githubusercontent.com/fikreab-s/lfm-enterprise-text-posttraining/main/viz/slerp_interpolation.png)

SLERP interpolation parameter study. t=0.5 achieves the best blend: domain accuracy peaks at 87%, general capability recovers to 98%, and win rate stabilizes at 74%.

---

## Preference Alignment

### dpo-preference-alignment-lab

![dpo-training](https://raw.githubusercontent.com/fikreab-s/dpo-preference-alignment-lab/main/viz/dpo_training_dynamics.gif)

Side-by-side training dynamics of DPO, IPO, KTO, and ORPO. DPO (β=5.0) converges fastest and reaches the highest win rate (72%), while IPO shows the most stable loss trajectory.

![beta-sensitivity](https://raw.githubusercontent.com/fikreab-s/dpo-preference-alignment-lab/main/viz/beta_sensitivity.png)

β sensitivity with three metrics on one chart. The optimal β=5.0 balances win rate, coherence, and length bias — the length-normalized formulation is critical for reducing exploitation at high β.

![variant-radar](https://raw.githubusercontent.com/fikreab-s/dpo-preference-alignment-lab/main/viz/variant_comparison_radar.png)

Radar chart comparing DPO, IPO, KTO, and ORPO across six dimensions. Each variant has tradeoffs: ORPO needs no reference model, KTO works with unpaired data, DPO achieves highest win rate.

---

## Reinforcement Learning

### small-model-rl-verifier-loop

![grpo-training](https://raw.githubusercontent.com/fikreab-s/small-model-rl-verifier-loop/main/viz/grpo_training.gif)

GRPO training animation. Left: GSM8K accuracy climbing from 35% to 48%. Right: live sampling of a group — the group-relative advantage is computed from *these* samples, not a learned value network.

![grpo-vs-ppo](https://raw.githubusercontent.com/fikreab-s/small-model-rl-verifier-loop/main/viz/grpo_vs_ppo.png)

GRPO vs PPO: 13pp accuracy gain (vs 7pp for PPO) with 43% less VRAM. The key insight — eliminating the value network saves memory while group-relative advantages provide a better baseline signal.

![group-advantage](https://raw.githubusercontent.com/fikreab-s/small-model-rl-verifier-loop/main/viz/group_advantage.png)

The GRPO mechanism visualized: raw binary verifier rewards (left) are normalized to zero-mean group-relative advantages (right). No learned value function needed.

---

## SFT & Fine-Tuning

### pharma-brand-analytics-sft

![pharma-training](https://raw.githubusercontent.com/fikreab-s/pharma-brand-analytics-sft/main/viz/training.gif)

Domain-specific SFT on marketing analytics data. LoRA r=32 reaches 87% domain accuracy while training only 0.4% of parameters.

![lora-ablation](https://raw.githubusercontent.com/fikreab-s/pharma-brand-analytics-sft/main/viz/lora_ablation.png)

LoRA rank ablation: r=32 is the sweet spot — accuracy plateaus beyond this point while parameter count keeps growing linearly.

### lfm-sft-enterprise-text

![lfm-sft-training](https://raw.githubusercontent.com/fikreab-s/lfm-sft-enterprise-text/main/viz/training.gif)

Enterprise text SFT with LoRA, QLoRA, and Unsloth acceleration. Compares fine-tuning methods on the same domain data.

![lfm-sft-comparison](https://raw.githubusercontent.com/fikreab-s/lfm-sft-enterprise-text/main/viz/comparison.png)

Method comparison: Full fine-tuning achieves 89% domain accuracy but requires 9.6GB VRAM; QLoRA matches at 85% with only 2.8GB.

### exec-summary-generation-sft

![exec-summary-training](https://raw.githubusercontent.com/fikreab-s/exec-summary-generation-sft/main/viz/training.gif)

Executive summary generation SFT. ROUGE-L improves from 32 → 62 with SFT + DPO, and BERTScore reaches 86%.

### axolotl-domain-sft-dpo

![axolotl-comparison](https://raw.githubusercontent.com/fikreab-s/axolotl-domain-sft-dpo/main/viz/comparison.png)

Axolotl-based three-stage pipeline (SFT → DPO → Merge) with full ablation across instruction following, win rate, domain accuracy, and general capability.

### torchtune-efficient-recipes

![torchtune-comparison](https://raw.githubusercontent.com/fikreab-s/torchtune-efficient-recipes/main/viz/comparison.png)

TorchTune recipe comparison: Full FT, LoRA, QLoRA, and Knowledge Distillation (3B→1.2B). KD achieves 88% accuracy — nearly matching full fine-tuning while being a fundamentally different approach.

---

## Synthetic Data & Data Quality

### synth-data-distilabel-pipeline

![data-quality](https://raw.githubusercontent.com/fikreab-s/synth-data-distilabel-pipeline/main/viz/data_quality.png)

Evol-Instruct quality distribution (left) and iteration-over-iteration complexity/diversity gains (right). Three rounds of evolution increase complexity 3.75× and diversity 2.7×.

### data-quality-sft-preference-engine

![curation-impact](https://raw.githubusercontent.com/fikreab-s/data-quality-sft-preference-engine/main/viz/curation_impact.png)

The core finding: **500 curated examples > 5,000 unfiltered**. IFD + perplexity filtering reduces the dataset 10× while *increasing* accuracy from 69% → 87%.

### synth-persona-hcp-targeting

![synth-persona-training](https://raw.githubusercontent.com/fikreab-s/synth-persona-hcp-targeting/main/viz/training.gif)

Synthetic persona generation for healthcare professional targeting. Synthetic data matches real data quality (82% vs 85% realism) while achieving 95% privacy score vs 40% for real data.

---

## Agentic Systems & RAG

### mmm-causal-explainer-agent

![mmm-training](https://raw.githubusercontent.com/fikreab-s/mmm-causal-explainer-agent/main/viz/training.gif)

Agentic causal inference for marketing mix models. The agent learns to select the right causal method (DiD, Synthetic Control, BSTS, Do-Calculus) based on the query.

### text-to-sql-agent-finetuned

![sql-training](https://raw.githubusercontent.com/fikreab-s/text-to-sql-agent-finetuned/main/viz/training.gif)

Text-to-SQL agent with self-correction. Execution accuracy climbs from 38% (zero-shot) to 74% (SFT + self-correct) on Spider benchmark.

### rag-brand-analytics-pipeline

![rag-comparison](https://raw.githubusercontent.com/fikreab-s/rag-brand-analytics-pipeline/main/viz/comparison.png)

RAG pipeline ablation: naive RAG → re-ranking → fine-tuned retriever → citation validation. Faithfulness jumps from 72% to 92% with the full pipeline.

---

## Inference & Edge Deployment

### edge-deploy-small-model-demo

![pareto](https://raw.githubusercontent.com/fikreab-s/edge-deploy-small-model-demo/main/viz/pareto_frontier.png)

Quantization Pareto frontier: GGUF Q4_0 is the sweet spot — 3.4× smaller with <3% quality loss. Q2_K crosses the quality threshold.

![cost](https://raw.githubusercontent.com/fikreab-s/edge-deploy-small-model-demo/main/viz/cost_comparison.png)

The business case for edge: **99.7% cost reduction** ($3,000/mo → $8/mo) + 12× lower latency + full data privacy.

### vllm-sglang-inference-optimizer

![engine-comparison](https://raw.githubusercontent.com/fikreab-s/vllm-sglang-inference-optimizer/main/viz/engine_comparison.png)

Four inference engines compared: SGLang leads on decode throughput (92 tok/s) and batch processing (2,100 tok/s at batch=32). llama.cpp trades throughput for simplicity.

---

## MLOps & Observability

### enterprise-ml-pipeline-dataiku

![mlops](https://raw.githubusercontent.com/fikreab-s/enterprise-ml-pipeline-dataiku/main/viz/mlops_dashboard.png)

Production MLOps dashboard: PSI-based drift detection with automated retrain triggers (left) and pipeline stage duration breakdown (right). The monitor threshold (PSI=0.1) catches drift 40 days before model degradation.

### observability-llm-production

![monitoring](https://raw.githubusercontent.com/fikreab-s/observability-llm-production/main/viz/monitoring_dashboard.png)

LLM production monitoring: latency (p50/p99) with SLA tracking, and error rate with alerting. The simulated degradation at minute 70 shows how p99 latency spikes are detected before p50 is affected.

---

## Optimization & Competitive Math

### interpretable-budget-optimizer

![optimization-convergence](https://raw.githubusercontent.com/fikreab-s/interpretable-budget-optimizer/main/viz/optimization_convergence.gif)

SLSQP optimization convergence: the allocation shifts from equal-weight to optimal as the solver exploits Hill function saturation curves. Total effect plateaus as marginal ROI equalizes across channels (KKT condition).

![hill-curves](https://raw.githubusercontent.com/fikreab-s/interpretable-budget-optimizer/main/viz/hill_saturation_curves.png)

Hill saturation curves for six marketing channels. EC50 dots mark 50% effectiveness — channels like Email saturate fast (low EC50) while TV has a long tail (high EC50).

![optimal-allocation](https://raw.githubusercontent.com/fikreab-s/interpretable-budget-optimizer/main/viz/optimal_allocation.png)

Optimal allocation (left) and marginal ROI at optimum (right). At the solution, all active channels have equal marginal ROI = μ* (shadow price) — the KKT complementary slackness condition visualized.

![sensitivity](https://raw.githubusercontent.com/fikreab-s/interpretable-budget-optimizer/main/viz/sensitivity_analysis.png)

Budget sensitivity: stacked area shows how optimal allocation shifts as total budget grows (top), and shadow price curve shows diminishing returns (bottom).

### santa2025-optimization-engine

![score-progression](https://raw.githubusercontent.com/fikreab-s/santa2025-optimization-engine/main/viz/score_progression.png)

Recursive Ensemble Seeding score progression across 7 rounds. Each round starts from the best solutions of the previous round with mutations. Score climbs from 55.2 → 70.7, surpassing the competitor floor of 68.92 by round 5.

### aimo3-math-reasoning-pipeline

![math-reasoning](https://raw.githubusercontent.com/fikreab-s/aimo3-math-reasoning-pipeline/main/viz/math_reasoning.png)

Tool-Integrated Reasoning ablation. TIR (Jupyter execution) gives the biggest single jump (+16pp). Difficulty-aware allocation maintains accuracy while early consensus cuts compute by 40%.

---

## Multimodal

### multimodal-hybrid-orchestrator

![hybrid-comparison](https://raw.githubusercontent.com/fikreab-s/multimodal-hybrid-orchestrator/main/viz/comparison.png)

Hybrid architecture wins: 90% accuracy at $0.003/query and 85ms latency. 90% of queries handled by edge (65ms, near-free); only 10% escalated to cloud.

### lfm-audio-voice-agent

![voice-latency](https://raw.githubusercontent.com/fikreab-s/lfm-audio-voice-agent/main/viz/voice_latency.png)

End-to-end voice agent latency breakdown: 415ms total, under the 500ms SLA for conversational feel. TTS dominates (180ms) — sentence chunking enables streaming.

### llama-factory-vlm-multimodal

![vlm-comparison](https://raw.githubusercontent.com/fikreab-s/llama-factory-vlm-multimodal/main/viz/comparison.png)

VLM fine-tuning results: DocVQA jumps from 62% → 82%, chart interpretation from 45% → 76%. Full fine-tuning beats LoRA by ~4pp but costs 3× the VRAM.

### multimodal-eval-suite

![eval-comparison](https://raw.githubusercontent.com/fikreab-s/multimodal-eval-suite/main/viz/comparison.png)

Unified Multimodal Score (UMS) across text, vision, audio, and cross-modal benchmarks. The full multimodal model achieves UMS=82 — 14% above text-only baseline.

---

## Responsible AI

### safe-interpretable-commercial-ai

![safety](https://raw.githubusercontent.com/fikreab-s/safe-interpretable-commercial-ai/main/viz/safety_metrics.png)

Conformal prediction calibration curve (left) and safety metrics with/without the framework (right). The framework reduces harmful output rate from 3.2% → 0.1% while maintaining 91.2% coverage at α=0.10.

---

## Regenerating visualizations

Each repo with a `scripts/visualize.py` can regenerate its figures locally:

```bash
cd <repo-folder>
python scripts/visualize.py
```

Static PNGs and animated GIFs are committed under `viz/`. All visualizations use a consistent dark-mode theme for portfolio cohesion.

---

*28 repositories · SFT → DPO → GRPO → Model Merging → Edge Deployment · Building small models that punch above their weight.*
