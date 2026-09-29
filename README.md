<h1 align="center">
  Shifting Mechanisms:<br>
  How Positional Encoding Choice Shapes In-Context Retrieval
</h1>

This is the official repository for our work on *Shifting Mechanisms*. It builds on the code from [Mixing Mechanisms](https://github.com/yoavgur/mixing-mechs).

---

## Getting Started

### Dependencies
We use Python 3.12 on Linux with a CUDA 12 GPU, which the pinned flash-attn wheel requires. Steps 1–2 use `requirements.txt`. Steps 3–5 use `requirements-pretrain.txt`, which pins `transformers==4.57.6` for compatibility with the pre-training checkpoints.

#### 1. Layer selection
First, we select the retrieval layer for each model in our battery using the messiness 0/4 layer scan.

```bash
./experiments/layer_scan/run_layer_scan.sh
```

#### 2. Model battery
Then, we can run the model battery across all 29 tasks.

```bash
./experiments/model_battery/run_model_battery.sh
```

#### 3. Mechanism allocation of the pre-training checkpoints
To analyze the pre-training checkpoints, we run the following layer scan as well as the mechanism allocation result on the Boxes task at layer 15.

```bash
./experiments/pretrain/run_mech_allocation.sh
```

#### 4. Linear probes
We can then probe the ordering ID creation vs semantic key creation using our linear probing result.

```bash
./experiments/probe_oi/run_probe_oi.sh
```

#### 5. RULER and confusable keys
Finally, we evaluate the pre-training checkpoints behaviorally on RULER, and our adapted confusable key task.

```bash
./experiments/ruler/run_ruler.sh
```

For all experiments, results go to `results/` and `plots/` inside each experiment's folder.
