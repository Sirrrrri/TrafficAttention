# TrafficAttention

Source code for the paper submission. This repository implements the TrafficAttention for network traffic classification.

## Project Structure

```
TrafficAttention/
├── data_process/
│   ├── Traffic Splitting/          # Split raw PCAP files into per-flow PCAPs
│   │   ├── split_flows.py          # Flow splitting script (uses SplitCap)
│   │   └── SplitCap.exe            # SplitCap binary (requires Mono on Linux)
│   ├── MFR Generating/             # Convert per-flow PCAPs to MFR format
│   │   └── data_process.py         # Extracts header+payload byte sequences
│   ├── Corpus Generating/          # Build pre-training corpus from MFR data
│   │   └── prepare_dataset_pretrain.py  # Converts MFR matrices to .pt corpus
│   └── Fine-Tuning Dataset Generating/
│       └── Fine-Tuning_Dataset_Generating.py  # Builds .pt datasets for fine-tuning
├── pre-training/
│   ├── two-task_with_two-attention_pretrain.py  # Pre-training entry point
│   ├── multi_task_trainer.py                    # Trainer with dual-task objectives
│   └── uer/                        # UER framework (encoders, targets, utils, etc.)
├── fine-tuning/
│   ├── Two-task_fine-tuning_two-attention.py    # Fine-tuning entry point
│   └── uer/                        # UER framework (same structure as pre-training)
└── models/
    └── bert/                       # BERT config files (tiny/mini/small/medium/base/large)
```

## Data Preparation

### Step 1 — Split raw PCAP files into flows

```bash
cd "data_process/Traffic Splitting"
python split_flows.py \
    --input_dir /path/to/raw_pcaps \
    --output_dir /path/to/flow_pcaps
```

Requires `mono` and `SplitCap.exe`. On Ubuntu: `sudo apt install mono-complete`.

### Step 2 — Generate MFR (Multi-Field Representation) dataset

```bash
cd "data_process/MFR Generating"
python data_process.py \
    --input_dir /path/to/flow_pcaps \
    --output_dir /path/to/mfr_dataset
```

Each packet is represented as 80 hex tokens: 40 header tokens + 40 payload tokens.

### Step 3 — Generate pre-training corpus

Edit `DATASETS_TO_PROCESS` and the MFR root path in `prepare_dataset_pretrain.py`, then:

```bash
cd "data_process/Corpus Generating"
python prepare_dataset_pretrain.py \
    --mfr_root /path/to/mfr_dataset \
    --output_path /path/to/dataset_pretrain.pt
```

Each flow's MFR matrix (8 rows × 100 tokens) is split into 5 packet sequences of 160 tokens each. The output `.pt` file is used as `--dataset_path` in pre-training.

### Step 4 — Generate fine-tuning dataset

Edit `SOURCE_BASE_DIR` and `TARGET_BASE_DIR` in `Fine-Tuning_Dataset_Generating.py`, then:

```bash
cd "data_process/Fine-Tuning Dataset Generating"
python Fine-Tuning_Dataset_Generating.py
```

Outputs `.pt` files ready for fine-tuning.

## Pre-Training

```bash
cd pre-training
python two-task_with_two-attention_pretrain.py \
    --dataset_path /path/to/pretrain_dataset.pt \
    --vocab_path /path/to/vocab.txt \
    --output_model_path /path/to/output/pretrained_model.bin \
    --config_path ../models/bert/base_config.json \
    --target multi_task \
    --total_steps 100000 \
    --batch_size 32 \
    --seq_length 330 \
    --nsp_indicating_weight 0.50 \
    --mfr_masking_weight 0.50 \
    --use_packet_attention \
    --use_flow_attention \
    --gpu_ranks 0 \
    --world_size 1
```

The two pre-training tasks are:
- **NSP-Indicating** (weight 0.5): next-packet sequence prediction
- **MFR-Masking** (weight 0.5): masked header/payload byte recovery

## Fine-Tuning

```bash
cd fine-tuning
python Two-task_fine-tuning_two-attention.py \
    --pretrained_model_path /path/to/pretrained_model.bin \
    --dataset_path /path/to/finetuning_dataset.pt \
    --vocab_path /path/to/vocab.txt \
    --output_model_path /path/to/output/finetuned_model.bin \
    --config_path ../models/bert/base_config.json \
    --labels_num <NUM_CLASSES> \
    --epochs_num 10 \
    --batch_size 32 \
    --gpu_ranks 0
```

## Model Architecture

The encoder stacks two levels of attention:

- **Packet-level attention** (per packet, 3 sub-layers):
  - `HeaderAttentionLayer` — self-attention over the 40 header tokens
  - `CrossAttentionLayer` — payload tokens attend to header tokens
  - `PayloadAttentionLayer` — windowed self-attention over payload tokens

- **Flow-level attention** :
  - `FlowAttention` — each packet's header and pooled payload attend to neighboring packets


## Datasets

The following public datasets are used. Download links are provided below.

| Dataset | Task | Download |
|---|---|---|
| NUDT-MobileTraffic | Mobile app classification | <!-- TODO: fill in --> |
| USTC-TFC2016 | Malware traffic classification | <!-- TODO: fill in --> |
| ISCXVPN2016 | VPN traffic classification | <!-- TODO: fill in --> |
| ISCXVPN2016-APP | App-level VPN classification | <!-- TODO: fill in --> |
| CICIoT2023 | IoT traffic classification | <!-- TODO: fill in --> |

## Pre-Trained Model

The pre-trained model checkpoint can be downloaded from:

<!-- TODO: fill in download link -->

Place the downloaded file at `models/pretrained_model.bin` and pass it via `--pretrained_model_path`.

## Requirements

```bash
pip install torch scapy dpkt tqdm numpy
```

Python 3.8+ and PyTorch 1.10+ are recommended.
