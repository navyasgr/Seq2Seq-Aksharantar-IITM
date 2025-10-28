#  Seq2Seq Transliteration with Attention — Aksharantar (IIT Madras)

**Romanized → Devanagari transliteration** using a character-level Seq2Seq model with **LSTM + Bahdanau Attention**.  
This repo is a compact, well-documented, and reproducible solution prepared for the IIT Madras Technical Aptitude challenge.

---

##  Quick summary
- **Task:** Map romanized character sequences (e.g., `ghar`) → native script (e.g., `घर`).  
- **Model:** Encoder (LSTM) + Decoder (LSTM) with Bahdanau additive attention.  
- **Language:** Python + PyTorch. Ready to run on Colab GPU.

---

##  Repo structure (what's important)
```
Seq2Seq-Aksharantar-IITM/
├── src/
│   ├── models/
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   ├── attention.py
│   │   └── seq2seq.py
│   ├── data/
│   │   ├── tokenizer.py
│   │   └── dataset.py
│   └── utils/
│       ├── metrics.py
│       └── training_loop.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── config/
│   ├── model_config.yaml
│   └── data_config.yaml
├── notebooks/
│   └── Transliteration_Report.ipynb
├── data/
│   └── raw/sample.tsv
├── requirements.txt
└── README.md
```

---

##  Architecture (ASCII diagram)
```
      Input (Romanized) chars
               │
         [Embedding Layer]
               │
         [LSTM Encoder]  -> encoder outputs (sequence of h vectors)
               │
               ├───────────────────┐
               │                   ▼
            (enc outputs)    Bahdanau Attention
               │                   │
               │                   ▼
            (context vector) ──> [LSTM Decoder] (uses prev token + context)
                                     │
                                  [Linear]
                                     │
                                  Softmax
                                     │
                                  Output char
```

---

##  Default Configs (chosen for Colab & IITM)
- `embedding_size = 128`  
- `hidden_size = 256`  
- `rnn_cell = LSTM`  
- `use_attention = true`  
- `num_layers = 1`  
- `dropout = 0.2`  
- `batch_size = 64`  
- `learning_rate = 0.001`

You can change these in `config/model_config.yaml`.

---

##  Math (Answer to assignment questions — explicit & worked example)

### Notation
- `e` = embedding dimension  
- `h` = hidden dimension (encoder & decoder)  
- `T` = input/output sequence length (assumed equal for derivation)  
- `V` = vocabulary size (same for source & target)  
- Single-layer encoder and decoder (adjust multipliers for multiple layers)

### Parameter count (vanilla RNN symbolic)
\[
P_\text{total} = 2Ve + 2(eh + h^2 + h) + hV + V
\]
(Embeddings + encoder RNN + decoder RNN + output projection)

### Computation (dominant matmuls) — forward pass
\[
\text{Matmuls} = T \cdot (2eh + 2h^2 + hV)
\]
Multiply by 2 for multiply+add FLOPs. Training (~backprop) ≈ ×3 forward cost.

### Worked numeric example
Use `e=128`, `h=256`, `V=5000`, `T=20`:

- Parameters ≈ **2,762,120** (≈ 10.5 MB in float32)  
- Forward matmuls ≈ **29,532,160** → ≈ **59M FLOPs** (multiply+add)

(Full derivation and LSTM/GRU variants are in `notebooks/Transliteration_Report.ipynb`.)

---

##  How to run (quick)
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Quick smoke test (uses tiny sample shipped in `data/raw/sample.tsv`):
```bash
python scripts/train.py --config config/model_config.yaml --quick_test
```

3. Full training:
```bash
python scripts/train.py --config config/model_config.yaml --out_dir checkpoints
```

4. Evaluate a checkpoint:
```bash
python scripts/evaluate.py --config config/model_config.yaml --checkpoint checkpoints/best.pt
```

---

##  Notebook & Visualizations
Open `notebooks/Transliteration_Report.ipynb` to see:
- Architecture explanation
- Mathematical derivation (step-by-step)
- Attention visualization sketch (heatmaps) and sample outputs

---

##  References & Acknowledgements
- AI4Bharat — Aksharantar dataset  
- Bahdanau et al. (2014) — Neural Machine Translation by Jointly Learning to Align and Translate  
- PyTorch Seq2Seq tutorial

---

##  Author
Prepared for IIT Madras Technical Aptitude Challenge by **Navyashree N**.

---
