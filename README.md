# iomt-graph-representation

**From Flow Features to Communication Topology: Representation and Evaluation in Graph-Based IoMT Intrusion Detection on CICIoMT2024**

Osman Yildiz and Abdulhamit Subasi  
University at Albany, SUNY — AI4HEALTH Lab (Subasi Lab)

---

## Overview

This repository contains the code and experimental pipeline for our study on graph-based intrusion detection for Internet of Medical Things (IoMT) networks. Rather than proposing a new model architecture, we systematically investigate how **data representation**, **graph construction**, **evaluation protocol**, and **task formulation** shape detection performance on the CICIoMT2024 benchmark.

### Key Findings

1. **Representation > Architecture**: PCAP-derived communication-topology graphs enable competitive GNN performance where feature-similarity graphs do not.
2. **Evaluation protocol matters**: Proper PCAP-level session-aware validation reduces reported macro-F1 by over 16 percentage points compared to naive test-based early stopping.
3. **Domain-typed edges help**: Same-subnet and gateway-linked edge augmentation improves macro-F1 by 7.4 pp and reduces seed variance by 4×.
4. **GNN effectiveness is attack-category dependent**: Topology-heavy attacks (DDoS, DoS) benefit from graph modeling; protocol-heavy attacks (MQTT, Spoofing) do not.
5. **Architectural complexity does not help**: Neuro-symbolic fusion, motif-aware heads, and node-role auxiliary supervision all failed to outperform a simpler adaptive baseline.

## Repository Structure

```
iomt-graph-representation/
├── README.md
├── requirements.txt
├── src/
│   ├── preprocess.py              # CICIoMT2024 CSV preprocessing
│   ├── graph_construction.py      # Feature-similarity graph builder
│   ├── pcap_extraction.py         # PCAP → IP/port/timestamp extraction
│   ├── natural_graph_builder.py   # Communication-topology graph construction
│   ├── domain_edges.py            # Domain-typed edge augmentation
│   ├── models.py                  # AdaptiveGAT, PureGAT, GraphSAGE
│   ├── evaluation.py              # PCAP-level split, training, metrics
│   └── utils.py                   # Shared utilities
├── scripts/
│   ├── run_final_clean.sh         # Main experiment (6-class, proper protocol)
│   ├── run_diagnostic.sh          # Diagnostic experiments (A/B/C)
│   └── run_motif_gat.sh           # 4-class topology-heavy subset
├── configs/
│   └── default.yaml               # Hyperparameters and paths
├── paper/
│   ├── paper_draft.md             # Full paper (markdown source)
│   └── IoMT_GNN_IDS_Paper.docx   # Paper (Word format)
└── results/                       # Populated after running experiments
```

## Dataset

This study uses [CICIoMT2024](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html) by the Canadian Institute for Cybersecurity.

- **CSV files**: 45 flow-level features, 8.77M flows, 6 coarse attack categories
- **PCAP files**: 59 GB raw packet captures with full header information

Download the dataset and place it according to paths in `configs/default.yaml`.

## Requirements

```bash
pip install -r requirements.txt
```

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric 2.4+
- scikit-learn, numpy, pandas

## Running Experiments

### 1. PCAP Extraction (run once)
```bash
python src/pcap_extraction.py --pcap_dir /path/to/pcaps --output_dir /path/to/ip_extracted
```

### 2. Graph Construction (run once)
```bash
python src/natural_graph_builder.py --csv_dir /path/to/ip_extracted --output_dir /path/to/natural_graphs
```

### 3. Main Experiments
```bash
# On SLURM cluster (DGX)
sbatch scripts/run_final_clean.sh

# Or directly
python src/evaluation.py --config configs/default.yaml --experiment final_clean
python src/evaluation.py --config configs/default.yaml --experiment diagnostic
python src/evaluation.py --config configs/default.yaml --experiment motif_4class
```

## Results Summary

### 6-Class (Proper Protocol, 5 Seeds)

| Model | Accuracy | F1-Macro |
|-------|----------|----------|
| RF | 0.853 ± 0.010 | 0.701 ± 0.007 |
| GraphSAGE | 0.867 ± 0.020 | 0.677 ± 0.079 |
| PureGAT | 0.850 ± 0.011 | 0.650 ± 0.072 |
| AdaptiveGAT | 0.875 ± 0.017 | 0.691 ± 0.060 |

### 4-Class Topology-Heavy (Domain-Typed Edges, 5 Seeds)

| Model | Accuracy | F1-Macro |
|-------|----------|----------|
| RF | 0.872 ± 0.008 | 0.784 ± 0.020 |
| AdaptiveGAT | 0.888 ± 0.011 | 0.800 ± 0.026 |

### Evaluation Protocol Effect

| Protocol | F1-Macro |
|----------|----------|
| Naive (test-based ES) | 0.852 |
| Proper (PCAP-level val) | 0.691 ± 0.060 |

## Citation

If you use this code or findings, please cite:

```bibtex
@article{yildiz2026iomt,
  title={From Flow Features to Communication Topology: Representation and Evaluation in Graph-Based IoMT Intrusion Detection on CICIoMT2024},
  author={Yildiz, Osman and Subasi, Abdulhamit},
  journal={Future Internet},
  year={2026},
  publisher={MDPI}
}
```

## License

MIT License

## Contact

- Osman Yildiz: oyildiz@albany.edu
- Abdulhamit Subasi: asubasi@albany.edu
