# From Flow Features to Communication Topology: Representation and Evaluation in Graph-Based IoMT Intrusion Detection on CICIoMT2024

**Osman Yildiz ¹,* and Abdulhamit Subasi ¹**

¹ Department of Information Science, College of Emergency Preparedness, Homeland Security and Cybersecurity, University at Albany, State University of New York, Albany, NY 12222, USA

\* Correspondence: oyildiz@albany.edu (O.Y.); asubasi@albany.edu (A.S.)

---

## Abstract

Graph neural networks have been increasingly explored for network intrusion detection, yet the effect of graph construction strategy on detection performance remains underexamined — particularly for IoMT networks. In this study, we systematically investigate how data representation, graph construction, evaluation protocol, and task formulation shape the effectiveness of graph-based intrusion detection on the CICIoMT2024 benchmark. We compare three representation strategies: flow-level tabular features, feature-similarity graphs, and PCAP-derived communication-topology graphs constructed from raw packet captures. We further examine the effect of domain-typed edge augmentation, PCAP-level validation protocols, and task decomposition into topology-heavy and protocol-heavy attack categories. Our experiments reveal several findings that challenge common assumptions. First, feature-similarity graphs do not provide GNNs with a reliable advantage over Random Forest baselines. Second, PCAP-derived communication topology yields a more meaningful relational representation, enabling GNNs to become competitive in topology-heavy attack detection. Third, domain-aware edge typing improves both performance and stability. Fourth, under proper PCAP-level validation with session-aware splits, previously reported gains diminish substantially, underscoring the importance of evaluation protocol. Fifth, in our experiments on this dataset, GNN effectiveness depends on attack category: topology-heavy attacks (DDoS, DoS, Recon) benefit from graph modeling, while protocol-heavy attacks (MQTT, Spoofing) do not. Across five random seeds, a domain-typed Adaptive Edge-Weighted GAT achieves a macro-F1 of 0.800 ± 0.026 on the topology-heavy subset, compared with 0.784 ± 0.020 for Random Forest. These results suggest that in IoMT intrusion detection, representation choice and evaluation protocol matter more than architectural complexity.

**Keywords:** Internet of Medical Things; intrusion detection system; graph neural network; graph construction; communication topology; evaluation protocol; CICIoMT2024

---

## 1. Introduction

IoMT devices are now routine in clinical settings. Patient monitors, infusion pumps, wearable sensors, and diagnostic equipment communicate over Wi-Fi, MQTT, and Bluetooth, generating continuous network traffic. These devices are difficult to patch, often lack built-in security mechanisms, and a compromised endpoint can directly affect patient care. In this setting, intrusion detection is part of operational safety rather than a purely technical add-on.

Machine learning classifiers have made strong progress on this front. Random Forest, XGBoost, and deep learning models have reported very high accuracy, often above 98%, under standard flow-level evaluation settings on IoMT benchmarks [1,2]. The CICIoMT2024 dataset [3], comprising traffic from 40 IoMT devices under 18 attack scenarios, has attracted considerable attention for this purpose. Published results on its pre-extracted CSV files are consistently high.

But accuracy on a flow-level tabular task and readiness for deployment are different things. Flow-level classifiers process each record independently. They cannot represent that 30 sources are converging on a single target — a DDoS fan-in signature — because they never see flows in relation to each other. A reconnaissance scan, where one host probes dozens of destinations, produces a fan-out pattern that is only visible when flows are considered as a communication graph. These topological patterns are characteristic of IoMT attacks, but they are not directly represented when flows are treated as isolated rows.

Graph Neural Networks can model such relational structure by representing devices as nodes and traffic flows as edges. Several studies have applied GNNs to intrusion detection on other benchmarks [4–8], demonstrating value especially when communication structure or inter-flow context is informative. However, a question that has received limited systematic attention is how much the graph construction strategy itself — as opposed to the GNN architecture — determines whether graph-based detection actually works.

This question turns out to matter more than we initially expected. Our investigation on CICIoMT2024 began with a standard feature-similarity graph formulation: flows as nodes, cosine similarity as edges. GNN models trained on these graphs performed reasonably but never consistently outperformed a window-level Random Forest baseline. The artificial edges did not appear to encode enough structural information to give GNNs a consistent advantage.

When we reconstructed the communication topology from the raw PCAP files — IPs as nodes, aggregated flows as edges, in 60-second windows — the graph models became substantially more competitive. Adding domain-typed edges (same-subnet, gateway-linked) further improved both performance and stability, consistent with findings that domain-aware graph construction matters in GNN-based IDS [7,8].

But the investigation also revealed uncomfortable findings. When we replaced naive evaluation with proper PCAP-level session-aware validation, previously encouraging margins shrank. In the full 6-class setting, our best GNN model did not consistently outperform Random Forest. It was only when we separated topology-heavy attacks (DDoS, DoS, Recon) from protocol-heavy attacks (MQTT, Spoofing) that GNNs showed a clearer and more stable advantage. Additional architectural complexity did not yield consistent gains over a simpler adaptive baseline.

Taken together, the experiments suggest that in IoMT intrusion detection on CICIoMT2024, the primary factors shaping graph-based detection performance are data representation, evaluation protocol, and task formulation — not architectural complexity. The specific contributions of this paper are:

1. A systematic comparison of three data representations — flow-level tabular, feature-similarity graphs, and PCAP-derived communication-topology graphs — showing that graph construction strategy has a larger effect on GNN performance than model architecture.

2. A PCAP-to-graph pipeline that extracts natural network topology from raw packet captures and augments it with domain-typed edges (communication, same-subnet, gateway-linked), yielding measurable gains in both performance and stability.

3. Evidence that evaluation protocol substantially affects reported results: under PCAP-level session-aware validation, gains that appear significant under naive splits diminish considerably.

4. A demonstration that, in our experiments on this dataset, GNN utility is attack-category dependent: graph models are more competitive for topology-heavy attacks but not for protocol-heavy attacks, motivating task decomposition in future IoMT IDS design.

5. Negative results from multiple architectural extensions — neuro-symbolic fusion, motif-aware detection heads, and node-role auxiliary supervision — reinforcing that representation matters more than model complexity in this setting.

The rest of this paper is organized as follows. Section 2 reviews related work. Section 3 describes the dataset, representations, models, and evaluation protocol. Section 4 presents results. Section 5 discusses implications and limitations. Section 6 concludes.

---

## 2. Related Work

### 2.1. Flow-Level Intrusion Detection

Tree-based ensembles — particularly Random Forest and XGBoost — remain strong baselines for flow-level network intrusion detection. On benchmarks such as NSL-KDD, CIC-IDS2017, and UNSW-NB15, these models benefit from the fact that CICFlowMeter-extracted features are already highly discriminative: packet counts, flag distributions, inter-arrival times, and byte statistics often separate attack categories with relatively simple models. Deep learning approaches (1D-CNNs, LSTMs, autoencoders) have been applied to the same datasets but tend to show advantages primarily when raw or sequential input is used rather than pre-computed tabular features.

### 2.2. Graph-Based Intrusion Detection

GNNs entered the IDS literature to exploit the relational structure that flow-level models discard. The argument is that network traffic has an inherent graph structure — hosts communicate with other hosts — and encoding this structure should improve detection.

E-GraphSAGE [4] extended GraphSAGE with edge feature support, performing edge classification to label individual flows on CIC-IDS2017. This established the basic formulation: IP-port pairs as nodes, flows as edges. Subsequent work explored variations in graph construction. MGF-GNN [7] argued that a single graph perspective is insufficient and proposed three separate graphs at host, port, and protocol granularity, fusing them through hierarchical message passing. BS-GAT [8] constructed graphs using behavioral similarity between flows rather than network topology. Other recent directions include continual learning for GNN-based IDS [13] and integration of learnable activation functions with graph attention [12].

A pattern across these studies is that graph construction is typically treated as a fixed preprocessing step while modeling effort concentrates on the GNN architecture. Our work reverses this emphasis: we hold the architecture relatively simple and vary the graph construction and evaluation protocol, finding that these factors have a larger effect on outcomes.

### 2.3. CICIoMT2024 Dataset

CICIoMT2024 [3] was released specifically for IoMT security research. It captures traffic from 40 devices (25 real, 15 simulated) over Wi-Fi and MQTT, covering 18 attack types in six categories. It provides both pre-extracted CSV files with 45 flow features and raw PCAP files with full packet captures.

Published work on this dataset has primarily relied on the CSV representation. A Random Forest model with SHAP analysis achieved 99% accuracy [1]. An LSTM hybrid reported 98.05% [2]. Federated learning variants have also been explored [10]. Recent work has highlighted that evaluation protocol and class balancing choices can substantially affect reported performance on IoMT datasets [9,10].

Two aspects of these results are relevant to our study. First, the reported accuracy is driven by majority classes — DDoS and DoS account for over 95% of flows — making overall accuracy a limited indicator for rarer attack types. Second, published work on CICIoMT2024 has primarily relied on the CSV representation, with little to no use of the raw PCAP files in reported detection pipelines. We address both gaps.

---

## 3. Materials and Methods

### 3.1. Dataset: CICIoMT2024

CICIoMT2024 [3] was collected from a testbed of 40 IoMT devices — 25 physical and 15 simulated — operating over Wi-Fi and MQTT protocols. The dataset covers 18 attack types grouped into six coarse categories. Table 1 shows the distribution.

**Table 1.** Class distribution in the CICIoMT2024 dataset (flow-level CSV files).

| Class | Train Flows | Test Flows | Percentage |
|-------|------------|------------|------------|
| Benign | 230,127 | 41,809 | 2.62% |
| DDoS | 6,092,247 | 1,413,732 | 69.49% |
| DoS | 2,289,186 | 515,634 | 26.11% |
| Recon | 131,596 | 29,651 | 1.50% |
| Spoofing | 17,525 | 3,944 | 0.20% |
| MQTT | 7,013 | 1,580 | 0.08% |

DDoS and DoS together account for over 95% of all flows. MQTT constitutes less than 0.1%. This imbalance means that overall accuracy is dominated by majority classes and provides limited information about detection quality for rarer attack types.

The dataset is distributed in two forms. The CSV files contain 45 pre-extracted flow-level features (packet counts, flag statistics, inter-arrival times, byte distributions) but no endpoint identifiers — no source or destination IP, no port numbers. The PCAP files contain raw packet captures with full header information. In this study, we use the CSV files for the feature-similarity baseline and the PCAP files for communication-topology reconstruction.

### 3.2. Representation Strategies

We compare three data representations, each providing a different level of relational information.

#### 3.2.1. Flow-Level Tabular (Baseline)

Each flow is a row with 31 features (after removing 14 near-constant or redundant columns identified during exploratory analysis). Classification is performed independently per flow. This is the standard approach in the CICIoMT2024 literature.

#### 3.2.2. Feature-Similarity Graphs

We divide the flow sequence into windows of 500 flows with a stride of 250. Within each window, every flow becomes a node. Edges are the union of two sets: cosine similarity edges connecting each node to its k = 10 nearest neighbors in feature space, and protocol homophily edges connecting nodes sharing the same protocol type. Each window receives a label by majority vote among its flows. Graph windows are generated separately within the original train and test partitions provided by the dataset. This produces 28,621 training graphs and 6,446 test graphs.

#### 3.2.3. PCAP-Derived Communication-Topology Graphs

This representation uses the raw PCAP files to reconstruct the actual network topology.

*Packet extraction.* Each PCAP is processed with tshark to extract source IP, destination IP, source and destination ports, timestamp, and protocol for every packet. This yields approximately 200 million records from 59 GB of raw captures.

*Temporal windowing.* Packets are segmented into non-overlapping 60-second windows. This duration was selected as a practical compromise between graph sparsity and temporal granularity. Each window inherits the label of its source PCAP file.

*Flow aggregation.* Within each window, packets are grouped by (source IP, destination IP) pairs. Each unique pair becomes a directed edge with 14 features: packet count, duration, bytes per second, mean and standard deviation of inter-arrival time, burstiness (std/mean IAT), unique TCP and UDP source and destination ports, dominant protocol, protocol diversity, TCP ratio, and UDP ratio.

Each unique IP becomes a node with 12 features computed from local topology: in-degree, out-degree, total incoming packets, total outgoing packets, mean incoming rate, mean outgoing rate, degree ratio (out/in), traffic asymmetry (outgoing/incoming packets), unique peers, internal IP indicator, mean incoming burstiness, and mean outgoing burstiness.

Table 2 shows the resulting graph distribution.

**Table 2.** Communication-topology graph distribution (60-second windows).

| Class | Train | Test | Avg Nodes | Avg Edges |
|-------|-------|------|-----------|-----------|
| Benign | 1,324 | 237 | 34.8 | 45.3 |
| DDoS | 283 | 87 | 42.3 | 62.7 |
| DoS | 193 | 64 | 37.3 | 57.9 |
| MQTT | 8 | 4 | 41.6 | 66.5 |
| Recon | 46 | 15 | 35.5 | 53.2 |
| Spoofing | 35 | 4 | 34.5 | 55.2 |

### 3.3. Domain-Typed Edge Augmentation

Communication-topology graphs capture who communicates with whom, but not the semantic type of the relationship. We augment the base communication edges with two domain-informed edge types:

*Same-subnet edges.* Internal IP addresses sharing the same /24 subnet prefix that do not already have a direct communication edge are connected. These edges provide a coarse network-locality prior.

*Gateway-linked edges.* Nodes corresponding to known gateway IP addresses (identified from dataset documentation and traffic analysis as 10.0.0.254) are connected to all internal nodes without existing direct edges. These represent a gateway-connectivity prior.

Each edge carries a 3-dimensional one-hot type indicator (communication, same-subnet, gateway-linked) appended to the 14 traffic features, yielding 17-dimensional edge feature vectors. Non-communication edges receive scaled proxy features (10% of the mean communication edge features), chosen as an exploratory design decision to keep structural edges informative but weaker than observed traffic edges.

### 3.4. Adaptive Edge-Weighted GAT

Our GNN model builds on GAT with edge feature injection and adaptive edge weighting.

*Edge feature injection.* Communication-topology graphs carry 14 (or 17 with typed edges) features per edge. We use the edge_dim parameter of GATConv to incorporate edge information into the attention computation.

*Adaptive edge weighting.* Following prior work on adaptive topology learning [11], we fuse prior and learned edge weights:

> w_fused = α · w_prior + (1 − α) · w_learned

where w_prior = 1 for all edges (uniform), w_learned is produced by a two-layer MLP with sigmoid activation applied to each edge feature vector, and α is a learnable scalar initialized at 0.5.

The architecture consists of a linear node projection to 128 dimensions, three GAT layers (4 heads in the first two, 1 in the last) with LayerNorm and residual connections, Global Attention Pooling, and a two-layer MLP classifier. Training uses AdamW (lr = 5 × 10⁻⁴, weight decay = 10⁻⁴), cosine annealing, class-weighted cross-entropy, and gradient clipping at 1.0.

### 3.5. Evaluation Protocol

Evaluation protocol is a central concern of this paper. We use two protocols to demonstrate the effect of validation design on reported results.

*Naive protocol.* Early stopping is based on test set macro-F1. This is methodologically incorrect — the test set influences model selection — but is reported only as an illustrative comparison and not used for final model selection claims.

*Proper protocol.* We reserve 15% of training graphs as a validation set using PCAP-level stratified sampling: all graphs from the same source PCAP file are assigned to the same split, preventing session-level leakage. For classes with two or fewer source PCAPs (MQTT, Spoofing, Benign in some configurations), all graphs remain in the training set and the class is absent from validation; this is noted where applicable. Early stopping with patience of 40 epochs is based on validation macro-F1. The best checkpoint is evaluated on the held-out test set. Feature scalers are fit on the training subset only. All experiments under proper protocol are repeated across five random seeds (42, 123, 456, 789, 2026), and we report mean ± standard deviation.

### 3.6. Task Formulations

We evaluate three task formulations to investigate whether GNN effectiveness depends on the attack categories included.

*Full 6-class.* All six coarse categories (Benign, DDoS, DoS, MQTT, Recon, Spoofing). This is the primary benchmark setting.

*Topology-heavy 4-class.* Benign, DDoS, DoS, Recon only. MQTT and Spoofing are excluded because they have very small support (8 and 35 training graphs respectively) and their attack signatures are more protocol- or identity-based than topology-based. This formulation is used to study topology-sensitive attack discrimination rather than to replace the full benchmark setting.

*Binary.* Attack versus Benign. This is a diagnostic task used to assess whether GNNs can reliably distinguish normal from anomalous network topology, independent of fine-grained categorization. Because the attack class dominates the dataset, we report macro-F1 and treat this task as diagnostic rather than as the primary deployment setting.

### 3.7. Baselines

For each graph formulation, we compare:

- **Random Forest.** Trained on graph-level aggregate features: mean, standard deviation, and maximum of all node features and edge features, plus graph structural statistics (node count, edge count, density). This provides a non-graph baseline at the same classification granularity.
- **GraphSAGE.** SAGE convolution with the same depth, hidden dimension, and pooling as the GAT models, but without edge features or adaptive weighting.
- **PureGAT.** GAT with edge feature injection but no adaptive edge weighting.

For flow-level and feature-similarity experiments, the corresponding window-level RF baseline is used. All graph baselines use the same node features, graph splits, and evaluation protocol unless otherwise noted.

---

## 4. Results

All communication-topology results in this section use the proper evaluation protocol described in Section 3.5 unless explicitly marked otherwise. Feature-similarity results use the dataset's original train/test split with a single run, as these serve as an exploratory reference rather than the primary evaluation.

### 4.1. Evaluation Protocol Matters

Before comparing representations, we first demonstrate why evaluation protocol is critical. Table 3 shows the same model (AdaptiveGAT on communication-topology graphs, 6-class) under two protocols.

**Table 3.** Effect of evaluation protocol on reported performance (AdaptiveGAT, 6-class, communication-topology graphs).

| Protocol | Accuracy | F1-Macro |
|----------|----------|----------|
| Naive (test-based ES) | 0.910 | 0.852 |
| Proper (PCAP-level val, 5 seeds) | 0.875 ± 0.017 | 0.691 ± 0.060 |

Under naive evaluation — where early stopping is based on test set performance — the model appears to achieve 0.852 macro-F1. Under proper PCAP-level validation with session-aware splits, the same architecture achieves 0.691 ± 0.060. This gap reflects evaluation bias rather than a genuine modeling gain. The naive protocol allows the model to select its best test-set epoch, inflating the result. We include this comparison because it highlights the need for clearer evaluation protocol reporting in the CICIoMT2024 literature, where evaluation details are not always described in sufficient detail.

All subsequent results use the proper protocol.

### 4.2. Representation Comparison

Tables 4a and 4b are not directly comparable: they use different graph formulations, different windowing, different splits, and different evaluation protocols. Table 4a serves as an exploratory reference showing the GNN-RF relationship under feature-similarity construction. Table 4b presents the primary evidence under communication-topology construction with proper validation.

**Table 4a.** Feature-similarity graphs (500-flow windows, single run, exploratory reference).

| Model | Accuracy | F1-Macro |
|-------|----------|----------|
| Window-RF | 0.992 | 0.942 |
| GCN | 0.871 | 0.900 |
| GraphSAGE | 0.863 | 0.894 |
| GAT | 0.893 | 0.940 |

On feature-similarity graphs, no GNN model outperforms the window-level RF in accuracy. GAT approaches RF in macro-F1 (0.940 vs 0.942) but the difference is negligible. In our experiments, cosine-similarity and protocol-homophily edges did not provide a consistent advantage over tabular baselines.

**Table 4b.** Communication-topology graphs (60-second PCAP windows, proper protocol, 5 seeds, primary evidence).

| Model | Accuracy | F1-Macro |
|-------|----------|----------|
| RF | 0.853 ± 0.010 | 0.701 ± 0.007 |
| GraphSAGE | 0.867 ± 0.020 | 0.677 ± 0.079 |
| PureGAT | 0.850 ± 0.011 | 0.650 ± 0.072 |
| AdaptiveGAT | 0.875 ± 0.017 | 0.691 ± 0.060 |

On communication-topology graphs with proper validation, the margins are much smaller. AdaptiveGAT is competitive with RF (0.691 vs 0.701 in macro-F1) where feature-similarity GNNs were not. The variance across seeds is notable, particularly for GraphSAGE and PureGAT. AdaptiveGAT shows the lowest variance among GNN models.

The point is not a direct numerical comparison between Tables 4a and 4b, but the qualitative shift: under feature-similarity construction, GNNs cannot match RF; under communication-topology construction, they become competitive. What changed is the information available to the model, not the model itself.

### 4.3. Effect of Domain-Typed Edges

Table 5 shows the effect of adding domain-typed edges (same-subnet, gateway-linked) to the communication-topology graphs. Both use AdaptiveGAT with proper protocol. This diagnostic comparison was run with three seeds for faster iteration; the primary model comparisons in Sections 4.4–4.5 use five seeds.

**Table 5.** Effect of domain-typed edge augmentation (AdaptiveGAT, 6-class, diagnostic comparison, 3 seeds).

| Edge Type | F1-Macro | Std |
|-----------|----------|-----|
| Communication only | 0.608 ± 0.030 | 0.030 |
| + Domain-typed | 0.682 ± 0.007 | 0.007 |

Domain-typed edges improve macro-F1 by 7.4 percentage points and reduce variance by a factor of four. The stability improvement is particularly notable: the model becomes much less sensitive to seed and split variation when domain relations are included.

### 4.4. Full 6-Class Results

Table 6 shows per-class F1 under the full 6-class setting (proper protocol, 5 seeds, communication-topology graphs).

**Table 6.** Per-class F1 scores (6-class, communication-topology graphs, proper protocol, 5 seeds).

| Class | RF | GraphSAGE | PureGAT | AdaptiveGAT |
|-------|----|-----------|---------|----|
| Benign | 0.955 ± 0.003 | 0.964 ± 0.004 | 0.968 ± 0.009 | 0.969 ± 0.007 |
| DDoS | 0.741 ± 0.032 | 0.779 ± 0.026 | 0.761 ± 0.007 | 0.801 ± 0.037 |
| DoS | 0.655 ± 0.021 | 0.713 ± 0.093 | 0.602 ± 0.043 | 0.751 ± 0.055 |
| MQTT | 0.933 ± 0.054 | 0.652 ± 0.184 | 0.560 ± 0.248 | 0.608 ± 0.221 |
| Recon | 0.616 ± 0.104 | 0.691 ± 0.128 | 0.640 ± 0.106 | 0.666 ± 0.101 |
| Spoofing | 0.307 ± 0.155 | 0.262 ± 0.185 | 0.366 ± 0.169 | 0.354 ± 0.116 |

For DDoS and DoS — attacks with strong topological signatures — AdaptiveGAT outperforms RF (DDoS: 0.801 vs 0.741; DoS: 0.751 vs 0.655). For MQTT, RF substantially outperforms all GNN models (0.933 vs 0.608). MQTT has only 8 training graphs; GNN models may not learn a stable representation from so few examples, while RF benefits from aggregate features that are sufficient for this category. Spoofing is poorly detected by all models, with very high variance reflecting only 4 test graphs.

These results illustrate an important pattern: in our experiments, GNN effectiveness is attack-category dependent. Topology-heavy attacks where structural patterns (fan-in, fan-out) are informative show GNN advantages. Protocol-heavy or extremely low-support categories do not.

### 4.5. Topology-Heavy 4-Class Subset

Motivated by the per-class analysis above, we evaluate the same models on a topology-heavy subset: Benign, DDoS, DoS, and Recon. MQTT and Spoofing are excluded due to very small support and protocol-dependent nature. This subset is used as a controlled analysis of topology-sensitive attack discrimination, while the full 6-class results in Section 4.4 remain the primary benchmark setting.

**Table 7.** Topology-heavy 4-class results (domain-typed edges, proper protocol, 5 seeds).

| Model | Accuracy | F1-Macro |
|-------|----------|----------|
| RF | 0.872 ± 0.008 | 0.784 ± 0.020 |
| AdaptiveGAT | 0.888 ± 0.011 | 0.800 ± 0.026 |

**Table 8.** Per-class F1 (4-class topology-heavy subset, 5 seeds).

| Class | RF | AdaptiveGAT |
|-------|-----|-----|
| Benign | 0.963 ± 0.002 | 0.969 ± 0.007 |
| DDoS | 0.765 ± 0.016 | 0.801 ± 0.037 |
| DoS | 0.670 ± 0.041 | 0.772 ± 0.033 |
| Recon | 0.739 ± 0.038 | 0.668 ± 0.066 |

In the 4-class setting, AdaptiveGAT outperforms RF in overall macro-F1 (0.800 vs 0.784). The advantage is clearest for DoS (+10.2 percentage points) and DDoS (+3.6 points). RF retains an advantage on Recon (0.739 vs 0.668), which has only 46 training graphs and may not provide sufficient data for GNN generalization. The GNN advantage, while present, is modest and should be interpreted in light of the variance.

### 4.6. Binary Detection

As a diagnostic task, we evaluate attack-versus-benign binary classification under proper protocol.

**Table 9.** Binary classification (proper protocol, 3 seeds).

| Model | F1-Macro |
|-------|----------|
| RF | 0.965 ± 0.002 |
| AdaptiveGAT | 0.975 ± 0.009 |

Both models perform well. AdaptiveGAT slightly outperforms RF across seeds. This result is reported as a diagnostic indicator of whether the communication-topology representation carries a useful anomaly signal, not as the primary evaluation of the proposed approach. It confirms that the challenge lies in fine-grained multi-class discrimination, not in detecting that something is wrong.

### 4.7. Negative Results: Architectural Extensions

We explored several architectural extensions beyond AdaptiveGAT under proper protocol: neuro-symbolic fusion with explicit IF-THEN rules and class-aware gates, node-role auxiliary supervision with pseudo-labeled victim/scanner/broker roles, and motif-aware detection heads with topology-grounded pattern scoring. In each case, the extension failed to consistently outperform the AdaptiveGAT baseline across seeds. Detailed results from these exploratory experiments are included in the Supplementary Materials.

We report this as a meaningful negative finding. These results suggest that, in this setting, representation appears to be the more limiting factor than additional architectural complexity.

### 4.8. Summary of Results

Across all experiments, three factors consistently shaped the results more than model architecture: (i) the evaluation protocol — naive early stopping inflated performance by over 16 percentage points; (ii) the data representation — communication-topology graphs enabled competitive GNN performance where feature-similarity graphs did not; and (iii) the task formulation — GNNs showed clearer advantages on topology-heavy attack subsets than on the full 6-class benchmark. Domain-typed edge augmentation was the most effective single graph-level intervention, improving both macro-F1 and seed-to-seed stability. Additional architectural complexity — neuro-symbolic fusion, motif detection, node-role supervision — did not yield consistent gains beyond the adaptive baseline.

---

## 5. Discussion

### 5.1. When Do Graph Models Help?

The results paint a nuanced picture. Graph-based models do not universally outperform tabular baselines on CICIoMT2024. Under proper evaluation, the overall 6-class macro-F1 of AdaptiveGAT (0.691) is comparable to but does not clearly exceed that of Random Forest (0.701). The advantage of graph models is selective and conditional.

Graph models are more competitive when three conditions are met. First, the graph must carry meaningful relational structure — communication-topology graphs derived from PCAP files satisfy this, while feature-similarity graphs do not. Second, domain-typed edges that encode network infrastructure relationships (subnet membership, gateway connectivity) improve both performance and stability. Third, the classification task must align with the relational inductive bias: topology-heavy attacks like DDoS (fan-in) and DoS show the clearest GNN advantage, while protocol-heavy attacks like MQTT do not benefit.

This selectivity is informative rather than discouraging. It suggests that graph models for IoMT IDS should not be applied as a universal replacement for tabular classifiers, but rather deployed where their structural bias matches the attack signature. A practical IoMT IDS might use a tabular classifier as a fast first-stage detector and reserve graph-based analysis for cases where topological context is expected to be informative.

### 5.2. Representation vs Architecture

Across our experiments, changes to data representation produced larger and more consistent performance shifts than changes to model architecture. Moving from feature-similarity to communication-topology graphs changed the competitive position of GNNs relative to RF. Adding domain-typed edges improved macro-F1 by 7.4 percentage points while reducing variance by a factor of four. In contrast, four separate architectural extensions — neuro-symbolic fusion, class-aware rule fusion, motif-aware heads, and node-role auxiliary supervision — each failed to outperform the simpler AdaptiveGAT under proper evaluation.

This finding has practical implications. Researchers working on GNN-based IDS may benefit more from investing in graph construction and relation design than from developing increasingly complex attention mechanisms or auxiliary objectives. The bottleneck, at least on CICIoMT2024, appears to be in what information reaches the model rather than how the model processes it.

### 5.3. The Evaluation Protocol Problem

The difference between naive and proper evaluation protocol was the single largest factor in our experiments — larger than any representation or architecture change. Under naive test-based early stopping, AdaptiveGAT achieved 0.852 macro-F1. Under proper PCAP-level validation, the same architecture achieved 0.691. This 16-point gap is not a modeling result; it is an evaluation artifact.

This finding highlights the need for clearer evaluation protocol reporting in the CICIoMT2024 literature, where very high accuracy figures are commonly reported but evaluation protocols are not always described in detail. Our results suggest that session-level data leakage — where graphs from the same attack session appear in both training and evaluation — can substantially inflate reported performance. We suggest that future work on this dataset explicitly describe split construction at the PCAP or session level and report validation-based model selection.

### 5.4. Implications for Deployment-Oriented IoMT IDS Design

Our findings suggest several design considerations for IoMT intrusion detection systems intended for real-world deployment.

First, raw PCAP access matters. The communication topology extracted from packet captures provides information that pre-extracted CSV features do not — endpoint identity, flow directionality, and network structure. Systems that can process raw traffic, even in aggregated form, have access to richer signals.

Second, task decomposition may be more effective than a single universal classifier. Topology-heavy attacks (DDoS, DoS, Recon) and protocol-heavy attacks (MQTT, Spoofing) respond differently to graph-based modeling. A hybrid architecture — with a tabular first stage and a graph-based second stage for topology-sensitive cases — may be more robust than either approach alone.

Third, evaluation must reflect deployment conditions. Session-aware splits, proper validation, and multi-seed reporting are not optional refinements; they are prerequisites for trustworthy performance claims.

We note that our study is conducted entirely on the CICIoMT2024 benchmark. We have designed a testbed architecture using ESP32-S3, Raspberry Pi 5, and medical-grade sensors for future validation under live network conditions, but testbed results are outside the scope of this paper.

### 5.5. Limitations

Several limitations should be noted. First, MQTT and Spoofing have very small support in the communication-topology formulation (8 and 35 training graphs respectively, with 4 test graphs each). Per-class results for these categories carry high variance and should be interpreted cautiously. Second, the PCAP-to-graph pipeline requires raw packet captures, which may not be available in all deployment settings; privacy and storage constraints may limit PCAP retention. Third, the 60-second window duration was selected as a practical compromise and its sensitivity was not formally evaluated. Fourth, our communication-topology graphs aggregate all traffic within a window into a single snapshot; temporal evolution within the window is not modeled. Fifth, the domain-typed edges rely on known gateway IPs and subnet structure, which may not generalize to networks with different topologies. Finally, all experiments are conducted on a single dataset; generalization to other IoMT benchmarks or real-world networks remains to be established.

---

## 6. Conclusions

This study investigated how data representation, graph construction, and evaluation protocol shape graph-based intrusion detection on CICIoMT2024. Three findings stand out. First, representation choice matters more than model architecture: PCAP-derived communication-topology graphs with domain-typed edges made GNNs competitive where feature-similarity graphs did not. Second, evaluation protocol is the single largest factor — proper PCAP-level validation reduced reported macro-F1 by over 16 percentage points compared to naive early stopping. Third, GNN effectiveness is attack-category dependent: topology-heavy attacks benefit from graph modeling, while protocol-heavy and low-support attacks do not.

These findings suggest that IoMT IDS research should prioritize proper evaluation and representation design before pursuing architectural complexity. Future work will validate these results on a physical IoMT testbed and explore temporal graph evolution and protocol-aware representations for non-topology-heavy attack categories.

---

## References

1. Akar, T.; Dutta, N. An interpretable dimensional reduction technique with an explainable model for detecting attacks in Internet of Medical Things devices. *Sci. Rep.* **2025**, *15*, 93404. https://doi.org/10.1038/s41598-025-93404-8

2. Uzen, H.; Turkoglu, M.; Hanbay, D. X-FuseRLSTM: A Cross-Domain Explainable Intrusion Detection Framework in IoT Using the Attention-Guided Dual-Path Feature Fusion and Residual LSTM. *Electronics* **2025**, *14*, 2547.

3. Dadkhah, S.; Neto, E.C.P.; Ferreira, R.; Molokwu, R.C.; Sadeghi, S.; Ghorbani, A.A. CICIoMT2024: A benchmark dataset for multi-protocol security assessment in IoMT. *Internet Things* **2024**, *28*, 101351. https://doi.org/10.1016/j.iot.2024.101351

4. Lo, W.W.; Layeghy, S.; Sarhan, M.; Gallagher, M.; Portmann, M. E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for IoT. In Proceedings of *NOMS 2022—IEEE/IFIP Network Operations and Management Symposium*, Budapest, Hungary, 2022; pp. 1–9.

5. Pujol-Perich, D.; Suárez-Varela, J.; Cabellos-Aparicio, A.; Barlet-Ros, P. Unveiling the potential of graph neural networks for robust intrusion detection. *ACM SIGMETRICS Perform. Eval. Rev.* **2022**, *49*(4), 111–117.

6. Tran, D.H.; Park, M. Enhancing Multi-Class Attack Detection in Graph Neural Network through Feature Rearrangement. *Electronics* **2024**, *13*, 2404. https://doi.org/10.3390/electronics13122404

7. Chen, Y.; Li, X.; Zhang, H. MGF-GNN: A Multi-Granularity Graph Fusion-based Graph Neural Network Method for Network Intrusion Detection. In Proceedings of *GAIIS '25*, ACM, 2025. https://doi.org/10.1145/3728725.3728822

8. Li, J.; Liu, Z.; Wang, X. BS-GAT: A network intrusion detection system based on graph neural network for edge computing. *Cybersecurity* **2025**, *8*, 296. https://doi.org/10.1186/s42400-024-00296-8

9. Yacoubi, M.; Moussaoui, O.; Drocourt, C. Enhancing IoMT Security with Explainable Machine Learning: A Case Study on the CICIoMT2024 Dataset. In *COCIA 2025*; LNNS; Springer: Cham, 2026; vol. 1584. https://doi.org/10.1007/978-3-032-01536-5_38

10. Ali, M.; Naeem, F.; Tariq, M.; Kaddoum, G. A novel adaptive hybrid intrusion detection system with lightweight optimization for enhanced security in internet of medical things. *Sci. Rep.* **2025**, *15*, 31897. https://doi.org/10.1038/s41598-025-31897-z

11. Yildiz, O.; Subasi, A. PAMG-AT: Physiological Adaptive Multi-Graph Attention Network for Stress Detection. *Biomed. Signal Process. Control*, submitted.

12. Wu, Y.; Zang, Z.; Zou, X.; Wei, F.; Liu, Y. Graph attention and Kolmogorov–Arnold network based smart grids intrusion detection. *Sci. Rep.* **2025**, *15*, 8648. https://doi.org/10.1038/s41598-025-88054-9

13. Lekssays, A.; Falah, B.; Aimeur, E. EL-GNN: A Continual-Learning-Based Graph Neural Network for Task-Incremental Intrusion Detection Systems. *Electronics* **2025**, *14*, 2756. https://doi.org/10.3390/electronics14142756

**Supplementary Materials:** Detailed exploratory results are included in the Supplementary Materials.
