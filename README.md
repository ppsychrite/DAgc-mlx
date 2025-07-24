## Dialogue Act Graph Classifier (DAgc)

### Overview

An implementation of the model in "Who is Speaking? Speaker-Aware Multiparty Dialogue Act Classification" [1] tested on the Meeting Recorder Dialog Act Corpus [2] in mlx. 

### Changes 
- I use a Graph Attention Network [3] instead of a Relational Graph Attention Network [4] due to the paper only defining one relation type $\(v_i^u, v_{S\(u_i\)}^s\)$ (p. 10125) and defining more only during the ablation study (p. 10129), making RAGT redundant for this use-case.
- ModernBERT is used over RoBERTa as the encoder. 

### References 
[1] [Who is Speaking? Speaker-Aware Multiparty Dialogue Act Classification](https://aclanthology.org/2023.findings-emnlp.678/) (Qamar et al., Findings 2023)

[2] [The ICSI Meeting Recorder Dialog Act (MRDA) Corpus](https://aclanthology.org/W04-2319/) (Shriberg et al., SIGDIAL 2004)

[3] [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (Veličković et al, arXiv 2018)

[4] [Relational Graph Attention Networks](https://arxiv.org/abs/1904.05811) (Busbridge et al., arXiv 2019)
