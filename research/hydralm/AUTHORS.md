# Authors

HydraLM is developed and maintained by:

- **cyh** — creator, lead maintainer, author of the Gated DeltaNet + SWA
  hybrid architecture, the chunkwise delta-rule kernel, the Muon/AdamW
  hybrid optimiser, the `FactBank` associative-memory API, the claim
  reproducer, and the HuggingFace / compiled-decoder deployment adapters.

## Contributors

Community contributors (in chronological order of first merged PR) will be
listed here. If you contribute a patch and are not yet listed, please open
a PR adding yourself to this file.

## Acknowledgements

The HydraLM design stands on prior work we gratefully acknowledge:

- S. Yang et al., *"Gated Delta Networks: Improving Mamba2 with Delta Rule"* (2024)
  — the delta-rule recurrence used by the linear-attention blocks.
- L. Ren et al., *"Samba: Simple Hybrid State Space Models for Efficient
  Unlimited Context Language Modeling"* (2024) — the interleaved SSM/SWA
  schedule that motivates the `swa_every = 4` default.
- N. Lieber et al., *"Jamba: A Hybrid Transformer-Mamba Language Model"*
  (2024) — the hybrid-block scaling analysis.
- S. Arora et al., *"Zoology: Measuring and Improving Recall in Efficient
  Language Models"* (ICLR 2024) — the MQAR benchmark used in claim 2.
- K. Jordan, *"Muon: An optimizer for the hidden layers of neural networks"*
  (2024) — the Newton–Schulz orthogonalised momentum optimiser.
- C. Chen et al., *"Accelerating Large Language Model Decoding with
  Speculative Sampling"* (2023) and Y. Leviathan et al., *"Fast Inference
  from Transformers via Speculative Decoding"* (2023) — the exact
  draft/target decoding algorithm.
- B. Widrow, *"Adaptive 'Adaline' Neuron Using Chemical 'Memistors'"* (1960)
  — the LMS update whose equivalence with the delta rule powers `FactBank`.

Contact: open an issue at <https://github.com/byte271/hydralm/issues>.
