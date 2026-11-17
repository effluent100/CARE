# CARE

Structural correction can refine vision-language anomaly maps, but its utility varies across unseen samples under domain shift. Existing zero-shot anomaly detection methods primarily focus on improving anomaly representations or constructing candidate corrections, while the decision of whether a correction should modify the original prediction remains underexplored. To address this problem, Correction Acceptance via Risk Evaluation for Cross-Domain Zero-Shot Anomaly Detection, abbreviated as CARE, is proposed.

CARE separates candidate construction from correction acceptance. A Global Normality Prototype and text-guided residual fusion first construct a semantically constrained structural correction. A lightweight source-trained policy then evaluates uncertainty, spatial dispersion, and inter-map consistency to determine the contribution of the correction to the final anomaly map, without target-domain supervision or adaptation.
