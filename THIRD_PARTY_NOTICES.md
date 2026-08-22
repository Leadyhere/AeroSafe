# Third-party research attribution

AeroInspect's compact MMR implementation is an independent adaptation of the method described in:

- Zilong Zhang, Zhibin Zhao, Xingwu Zhang, Chuang Sun, and Xuefeng Chen,
  “Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale
  Reconstruction,” 2023. <https://arxiv.org/abs/2304.02216>
- Official implementation: <https://github.com/zhangzilongc/MMR> (Apache-2.0 code; the AeBAD dataset is
  distributed separately under CC BY 4.0 according to its repository).

The implementation follows the published masked-patch, MAE/ViT, simple-FPN, frozen hierarchical-teacher,
and cosine feature-discrepancy design. It is rewritten for this repository and does not include copied
model weights or dataset files.

Deformable DETR is loaded through Hugging Face Transformers using a SenseTime pretrained checkpoint.
See <https://huggingface.co/docs/transformers/model_doc/deformable_detr>.

The comparison baselines use TorchVision's Faster R-CNN ResNet50-FPN and Wide ResNet50-2 model
implementations and pretrained weights. The PatchCore design follows Roth et al., "Towards Total Recall
in Industrial Anomaly Detection" (2021), <https://arxiv.org/abs/2106.08265>. Baseline code in this
repository is an independent implementation and contains no copied pretrained weights.

Users are responsible for reviewing and complying with every dataset's current terms before training or
redistribution. Dataset images, annotations, and pretrained weights are not included in AeroInspect.
