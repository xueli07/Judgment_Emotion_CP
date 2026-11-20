# Stimulus availability

Facial stimuli derive from the Tsinghua Facial Expression Database:

Yang T, Yang Z, Xu G, et al. (2020). Tsinghua facial expression database: A database of facial expressions in Chinese young and older women and men. PLOS ONE 15(4): e0231304. https://doi.org/10.1371/journal.pone.0231304

Selected Experiment 6 resources are included to support reproducibility: the average face, facial mask, visualization backgrounds, participant-level classification images, and an RData file containing the base image and parameters used to reconstruct the original noise patterns.

Average-face permission has been confirmed. `code/analysis/exp06/input_pic/` contains `average_face.jpg`, `average_face_mask.png`, `thinkstyle1_autoscaled.png`, `thinkstyle2_autoscaled.png` (all four images are 512 × 512), and `exp06_noise_seed.RData`. The `category/` and `continuous/` folders each contain 30 participant CI PNGs. CI PNGs are derived classification images, not original stimulus images.

The shared RData contains `base_faces$chinese` (512 × 512), `base_face_files$chinese` (`average_face.jpg`), `stimuli_params$chinese` (600 × 4092), `n_trials = 600`, `seed = 1`, and `noise_type = sinusoid`. It was renamed from `face_seed_1_time_8月_28_2025_20_13_副本.Rdata` without changing its objects. The JPG cannot replace this noise resource. Object availability alone does not establish successful reconstruction of the manuscript results.

Individual noise-added stimulus face images, other original facial stimulus images, and morphed facial stimulus images are not distributed. Third-party materials and their derivatives remain subject to the applicable source permissions and license terms. Their inclusion does not grant additional rights or automatically license them under CC BY 4.0.
