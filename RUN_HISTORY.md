# DATASCAPE 2026 run history and results

Last updated 2026-10-06, after run 32. The detailed log with every number is in the Claude memory folder (`project_baseline_results.md`), this file is the readable summary.

## Task and setup

- Competition: DATASCAPE 2026 Data Mining Competition (Kaggle). Classify package condition images as Aman, Rusak or Lainnya. Metric is macro F1.
- Data: 12,224 training images (8,800 Aman, 3,001 Rusak, 423 Lainnya), 3,057 test images. 52 images in 26 identical-file groups with conflicting labels are dropped, leaving 12,172.
- Leaderboard: 30% public and 70% private (private hidden until close). 3 submissions per day, shared with a teammate. Kaggle auto-picks the best 2 by public score.
- Rules that matter: only organizer data, no manual inference on test images, AI declaration needed in the final notebook (not added yet), admin allows fine-tuning CLIP-style encoders.
- Deadline: Friday Oct 9 or 10, 2026. Deliverables are `NamaTim_Notebook.ipynb` and `NamaTim_Writeup.pdf` (writeup not started).
- Pipeline: timm backbone, letterbox to square, weighted sampler, focal loss (gamma 2, no alpha), AdamW, cosine schedule, 5-fold stratified CV with `SEED=42` (never change it, it seeds the folds), best epoch per fold, test predicted by averaging the 5 fold models with horizontal flip.
- Scores below are pooled OOF macro F1 over all 12,172 images with flip probabilities, unless marked otherwise. Runs 1 to 10 used one 15% validation split and are not comparable to the rest.

## Leaderboard scores (public)

| Submission | OOF | Leaderboard | Note |
|---|---|---|---|
| Run 12 (probable match) | 0.7714 | 0.70350 | focal with alpha |
| Run 13 (probable match) | 0.8293 | 0.74062 | convnext_tiny, no shift aug |
| Run 14 plus Lainnya 3x | about 0.826 | 0.72042 | 15 labels changed |
| Run 14 plus stage 2 size model | about 0.826 | 0.72079 | 160 labels changed |
| Blend 23+26+29 (equal) | 0.9203 | 0.79133 | submitted 2026-10-05 |
| Blend 23+26+29+30+31+32+34 (equal) | 0.9254 | 0.81321 | submitted 2026-10-09, +0.0219 over the 3-way blend with only 66 test labels different, a larger move than the +0.005 OOF gain predicted |
| Blend 30+31+32+34+37+38+39+44 (equal, new models only, `submission-blend-new-models-plus-run44-equal.csv`) | 0.9249 | 0.81781 | submitted 2026-10-10, +0.0046 over the seven-way with 41 test labels different, inside the public noise of about 0.01 |
| Blend 30+31+32+34+37+43+44 (equal, five SigLIP plus both DINO, `submission-blend-siglip5-plus-dino-equal.csv`) | 0.9256 | 0.82283 | submitted 2026-10-10, best own score, +0.0050 over the previous file with 32 labels different and +0.0096 over the seven-way. The scores rose in the order seven-way 0.81321, new models plus DINOv3 0.81781, SigLIP five plus both DINO 0.82283 while OOF stayed flat near 0.925, each step inside the public noise but all pointing the same way, toward the DINO models and away from the older CLIP and ConvNeXt members |
| Blend 30+31+32+34+37+43+44+59+61 (equal, five SigLIP plus four DINO runs, `submission-blend-siglip5-plus-dino4-equal.csv`) | 0.9246 | 0.82328 | submitted 2026-10-10, +0.0005 over the previous file with only 19 labels different, same score within noise. Rusak predicted 999 against 1013 |
| Blend 30+31+32+34+37+43+44+59+61 plus run 55 at weight 2 (`submission-blend-best-plus-run55-x2-equal.csv`) | 0.9248 | 0.82272 | late submission after the close (2026-10-11), does not count for the ranking, -0.0006 against the 0.82328 file with 23 labels different, so adding the DINOv3 large run changed nothing measurable on the public part |

Public leaderboard snapshot 2026-10-09 (30% public, final score combines public 30% and private 70%). The top teams score 0.985, 0.978, 0.965, 0.959 and 0.946. Our team is shown at 22nd with 0.84492, which is a teammate's anchored file and not ours (see risks), and our best own submission is the seven-way blend at 0.81321.

Lainnya check (2026-10-09, no test labels). The seven-way blend predicts 1.8% Lainnya and 34.4% Rusak on test against 3.5% and 24.4% in train. Train Lainnya sits in one embedding cluster (cluster 28 holds 309 of the 423 Lainnya, 82% of that cluster is Lainnya) and the test set has half as many images in it (48, 1.6% against 3.1%), all 48 predicted Lainnya. Applying train class rates inside each cluster to the test cluster counts gives about 46 Lainnya and 1056 Rusak expected, against 56 and 1052 predicted. So the lower Lainnya share and higher Rusak share are explained by the test cluster mix and Lainnya is not hidden. It is not a main cause of the leaderboard gap (this does not rule out an unseen Lainnya type sitting in other clusters).

The gap between OOF and leaderboard is large (0.09 for run 13, 0.13 for the 3-way blend, 0.112 for the seven-way blend, so it shrank with the SigLIP models). Higher OOF still goes with a higher leaderboard score, so the direction is right even if the size is not.

## Run history

| Run | Backbone and change | Platform | OOF macro F1 | Result |
|---|---|---|---|---|
| 1 to 4 | convnext_tiny, 15 to 50 epochs, early stop tuning | Kaggle | 0.7907 to 0.8381 (single split) | run 4 looked best but was a lucky unseeded roll |
| 5, 6 | mixup and cutmix | Kaggle | 0.7245, 0.6860 | failed, dropped |
| 7 | weight decay 5e-2 | Kaggle | 0.7683 | not worse than the true baseline once nondeterminism was fixed |
| 8 to 10 | same config, seeding and cuDNN determinism fixes | Kaggle | 0.7631, 0.7485, 0.7485 | reproducible baseline is 0.7485, run 10 is byte-identical to run 9 |
| 11 | 5-fold CV, class-weighted CE | Kaggle | 0.7948 | first honest estimate |
| 12 | plus focal loss with alpha | Kaggle | 0.7714 | worse |
| 13 | focal loss gamma 2, no alpha | Kaggle | 0.8293 | best of the early runs |
| 14 | plus drop path 0.1 | Kaggle | 0.8258 | no gain |
| 15 | plus shift augmentation and image cache | Kaggle | 0.7965 | shift aug cost about 0.03 at LR 3e-4 |
| 16 | dedup, convnext_tiny, shift aug | Kaggle | 0.7979 | control for run 17 |
| 17 | efficientnet_b0, same setup | Kaggle | 0.8716 | +0.074 over convnext at LR 3e-4 |
| 18 | efficientnet_b0 Noisy Student weights | Modal | 0.8673 | no gain |
| 19 | efficientnet_b0 at 288 px | Modal | 0.8758 | within noise of run 17 |
| 20 | convnext_tiny at LR 5e-5, no shift aug | Modal | 0.8952 | LR was the problem, big jump |
| 21 | convnext_tiny at LR 3e-5 | Modal | 0.9008 | small gain |
| 22 | run 21 with shift aug | Modal | 0.8962 | shift aug now costs only 0.005 |
| 23 | convnext_small at LR 3e-5 | Modal | 0.9078 | best convnext |
| 24 | DINOv2 small, layer decay 0.75, LR 5e-5 | Kaggle | 0.8989 | different family, no score gain |
| 25 | efficientnet_b0 at low LR 5e-5 | Modal | 0.7749 | failed, effnet needs a high LR |
| 26 | CLIP ViT-B/16 (OpenAI), LR 3e-5, layer decay 0.75 | Kaggle | 0.9131 | best single until run 30 |
| 27 | convnext_small seed 1 on the laptop | laptop | none | stopped, about 40 h projected on a GTX 1650 Ti |
| 28 | CLIP ViT-B/16 LAION-2B weights | Kaggle | 0.9096 | within noise of run 26 |
| 29 | run 26 with shift aug on | Kaggle | 0.9091 | within noise, test calls moved toward Rusak |
| 30 | SigLIP ViT-B/16 (224 px) | Kaggle | 0.9153 | best single until run 32 (0.9167 without TTA) |
| 31 | SigLIP ViT-B/16 at 256 px | Modal | 0.9136 | no gain from resolution |
| 32 | SigLIP 2 ViT-B/16 (224 px) | Kaggle | 0.9183 | best single model (0.9186 without TTA) |
| 33 | SigLIP 2 ViT-B/16, seed offset 1 | shared L40 server | cancelled | stopped about 11 minutes in, GPU 1 ran 124 s per epoch and GPU 2 about 312 s because other users share the GPUs |
| 34 | SigLIP 2 ViT-B/16, seed offset 1 | shared L40 server (3 GPUs) | 0.9164 | second seed of run 32, same 30 epochs, about 2.7 hours, per-class F1 0.9523/0.9361/0.8608 for Aman/Lainnya/Rusak, Rusak recall 0.8336 |
| 35 | SigLIP 2 ViT-B/16, shift augmentation on, seed offset 1 | shared L40 server (3 GPUs) | 0.9160 | same as run 34 except USE_SHIFT_AUG, per-class F1 0.9515/0.9385/0.8581 for Aman/Lainnya/Rusak, Rusak recall 0.8363, no OOF cost, test labels agree 97.4% with run 34 so the augmentation barely changed what it predicts on test |
| 36 | SigLIP 2 ViT-B/16, seed offset 2 | shared L40 server (3 GPUs) | 0.9183 | third seed of SigLIP 2, same OOF as run 32, 2.48 hours, per-class F1 0.9528/0.9402/0.8618, Rusak recall 0.8323, 77% of its OOF errors are shared with run 32 so it adds almost nothing to a blend (6 test labels change in the best file) |
| 37 | SigLIP v1 ViT-B/16 (224 px), seed offset 1 | shared L40 server (3 GPUs) | 0.9177 | second seed of run 30 (0.9153), about 2.85 hours, per-class F1 0.9518/0.9443/0.8570 for Aman/Lainnya/Rusak, Rusak recall 0.8242, test labels agree 98.1% with run 30 |
| 38 | CLIP ViT-B/16 DFN (`vit_base_patch16_clip_224.dfn2b`) | shared L40 server (3 GPUs) | 0.9117 | 2.83 hours, per-class F1 0.9500/0.9335/0.8515 for Aman/Lainnya/Rusak, Rusak recall 0.8131, weakest of the recent ViTs, 71% of its OOF errors are also run 32 errors (76% for run 34 against run 32) |
| 39 | ConvNeXt base CLIP (`convnext_base.clip_laion2b_augreg_ft_in12k_in1k`, 256 px, no layer decay) | shared L40 server (3 GPUs) | 0.9116 | about 3.8 hours from start to end, per-class F1 0.9503/0.9293/0.8552 for Aman/Lainnya/Rusak, Rusak recall 0.8229, most different errors so far (only 66% of its OOF errors are also run 32 errors), pairs well with SigLIP (30+39 0.9226) |
| 40 | CLIP ViT-B/16 MetaCLIP (`vit_base_patch16_clip_224.metaclip_2pt5b`) | shared L40 server (3 GPUs) | 0.9138 | 2.53 hours, per-class F1 0.9517/0.9317/0.8579 for Aman/Lainnya/Rusak, Rusak recall 0.8208, pairs 32+40 0.9246 |
| 41 | CLIP ViT-B/16 `laion2b_ft_in12k_in1k` | shared L40 server (3 GPUs) | 0.9003 | weakest of the recent ViTs, per-class F1 0.9460/0.9136/0.8413 for Aman/Lainnya/Rusak, Rusak recall 0.8044, predicts fewer Rusak on test (1017), the ImageNet fine-tuning of CLIP did not help here |
| 42 | CLIP ViT-B/16 DataComp (`vit_base_patch16_clip_224.datacompxl`) | shared L40 server (3 GPUs) | 0.9165 | 3.01 hours, best CLIP so far (run 26 was 0.9131), per-class F1 0.9519/0.9402/0.8575, Rusak recall 0.8205, pairs 34+42 0.9250 |
| 43 | DINOv2 ViT-B/14 (`vit_base_patch14_dinov2.lvd142m`, 224 px) | shared L40 server (3 GPUs) | 0.9101 | per-class F1 0.9533/0.9150/0.8618 for Aman/Lainnya/Rusak, Rusak recall 0.8205, predicts only 924 Rusak on test (fewest of any run, like DINOv3 with 981), agrees 94.3% with the seven-way blend, 69% of its OOF errors are run 32 errors |
| 44 | DINOv3 ViT-B/16 (`vit_base_patch16_dinov3.lvd1689m`, 224 px) | shared L40 server (3 GPUs) | 0.9048 | per-class F1 0.9454/0.9248/0.8443 for Aman/Lainnya/Rusak, Rusak recall 0.8272, most different test predictions of any run (agrees only 94.3% with the submitted blend, predicts 981 Rusak against about 1050), only 58% of its OOF errors are run 32 errors |
| 45 | CLIP ViT-B/16 OpenAI fine-tuned on ImageNet-12k then 1k (`vit_base_patch16_clip_224.openai_ft_in12k_in1k`) | shared L40 server (3 GPUs) | 0.8944 | weakest run since run 25, per-class F1 0.9453/0.9005/0.8375 for Aman/Lainnya/Rusak, Rusak recall 0.7987, like run 41 the ImageNet fine-tuning of CLIP hurts and lowers every blend it joins |
| 46 | SigLIP 2 ViT-B/16, LR 5e-5 (default 3e-5), seed offset 1 | shared L40 server (3 GPUs) | 0.9184 | tuning run against run 34 (0.9164), +0.0020 is inside the seed noise of about 0.002, per-class F1 0.9532/0.9396/0.8623, 2.47 hours |
| 47 | SigLIP 2 ViT-B/16, LR 2e-5, seed offset 1 | shared L40 server (3 GPUs) | 0.9156 | tuning run against run 34 (0.9164), -0.0008 is inside the noise, 2.48 hours, so LR 2e-5 to 5e-5 all give about the same |
| 48 | SigLIP 2 ViT-B/16, layer decay 0.85 (default 0.75), seed offset 1 | shared L40 server (3 GPUs) | 0.9149 | tuning run against run 34 (0.9164), -0.0015 is inside the noise, 2.46 hours, Rusak F1 0.8570, test 1962/56/1039 |
| 49 | SigLIP 2 ViT-B/16, drop path 0.1, seed offset 1 | shared L40 server (3 GPUs) | 0.9112 | tuning run against run 34, -0.0052 is beyond the noise so drop path hurts, 2.47 hours, Rusak F1 0.8503 |
| 50 | SigLIP 2 ViT-B/16, plain cross-entropy with class weights instead of focal loss, seed offset 1 | shared L40 server (3 GPUs) | 0.8788 | -0.0376 against run 34, Lainnya F1 falls to 0.8420 (recall 0.9385, precision collapses) because the class weights come on top of the weighted sampler, test calls shift to 67 Lainnya and 1121 Rusak, 2.48 hours |
| 51 | SigLIP 2 ViT-B/16, weight decay 0.05 (default 0.01), seed offset 1 | shared L40 server (3 GPUs) | 0.9173 | tuning run against run 34, +0.0009 is inside the noise, 2.48 hours, 78% of its errors are shared with run 34 |
| 52 | SigLIP 2 ViT-B/16, 15 epochs instead of 30, seed offset 1 | shared L40 server (3 GPUs) | 0.9148 | -0.0016 against run 34 for half the compute (1.24 hours), inside the noise, so 15 epochs is almost as good |
| 55 | DINOv3 ViT-L/16 (`vit_large_patch16_dinov3.lvd1689m`, 224 px, batch 16x2, layer decay 0.85) | shared L40 server (3 GPUs) | 0.9185 | 8.93 hours, best single model by a hair (run 32 is 0.9183), the first DINO model to match SigLIP, per-class F1 0.9523/0.9412/0.8619, best Rusak recall of any run (0.8383), predicts only 941 Rusak on test (fewest of any run), agrees 95.6% with the best file |
| 58 | SigLIP 2 base at 384 px (`vit_base_patch16_siglip_384.v2_webli`, IMG_SIZE 384, seed offset 1) | shared A100 (Jupyter, 1 GPU) | 0.9190 | per-class Rusak F1 0.8697, test 2009/56/992, public 0.82228 (late submission), higher resolution gave +0.0026 over the 224 px seeds, inside the noise |
| 59 | DINOv3 ViT-B/16, seed offset 2 | shared L40 server (3 GPUs) | 0.9037 | second seed of run 44 (0.9048), per-class F1 0.9453/0.9229/0.8429, Rusak recall 0.8225, test counts 2002/57/998 so it repeats run 44's low Rusak count, 58% of its OOF errors are run 32 errors |
| 60 | ConvNeXt base distilled from DINOv3 (`convnext_base.dinov3_lvd1689m`, 224 px, no layer decay) | shared L40 server (3 GPUs) | 0.9020 | per-class F1 0.9492/0.9053/0.8516, Rusak recall 0.8131, test counts 1960/55/1042 so it does not share the DINO models' low Rusak count, below the other ConvNeXt (run 39 at 0.9116), lowers the blends it joins |
| 61 | DINOv2 ViT-B/14, seed offset 2 | shared L40 server (3 GPUs) | 0.9111 | second seed of run 43 (0.9101), per-class F1 0.9506/0.9264/0.8563, Rusak recall 0.8323 (highest of the recent runs), test counts 2012/54/991 so it repeats the low Rusak count of the DINO ViTs, 65% of its OOF errors are run 32 errors |
| 62 | PE core ViT-B/16 (`vit_pe_core_base_patch16_224.fb`, mean and std 0.5, batch 16x2, seed offset 1) | shared A100 (Jupyter, 1 GPU) | 0.9234 | best single run so far (previous best 0.9185), 1.59 hours on the A100, per-class F1 0.9568/0.9405/0.8729 (best Rusak F1 of any run), test 1985/54/1018, public score 0.81602 (late submission), files copied to arashi |
| 63 | EVA02 base CLIP (`eva02_base_patch16_clip_224.merged2b`, CLIP mean and std, batch 16x2, seed offset 1) | shared A100 (Jupyter, 1 GPU) | 0.9163 | per-class F1 0.9518/0.9388/0.8584, test 1982/55/1020, public score 0.81682 (late submission), adds +0.0026 to the 30+31+32+34+43 blend (CI +0.0007 to +0.0046) |
| 64 | SigLIP 2 base with GAP head (`vit_base_patch16_siglip_gap_224.v2_webli`, mean and std 0.5, seed offset 1) | shared A100 (Jupyter, 1 GPU) | 0.9205 | second best single, per-class F1 0.9545/0.9402/0.8669, test 1967/55/1035, public score 0.81321 (late submission), 70% of its OOF errors are also run 34 errors |
| 65 | DINOv3 ViT-B/16 EUPE (`vit_base_patch16_dinov3_qkvb.eupe_lvd1689m`, img_size 224, seed offset 1) | shared A100 (Jupyter, 1 GPU) | 0.9049 | like the other DINOv3 base run (0.9048), per-class F1 0.9441/0.9298/0.8407, test 1924/55/1078 (the most Rusak of the new runs), public score 0.81867 (late submission) |
| 66 | DINOv2 ViT-B/14 with 4 registers (`vit_base_patch14_reg4_dinov2.lvd142m`, img_size 224, seed offset 1) | shared A100 (Jupyter, 1 GPU) | 0.9168 | 1.05 hours, per-class F1 0.9531/0.9322/0.8651, test 2028/56/973 (the fewest Rusak of any recent run), public score **0.82953** (late submission, second best own file after run 43 at 0.83349) |
| 67 | SigLIP 2 base at 256 px (`vit_base_patch16_siglip_256.v2_webli`, IMG_SIZE 256) | shared A100 (Jupyter, 1 GPU) | 0.9202 | Rusak F1 0.8657, test 1997/55/1005, public 0.82165 (late submission) |
| 68 | SigLIP 2 ViT-B/16, focal gamma 1 (default 2), seed offset 1 | shared L40 server (3 GPUs) | 0.9165 | tuning run against run 34, +0.0001 so no effect, 2.45 hours |
| 69 | SigLIP 2 ViT-B/16, focal loss with class weights (alpha), seed offset 1 | shared L40 server (3 GPUs) | 0.8660 | -0.0504 against run 34, Lainnya F1 falls to 0.8056, same cause as run 50 (weights on top of the weighted sampler), test calls shift to 87 Lainnya and 1108 Rusak, 2.47 hours |
| 70 | SigLIP 2 ViT-B/16, layer decay 0.65 (default 0.75), seed offset 1 | shared L40 server (3 GPUs) | 0.9176 | tuning run against run 34 (0.9164), +0.0012 is inside the noise, Rusak F1 0.8575, test 1930/55/1072 |
| 71 | SigLIP 2 ViT-B/16, batch size 16 with no accumulation (default 32), seed offset 1 | shared L40 server (3 GPUs) | 0.9153 | tuning run against run 34, -0.0011 is inside the noise, more optimizer steps did not help and the run was slower (3.19 hours), Rusak F1 0.8524 |
| 72 | SigLIP 2 ViT-B/16, 45 epochs (default 30), seed offset 1 | shared L40 server (3 GPUs) | 0.9178 | tuning run against run 34 (0.9164), +0.0014 is inside the noise, so 50% more epochs did not help, 3.67 hours, Rusak F1 0.8601, test 1973/54/1030, public 0.81682 (late submission) |
| 73 | SigLIP 2 ViT-B/16, LR 5e-5, layer decay 0.85, drop path 0.1, seed offset 1 | shared L40 server (3 GPUs) | 0.9125 | stacked changes did worse than the default (-0.0039), 3.27 hours, Rusak F1 0.8486, test 1977/54/1026, public 0.81842 (late submission) |
| 77 | DINOv3 large (`vit_large_patch16_dinov3.lvd1689m`), seed offset 2 | shared L40 server (3 GPUs) | 0.9200 | second seed of run 55 (0.9185), 11.80 hours, Rusak F1 0.8616, test 2068/55/934 (the fewest Rusak of any single file), public **0.82914** (late submission, best DINOv3 file, run 55 got 0.82379), adds +0.0022 to the 7-way blend (CI +0.0001 to +0.0044) |
| 78 | DINOv2 ViT-B/14, seed offset 3 | shared A100 (Jupyter, 1 GPU) | 0.9145 | seed repeat of run 43, Rusak F1 0.8610, test 2041/54/962, public **0.82902** (late submission) |
| 79 | DINOv2 ViT-B/14, seed offset 4 | shared A100 (Jupyter, 1 GPU) | 0.9171 | seed repeat of run 43, Rusak F1 0.8658, test 2042/53/962, public 0.82665 (late submission) |
| 80 | DINOv3 small (`vit_small_patch16_dinov3.lvd1689m`) | shared A100 (Jupyter, 1 GPU) | 0.8700 | weakest ViT in a long time, Rusak F1 0.7910, test 1851/56/1150 (the most Rusak of any run), public 0.78981 (late submission), only 40% of its OOF errors shared with run 43 |
| 81 | EVA02 base MIM (`eva02_base_patch14_224.mim_in22k`, layer decay 0.65) | shared A100 (Jupyter, 1 GPU) | 0.9140 | Rusak F1 0.8565, test 1906/55/1096, public 0.79965 (late submission), the second most Rusak and the second lowest public score |

## Blends (equal weights on flip probabilities)

| Blend | OOF | Note |
|---|---|---|
| 14+15 | 0.8278 | early shift aug blend |
| 17+18 | 0.8808 | efficientnet pair |
| 23+21 | 0.9106 | convnext pair |
| 23+24 | 0.9116 | convnext plus DINOv2 |
| 23+26 | 0.9185 | |
| 23+26+28 | 0.9189 | |
| 23+26+29 | 0.9203 | submitted, leaderboard 0.79133 |
| 23+26+29+30 | 0.9225 | |
| 28+29+30 | 0.9232 | picked from a search, optimistic |
| 23+29+30+31 | 0.9248 | picked from a search, optimistic |
| 26+29+30+32 | 0.9261 | picked from a search, optimistic |
| 23+26+29+30+31 | 0.9258 | all members, not searched, +0.0054 over the submitted blend (95% CI +0.0024 to +0.0087) |
| 23+26+29+30+31+32 | 0.9253 | all members, adding run 32 does not help (-0.0005 against the five-way) |
| 23+26+29+30+31+34 | 0.9255 | run 34 in place of run 32 |
| 23+26+29+30+31+32+34 | 0.9254 | seven-way, -0.0004 against the five-way (95% CI -0.0021 to +0.0012), +0.0050 over the submitted blend (CI +0.0014 to +0.0084) |
| 32+34 | 0.9221 | two seeds of SigLIP 2 together, +0.0038 over run 32 alone |
| 26+29+30+32+34 | 0.9265 | searched, optimistic |

Adding run 35 to the submitted seven-way blend gives 0.9251 (10 test labels change), 26+29+30+31+32+34+35 gives 0.9261. Adding run 37 to the seven-way gives 0.9242 (-0.0012, 95% CI -0.0027 to +0.0001, 8 test labels change), and 30+31+32+34+37 gives 0.9250. Adding run 38 to the seven-way gives 0.9254 (no change, CI -0.001 to +0.001, 13 test labels change), and 30+31+32+34+37+38 gives 0.9257. Adding run 39 to the seven-way gives 0.9250 (-0.0003, CI -0.0016 to +0.0010, 15 test labels change), adding 38 and 39 gives 0.9256, and 30+31+32+34+37+38+39 gives 0.9252. Adding run 41 to the seven-way gives 0.9250 (-0.0004, CI -0.0013 to +0.0005), and the 12-way blend of every run since 23 except 28 gives 0.9255. Adding run 44 to the seven-way gives 0.9256 (+0.0002, CI -0.0020 to +0.0021, 19 test labels change). Adding run 43 to the seven-way gives 0.9251 (-0.0003, CI -0.0018 to +0.0009, 28 test labels change), and the new-models plus both DINO runs (30+31+32+34+37+38+39+44+43) gives 0.9245. Adding run 40 to the seven-way gives 0.9262 (+0.0008, CI -0.0007 to +0.0025, 12 test labels change) and run 42 gives 0.9252 (-0.0002). The 16-way of every run since 23 except 28 gives 0.9258. Pairs of one SigLIP 2 run and one of the new CLIP runs already reach 0.925 (34+42 0.9250, 32+40 0.9246), as high as the large blends. Adding run 45 lowers the seven-way to 0.9245 (-0.0009, CI -0.0018 to -0.0001) and the new-models blend with both DINO runs to 0.9239. Adding run 59 to the seven-way gives 0.9257 (+0.0003) and to the SigLIP five plus both DINO blend 0.9256 (no change, 13 test labels differ from the 0.82283 file). Adding run 60 to the SigLIP five plus both DINO blend gives 0.9245 (-0.0011, 15 test labels differ from the 0.82283 file). Adding run 61 to the SigLIP five plus both DINO blend gives 0.9261 (+0.0005, 8 test labels differ from the 0.82283 file), and with run 59 as well (four DINO runs) 0.9246. Adding run 55 (DINOv3 large) to the SigLIP five plus four DINO blend gives 0.9257 (+0.0011, 14 test labels differ from the 0.82328 file), and with run 55 at weight 2 gives 0.9248 (23 labels differ). Run 34 adds nothing to the blend (test labels agree 97.5% with run 32). Candidate files not submitted yet are `submission-blend-run23-run26-run29-run30-run31-run32-equal.csv` (1947/1054/56 for Aman/Rusak/Lainnya, 58 labels differ from the submitted blend), `submission-blend-run23-run26-run29-run30-run31-equal.csv` (45 differ) and `submission-blend-run23-run26-run29-run30-run32-equal.csv`, `submission-blend-run23-run26-run29-run30-run31-run32-run34-equal.csv` (1949/1052/56, 66 labels differ from the submitted blend) and `submission-blend-run23-run26-run29-run30-run31-run34-equal.csv`. Run 34 alone is `submission-5fold-focal-siglip2-b16-seed1-run34.csv`. The blend has saturated near 0.925 OOF.

## Late submissions (2026-10-11, after the close, public score only, private still blank)

Sent to learn from, they do not count for the ranking. Scores are public (about 30% of test, noise about 0.01).

| File | Public score |
|---|---|
| `submission-blend-searched-run32-run37-run42-run35-run26-equal.csv` | 0.81701 |
| `submission-blend-all-runs-equal.csv` | 0.81621 |
| `submission-blend-siglip5-dino4-x2-equal.csv` | 0.82373 |
| `submission-blend-best-plus-run55-equal.csv` | 0.82210 |
| `submission-run42-clip-datacompxl.csv` | 0.81283 |
| `submission-run43-dinov2-b14.csv` | 0.83349 |
| `submission-run44-dinov3-b16.csv` | 0.81462 |
| `submission-run55-dinov3-large.csv` | 0.82379 |
| `L6-best-blend-rusak-x1.10.csv` | 0.82202 |
| `L4-best-blend-rusak-x0.90.csv` | 0.82210 |
| `L3-family-clip-only-5.csv` | 0.80828 |
| `submission-run46-siglip2-lr5e-5.csv` | 0.81781 |
| `submission-run36-siglip2-seed2.csv` | 0.82707 |
| `submission-5fold-focal-siglip2-b16-run32.csv` | 0.82527 |
| `L5-best-blend-rusak-x0.80.csv` | 0.82500 |
| `L1-family-siglip-only-7.csv` | 0.81682 |
| `L2-family-dino-only-5-with-run55.csv` | 0.82500 |
| `submission-blend-best-plus-run55-x2-equal.csv` | 0.82272 |

Findings.

- Three SigLIP 2 base runs with almost the same OOF (0.9183, 0.9183, 0.9184) scored 0.8253 (run 32), 0.8271 (run 36) and 0.8178 (run 46), so the public noise between equal models is about 0.009.
- OOF does not predict the public score here. Spearman between OOF and public is -0.26 over 18 files and 0.08 over the 8 single runs. DINOv2 base (OOF 0.9101) scored 0.8335, the best of all our files, while DINOv3 base (OOF 0.9048) scored 0.8146.
- Blending did not help on the public part. Single runs scored 0.813 to 0.833 and blends 0.816 to 0.824, even though blends gain about 0.007 OOF.
- The number of Rusak predicted on test tracks the public score. Spearman is -0.69 over 21 files, fewer predicted Rusak means a higher score (924 Rusak scored 0.8335, 1055 scored 0.8083). It is confounded with model family (the DINO models predict fewer Rusak) and public noise, so it is a lead and not proof.
- Run 50 (plain cross-entropy with class weights, OOF 0.8788, 1121 Rusak and 67 Lainnya predicted) scored 0.80251, the lowest public score of all our files. It fits the Rusak trend but its Lainnya over-prediction also lowers the score, so it is not a clean test.
- Two more tuning runs were submitted late, run 70 (layer decay 0.65, 1072 Rusak) scored 0.81634 and run 71 (batch size 16, 1010 Rusak) scored 0.82283. Again the one with more predicted Rusak scored lower, while their OOF values (0.9176 and 0.9153) point the other way.
- Run 62 (PE core base, the best OOF single at 0.9234, 1018 Rusak predicted) scored 0.81602 publicly, below DINOv2 base (0.8335, OOF 0.9101) and below the SigLIP 2 runs (0.8253 and 0.8271). The best OOF model is not the best on the public part, as with the other files.
- Family only blends scored DINO 0.8250 (959 Rusak), SigLIP 0.8168 (1033) and CLIP 0.8083 (1055), the opposite order of their OOF.
- Rescaling the Rusak probability of the 0.82328 blend by 0.8, 0.9 and 1.1 gave 0.8250, 0.8221 and 0.8220, a difference inside the noise.
- The answer key, once released, can settle whether the models over-predict Rusak on test (we predict about 34% against 24% in train).

- 2026-10-12, A100 runs fetched and submitted: run 66 DINOv2 reg4 0.82953 (973 Rusak), run 65 DINOv3 EUPE 0.81867 (1078 Rusak), run 63 EVA02 base CLIP 0.81682 (1020), run 64 SigLIP 2 GAP 0.81321 (1035) while its OOF 0.9205 was the second best. Again the file with the fewest predicted Rusak scored highest and the order of OOF did not predict the public order.
- The A100 runs list the training files in a different order from the L40 and Kaggle runs (unsorted iterdir), so their folds differ and their OOF rows must be re-aligned by file name before blending (done in a scratch copy, the saved files are unchanged). Each run's own OOF is still valid.

- 2026-10-12, four new blends from the A100 and DINOv2 files (equal weights, files in `late-probes/`): 32+34+43+62+63+64+66 OOF 0.9312, 987 Rusak, public **0.82854**. 43+66 OOF 0.9240, 947 Rusak, public 0.82784. 43+66+62 OOF 0.9300, 971 Rusak, public 0.82473. 30+31+32+34+43+62+63+66 OOF 0.9298, 1001 Rusak, public 0.82427. None beat the best single file (run 43, 0.83349), and the OOF gain of about 0.005 from adding PE core (62) did not show on the public part, as with every earlier blend. The two-model DINOv2 blend scored about as high as the seven-model one.

- 2026-10-13, A100 runs 58, 67, 78, 79, 80, 81 submitted: DINOv2 seed 3 0.82902 and seed 4 0.82665 (both 962 Rusak) against run 43 0.83349 and run 61 (seed 2), so five DINOv2 base seeds now sit between 0.8267 and 0.8335 and the family is consistently the best on the public part. SigLIP 2 at 384 px 0.82228 and 256 px 0.82165, both higher than the 224 px seeds (0.8178 to 0.8271). The two models with the most Rusak predictions scored lowest (DINOv3 small 1150 Rusak 0.78981, EVA02 MIM 1096 Rusak 0.79965).
- Kaggle uploads hang at 0% on this network because IPv6 to storage.googleapis.com times out. Running the CLI with IPv4 forced (`scratchpad/kaggle_v4.py`, patches urllib3 allowed_gai_family) fixed it.

- 2026-10-13, L40 runs 72, 73, 77 submitted: DINOv3 large seed 2 0.82914 (934 Rusak, OOF 0.9200), SigLIP 2 ep45 0.81682, SigLIP 2 stacked tuning 0.81842. The two DINOv3 large seeds scored 0.82379 and 0.82914, a spread of 0.005 on equal recipes, and the one with fewer Rusak scored higher again. More epochs and stacked tuning changes did nothing for SigLIP 2, the default recipe stays at its ceiling.

## What we learned

- Learning rate mattered most. Low LR (3e-5 to 5e-5) lifted convnext from 0.80 to 0.90, and efficientnet is the opposite and needs a high LR.
- Pretrained family matters next. CLIP, SigLIP and SigLIP 2 ViT-B/16 with layer decay 0.75 beat convnext_small by about 0.005 to 0.010 (SigLIP 2 is the best single at 0.9183), and DINOv2 small matched convnext_tiny.
- Resolution, Noisy Student weights, drop path, multi-scale TTA and larger weight decay gave nothing measurable. Mixup and cutmix hurt.
- Shift augmentation costs 0.03 at high LR and about 0.005 at low LR, and its effect on test calls is not predictable (it moved calls toward Rusak for CLIP and away from Rusak for convnext).
- Tuning SigLIP 2 base against run 34 (0.9164), nothing helped beyond noise. LR 2e-5 to 5e-5, layer decay 0.65 and 0.85, weight decay 0.05, focal gamma 1, batch size 16 and 15 epochs (half the compute) all stayed within 0.002. Drop path 0.1 hurt by 0.005. Class weights in the loss (plain cross-entropy or focal with alpha) hurt by 0.04 to 0.05 because the weighted sampler already balances the classes and Lainnya precision collapses.
- Single-run differences under about 0.005 are noise (per-fold standard deviation is about 0.01). Blends have saturated near 0.925, run 32 added nothing to the five-way blend.
- Rusak is the weak class (F1 about 0.85, recall about 0.82). Aman is about 0.95 and Lainnya about 0.94.
- Training has real run-to-run nondeterminism unless the DataLoader seeding and cuDNN determinism fixes are on (they are now).

## Why the leaderboard is lower than OOF (diagnostics, no test labels used)

- Test images differ from train in format (test is 19.5% WEBP against 11.2%), JPEG quality (66% of test JPEGs sit in quality 70 to 78 against 30% of train), file size and dimensions.
- Reweighting the OOF to the test mix of format, quality and size raises it (blend 0.9203 to 0.9415), so this composition shift does not explain the gap.
- A CLIP embedding classifier tells train from test with AUC 0.87, so the content differs. Test is not farther from train than train is from itself, and near-duplicate overlap is the same (23% against 22%). OOF accuracy is flat across test-likeness levels, so measurable covariate shift does not explain the gap either.
- Possible error pockets are clusters 25, 19 and 17 (about 15% of test) where model confidence drops, and Lainnya, which is predicted for only 1.8% of test against 3.5% of train. Boosting Lainnya costs a lot of OOF (10.8x needed to match the train share, OOF falls from 0.92 to 0.83), so it was not pursued.
- The remaining explanation is label related and cannot be checked without test labels.

## Compute and cost

- Kaggle 2xT4 runs 2.3 to 3.0 hours for ViT-B and costs quota only (about 2.5 hours of weekly quota left after run 32). Modal runs 5 folds in parallel on T4s in about an hour, run 31 cost $5.11, and $4.44 of credit is left.
- Laptop GTX 1650 Ti is too slow (about 970 s per epoch for convnext_small, my benchmark was off by about 7x).
- Five lab Mac minis are available (setup planned Oct 6). Timing must be measured with one real epoch before planning, rough guesses range from 2 to 7 hours per ViT-B fold.
- GPU access status (2026-10-09). The shared RTX 3060 is full (no free capacity). The shared L40 server `gpu-labkc` is the one in use now (runs 34 and 35 trained there, three L40s shared with other users). An NVIDIA DGX A100 request is waiting with no answer and is in the queue.

## Risks and open items

- Teammate files with hardcoded test labels of unknown origin score 0.755 to 0.845 on the leaderboard. They are a rules risk and are never used or tuned against here, and Kaggle's auto-pick could select them as the team's final two. Choose the final two submissions manually.
- Next submissions are the seven-way blend (or the six-way) and run 32 alone.
- Kaggle quota is nearly used up. Further runs go to the shared L40 server (`gpu-labkc`, 3 GPUs shared with other users, about 2.7 hours per ViT-B run, queue driver `run_queue.py`), or the Modal credit ($4.44).
- Writeup and notebook cleanup (including the AI declaration) are not done.

## File naming

- Submissions in the project root are named `submission-5fold-focal-<model>-run<N>.csv` for single runs and `submission-blend-run<A>-run<B>-equal.csv` for blends.
- Probabilities are in `archive/probs-run<N>-<model>.zip` or `.npz` with the keys `oof_probs_flip`, `oof_targets`, `oof_filepaths`, `test_probs`, `test_filenames` and `classes`. Compare rows by the last two path parts because Modal and Kaggle prefixes differ.
- `scripts/modal_train.py` runs Modal training (it now supports ViTs through `--layer-decay`, `--norm` and `--multi-scale-tta`), and `scratchpad/` holds the CPU venv and the analysis scripts.
