# Lessons from DATASCAPE 2026

These come from about 60 runs on a 3-class image task (12,172 train images, 3,057 test images, macro F1). The final public score was 0.82328 from an equal-weight blend of nine models, while out-of-fold (OOF) macro F1 sat at 0.925. Every number below is OOF macro F1 unless it says otherwise. The full run table is in `RUN_HISTORY.md` of the original project.

## Set up the validation before training anything

- Use stratified k-fold with a fixed `SEED` and never change it afterwards. The seed decides the folds, and changing it breaks the row alignment between every saved run, so no blend is possible. Vary training randomness with a separate `SEED_OFFSET` instead.
- Save the OOF probabilities, the test probabilities, the OOF file paths and the test file names in one file per run. The blend tool checks the alignment of rows, targets and test order, which catches mixed-up runs immediately.
- Hash every training image before the first fold is built. This project had 52 images in 26 byte-identical groups that carried conflicting labels, and 25% of the train set had near-duplicates. Conflicting copies cannot both be right, so they were dropped. Near-duplicates between folds make OOF optimistic, but test overlapped train by the same amount here, so it did not explain the gap to the leaderboard.
- Sort the file list before building the folds (`sorted(dir.iterdir())`). In this project the A100 listed the images in a different order from the L40 server, so the same `SEED` gave different folds there and the saved OOF rows did not line up. The OOF stayed valid but every blend had to re-align the rows by file name first.
- Make training deterministic early (seeded DataLoader workers, cuDNN determinism). Before this fix, repeated runs differed by more than the effects being compared.
- Treat any single-run difference below about 0.005 as noise. The per-fold standard deviation was about 0.01, and the same recipe with a different seed moved OOF by about 0.002 (SigLIP 2 gave 0.9183, 0.9164 and 0.9183 on three seeds).

## What moved the score

- The learning rate mattered most in the early phase. Dropping ConvNeXt from a high LR to 3e-5 to 5e-5 lifted OOF from about 0.80 to 0.90. EfficientNet went the other way and needed a high LR. Check the LR again whenever the architecture family changes.
- With a ViT and layer-wise LR decay, the learning rate no longer mattered much. SigLIP 2 at 2e-5, 3e-5 and 5e-5 gave 0.9156, 0.9164 and 0.9184, all inside the seed noise.
- The pretraining source mattered more than size or tuning. SigLIP 2 and SigLIP v1 (0.918) beat CLIP variants (0.909 to 0.917), which beat ConvNeXt small (0.908) and DINOv2 or DINOv3 base (0.90 to 0.91).
- CLIP weights that were fine-tuned on ImageNet afterwards were the weakest of the family (0.894 to 0.900 against 0.913 for plain CLIP). Prefer the raw pretrained checkpoint.
- A much larger model gave almost nothing. DINOv3 large took 9 hours and reached 0.9185, while DINOv3 base took about 2.5 hours and reached 0.9048. SigLIP 2 base already matched it at 0.9183. The large model did add diversity to blends, but it was not worth the time as a single model.
- Do not stack class balancing. The weighted sampler already balanced the classes here, and adding class weights to the loss on top cost 0.04 to 0.05 because the rarest class lost precision (its F1 fell from 0.94 to 0.81 to 0.84). Pick one of the two.
- Half the epochs were nearly as good. 15 epochs scored 0.9148 against 0.9164 for 30 on the same recipe, so use short runs for exploring and spend the long ones on final models.
- These gave nothing measurable. Higher resolution (256 px), Noisy Student weights, drop path, multi-scale TTA, larger weight decay and distilling a ConvNeXt from DINOv3. Mixup and cutmix hurt. Shift augmentation cost 0.03 at high LR and about 0.005 at low LR.
- The weak class stayed weak. Rusak had F1 near 0.86 and recall near 0.83, while the other two classes sat at 0.94 to 0.95. Per-class numbers show where the ceiling is better than the macro average.

## Blending

- Equal-weight probability averaging was enough. Blends saturated near 0.925 OOF after about five diverse models, and later additions moved it by 0.001 or less.
- Diversity beats strength. Adding a run is useful when its errors differ from the blend (check the share of OOF errors it shares with the blend), not when its own score is high. A third SigLIP 2 seed shared 77% of its errors with the first and added almost nothing.
- Searching for the best subset on OOF overfits. A searched five-model subset reached 0.9277, but a half-split test showed it did not beat the plain equal-weight blend on rows it had not been selected on. Use a bootstrap interval of the difference against a base blend, and trust it only when the interval clears zero.
- Check the predicted class counts of every blend on test. The DINO models predicted fewer Rusak on test (about 940 to 1000) than the SigLIP and CLIP models (about 1050). Blends with more DINO weight scored higher on the leaderboard (0.813 to 0.823), a lead worth testing but not proof, because the public part had only about 900 images and its noise was near 0.01.

## Why the leaderboard was lower than OOF

Everything measurable was checked and none of it explained the gap. The remaining candidate is something label related, which cannot be tested without test labels.

- Test differed in file format, JPEG quality and size, but reweighting the OOF to the test mix raised OOF instead of lowering it.
- A classifier could tell train from test with an AUC of 0.87, yet test was not farther from train than train was from itself, and OOF accuracy was flat across test-likeness levels.
- The Lainnya class was predicted for 1.8% of test against 3.5% of train. A cluster check showed this followed the different cluster mix of test, so missed Lainnya was not the cause.
- Lesson for next time. Run the adversarial validation (`diagnostics/adversarial_validation_cells.py`) and the duplicate check on day one, and do not spend days on explanations that cannot be tested without test labels.

## What the late submissions showed

After the close, 17 more files were sent to Kaggle as late submissions to learn from.

- OOF did not predict the public score across our files (Spearman -0.26 over 18 files, 0.08 over single runs), and blends that gained 0.007 OOF did not score higher than single runs. Three equal SigLIP 2 runs spread by 0.009 on the public part.
- The number of test images predicted as the minority-heavy class tracked the public score (Spearman -0.69, fewer Rusak scored higher). Models may over-predict that class on test because the test class mix differs from train.
- For next time, compare the predicted class shares on test with the train shares early, and keep the public score noise (about 0.01 for a 900 image public part) in mind before reading anything into small differences. Do not choose between models on OOF gaps below 0.005 or on public gaps below 0.01.

## Rules and integrity

- Read the competition rules for test-time rules before using any shortcut. A teammate's notebook in this project hardcoded test labels of unknown origin, which scored up to 0.845 on the leaderboard. It was never used, never tuned against, and never ticked as a final submission, because Kaggle's automatic pick could have selected it.
- Choose the final submissions by hand on the last day. Do not let the automatic pick decide.
- The final score takes the better of two picks, so make the second pick a different kind of model blend instead of a near copy of the first.

## Compute and process

- On a shared GPU server the GPUs differ in speed because of other users' load (one GPU was 2.4 times slower than the others). Split folds by measured speed, not evenly.
- A queue driver (`run_queue.py`) with one JSON line per run kept the server busy overnight with no one watching. It re-reads the file before every run, skips failures with a flag file, and waits for any run you started by hand. Copy files used by a live driver atomically (copy to a temporary name, then `mv`), because a plain copy can be read half-written.
- Log every run and every leaderboard score in one place right after it finishes. The run history file made it possible to answer "what is our best single run" in seconds.
- Do not keep heavy work on a laptop. Short CPU analysis of saved probability files is fine, training is not.
- Clean up the server and revoke tokens after the last run.
