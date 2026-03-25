# Workflow And TODO (2026-03-26)

## Working Directory Policy
- Active working directory: `/Users/shiyi/备份（mac_vs专用）`
- Frozen archive / paper-facing export: `/Users/shiyi/apple IMU`
- Rule:
  - All ongoing experiments, scripts, quick patches, temporary analysis, and iterative work should happen in the active working directory.
  - Only after a route is considered frozen/stable should it be copied or synchronized into `apple IMU`.
  - `apple IMU` should be treated as a clean reference snapshot for paper writing, handoff, and structured review.

## Current Core Conclusions
- Stage1 is no longer the dominant bottleneck after dense labeling + complete-hit-first posthoc.
- Best automatic pipeline on fair6 currently reaches about `char_top1=0.6404`, `CER=0.3034`.
- CTC under `stage1_bestpred` is already close to the best automatic pipeline, with `CER=0.3146`.
- Pipeline and CTC are now complementary routes, both worth preserving for the paper.

## TODO List From Current Discussion
1. Supplement password lengths
   - Need to define the exact target lengths.
   - Recommended clarification: decide whether this means extending beyond the current `len8/len9/len10` setting, or strengthening the existing 8/9/10 coverage first.
2. Supplement mixed dataset
   - Reasonable and likely useful.
   - Important constraint: preserve clean train/eval session separation and keep fair holdout logic consistent.
3. Build demo
   - Good idea.
   - Recommended to do after finalizing the paper-facing baseline, so the demo reflects a stable route.
4. Multi-user generalization
   - Very important scientifically.
   - This is likely one of the strongest next-step directions after the current single-user route is frozen.
5. Long-duration recording
   - Worth doing, but should be defined clearly.
   - Suggested interpretation: test drift across long recording spans, posture/device shifts, and session aging.
6. Add traditional baseline comparisons
   - High priority.
   - This is especially valuable for the paper because it strengthens the story beyond internal ablations.
7. Defense mechanisms
   - Important, but best done after attack baselines are frozen.
   - Otherwise the target keeps moving.

## Suggested Priority (Current View)
1. Add traditional baseline comparisons
2. Multi-user generalization planning / pilot collection
3. Supplement mixed dataset where coverage is still thin
4. Clarify and, if needed, supplement password lengths
5. Build demo on top of the frozen best route
6. Long-duration recording protocol
7. Defense mechanisms

## Notes For Future Chats
- When discussing experiment results, treat `/Users/shiyi/备份（mac_vs专用）` as the source of truth for ongoing work.
- Treat `/Users/shiyi/apple IMU` as the curated export, not the place for day-to-day iteration.
- If a route is declared frozen, copy both:
  - code/scripts
  - result reports and the dataset usage notes
  into `apple IMU`.
