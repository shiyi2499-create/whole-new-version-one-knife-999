# Password `len_8` Results

This file records the first complete `password`-route result on the dedicated
`len_8` dataset, using the first `100` strings in the current pool.

## Dataset

- path: `data/raw/password/len_8`
- charset: `a-z0-9`
- length: `8`
- current total pool size: `200`
- current grouping: `20 x 10`
- result subset used here: `part 1-10`

## Collection quality

- the first `10` parts were collected successfully
- effective sampling rate remains near `200 Hz`
- successful prompts match the generated password profile
- the dataset is suitable for downstream password-route experiments

## Single-key diagnostic

Held-out diagnostic using the same current Inception password-route trainer:

- `val_best_accuracy = 89.0%`
- `test_top1_accuracy = 85.7%`
- `test_top3_accuracy = 98.7%`
- `test_top5_accuracy = 99.6%`

Interpretation:

- the trainer is strong on isolated-key data
- low password zero-shot accuracy is therefore not explained by an obviously
  weak trainer

## Zero-shot password result

Training source:

- `single_key + boost`

Test source:

- `data/raw/password/len_8`

Metrics:

- `char_top1 = 8.1%`
- `char_top3 = 16.2%`
- `char_top5 = 27.5%`
- `sequence_top100 = 0.0%`
- `CER = 91.9%`

Interpretation:

- direct transfer from isolated-key training to continuous password input is
  weak
- the main issue is domain shift rather than unsupported characters or prompt
  mismatch

## Adaptation result

Protocol:

- baseline from `single_key + boost`
- password adaptation on parts `1-8` (`80` strings)
- held-out test on parts `9-10` (`20` strings)

Metrics:

- `char_top1 = 62.5%`
- `char_top3 = 87.5%`
- `char_top5 = 96.2%`
- `sequence_top100 = 35.0%`
- `CER = 37.5%`

Interpretation:

- password-style adaptation substantially improves attack performance
- the IMU signal contains meaningful information for continuous password input
- the key remaining challenge is not whether signal exists, but how much
  password-style data is required to bridge the domain gap

## Current takeaway

The current best-supported statement is:

> `single_key + boost` alone does not transfer well to continuous password
> input in a zero-shot setting, but a modest amount of password-style
> adaptation strongly improves top-k character accuracy and bounded-guess
> sequence success.

## Expanded `200`-string result

The `len_8` pool was then extended from the first `100` strings to the full
`200`-string pool (`20` parts total).

### Fixed 160/40 split

Protocol:

- baseline from `single_key + boost`
- password adaptation on parts `1-16` (`160` strings)
- held-out test on parts `17-20` (`40` strings)

Metrics:

- `char_top1 = 73.4%`
- `char_top3 = 97.8%`
- `char_top5 = 99.1%`
- `sequence_top100 = 65.0%`
- `CER = 26.6%`

Interpretation:

- adding more password-style data continues to help
- the route remains positive after scaling beyond the initial `100`-string pool

### Multi-split adaptation stability

Protocol:

- total pool: `20` password parts
- repeated random group splits
- per split:
  - `16` parts for password adaptation
  - `4` parts held out for test
- `5` random splits

Summary:

- `char_top1 mean/std = 67.3% / 1.0%`
- `char_top3 mean/std = 91.7% / 1.2%`
- `char_top5 mean/std = 96.8% / 0.6%`
- `sequence_top100 mean/std = 46.5% / 5.1%`
- `CER mean/std = 32.7% / 1.0%`

Interpretation:

- the positive adaptation result is not a one-off lucky split
- the password route remains stable under repeated random group-level partitioning

### Multi-split password-only comparison

Protocol:

- no `single_key + boost`
- train directly on password parts
- repeated random group splits
- per split:
  - `16` parts train
  - `4` parts test
- `5` random splits

Summary:

- `char_top1 mean/std = 66.9% / 2.8%`
- `char_top3 mean/std = 91.0% / 2.3%`
- `char_top5 mean/std = 95.4% / 2.0%`
- `sequence_top100 mean/std = 38.0% / 10.2%`
- `CER mean/std = 33.1% / 2.8%`

Interpretation:

- `password only` is viable, but does not outperform
  `single_key + password adaptation`
- the isolated-key baseline still appears to provide useful prior structure
