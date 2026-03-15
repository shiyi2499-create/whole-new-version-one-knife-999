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
