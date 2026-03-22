"""
Vocabulary for frame-level CTC password decoding.

Index 0 = CTC blank (required by torch CTC).
Index 1-26 = a-z
Index 27-36 = 0-9
Index 37 = <unk> (backspace, typo, unrecognised)
"""

VOCAB = ['<blank>'] + list('abcdefghijklmnopqrstuvwxyz0123456789') + ['<unk>']
BLANK_IDX = 0
UNK_IDX = len(VOCAB) - 1
NUM_CLASSES = len(VOCAB)  # 38

CHAR_TO_IDX = {ch: i for i, ch in enumerate(VOCAB)}
IDX_TO_CHAR = {i: ch for i, ch in enumerate(VOCAB)}

IGNORE_KEYS = {'enter', 'return', 'shift', 'capslock', 'tab', 'escape',
               'backspace', 'delete', 'command', 'control', 'option', 'alt',
               'fn', 'space'}


def is_ignored_key(key: str) -> bool:
    k = (key or '').lower().strip()
    return k in IGNORE_KEYS


def char_index(key: str) -> int:
    """Map a key string to vocab index. Returns UNK for unknown keys."""
    k = (key or '').lower().strip()
    if k in IGNORE_KEYS:
        return UNK_IDX
    if k in CHAR_TO_IDX:
        return CHAR_TO_IDX[k]
    if len(k) == 1 and k in CHAR_TO_IDX:
        return CHAR_TO_IDX[k]
    return UNK_IDX
