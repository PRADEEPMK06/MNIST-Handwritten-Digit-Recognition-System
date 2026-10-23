"""
labels.py
---------
Shared label constants for the combined digits + letters model.

Kept dependency-free (no tensorflow/emnist imports) so lightweight scripts
like predict.py don't need to pull in heavy dataset-loading libraries just
to know what class 17 means.

Label scheme:
    label 0-9   -> digits '0'-'9'
    label 10-35 -> letters 'A'-'Z'
"""

NUM_DIGIT_CLASSES = 10
NUM_LETTER_CLASSES = 26
NUM_CLASSES = NUM_DIGIT_CLASSES + NUM_LETTER_CLASSES  # 36

LABEL_MAP = {i: str(i) for i in range(NUM_DIGIT_CLASSES)}
LABEL_MAP.update({NUM_DIGIT_CLASSES + i: chr(65 + i) for i in range(NUM_LETTER_CLASSES)})
