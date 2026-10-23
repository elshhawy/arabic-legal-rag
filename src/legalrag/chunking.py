"""Split one article into chunks, by model tokens, keeping the original text.

An article is the natural citation unit, so the baseline is NO splitting
(chunk_size == 0). Size-based splitting only exists to be compared against
that baseline in the MLflow experiments.

This module is pure on purpose: it takes token offsets, not a model, so it
can be tested without downloading anything.
"""


def chunk_text(text, offsets, chunk_size, overlap):
    """Return the text pieces for one article.

    offsets: one (start_char, end_char) pair per token of `text`.
    chunk_size: tokens per chunk; 0 means keep the whole article as one chunk.
    overlap: tokens shared by two neighbouring chunks.
    """
    if chunk_size < 0 or overlap < 0:
        raise ValueError("chunk_size and overlap must be >= 0")
    if chunk_size == 0 or len(offsets) <= chunk_size:
        return [text]
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    pieces = []
    start = 0
    while True:
        end = min(start + chunk_size, len(offsets))
        piece = text[offsets[start][0] : offsets[end - 1][1]].strip()
        if piece:
            pieces.append(piece)
        if end == len(offsets):
            break
        start += step
    return pieces