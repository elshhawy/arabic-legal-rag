| Run | chunk_size | overlap | embedding_model | chunks | hit@1 all | hit@3 all | hit@3 ar | hit@3 en |
|---|---|---|---|---|---|---|---|---|
| baseline-whole-article | 0 | 0 | intfloat/multilingual-e5-small | 1093 | 0.30 | 0.35 | 0.60 | 0.10 |
| tokens-256-overlap-32 | 256 | 32 | intfloat/multilingual-e5-small | 1096 | 0.30 | 0.35 | 0.60 | 0.10 |
| tokens-128-overlap-32 | 128 | 32 | intfloat/multilingual-e5-small | 1228 | 0.25 | 0.35 | 0.60 | 0.10 |
| tokens-64-overlap-16 | 64 | 16 | intfloat/multilingual-e5-small | 1900 | 0.30 | 0.45 | 0.80 | 0.10 |
| tokens-32-overlap-8 | 32 | 8 | intfloat/multilingual-e5-small | 3510 | 0.10 | 0.35 | 0.70 | 0.00 |
