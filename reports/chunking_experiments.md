| Run | chunk_size | overlap | chunks | hit@1 all | hit@3 all | hit@3 ar | hit@3 en | faithfulness | scored / asked | refusal rate | judge |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline-whole-article | 0 | 0 | 1093 | 0.30 | 0.35 | 0.60 | 0.10 | 0.92 | 8 / 20 | 0.60 | gemini-3.5-flash-lite |
| tokens-256-overlap-32 | 256 | 32 | 1096 | 0.30 | 0.35 | 0.60 | 0.10 | 0.92 | 8 / 20 | 0.60 | gemini-3.5-flash-lite |
| tokens-128-overlap-32 | 128 | 32 | 1228 | 0.25 | 0.35 | 0.60 | 0.10 | 0.98 | 8 / 20 | 0.60 | gemini-3.5-flash-lite |
| tokens-64-overlap-16 | 64 | 16 | 1900 | 0.30 | 0.45 | 0.80 | 0.10 | 0.95 | 10 / 20 | 0.50 | gemini-3.5-flash-lite |
| tokens-32-overlap-8 | 32 | 8 | 3510 | 0.10 | 0.35 | 0.70 | 0.00 | 1.00 | 7 / 20 | 0.65 | gemini-3.5-flash-lite |
