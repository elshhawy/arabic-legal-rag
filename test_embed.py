from legalrag.embeddings import embed_query, embed_passages

vec = embed_query("هل يجوز فسخ العقد؟")
print("Shape:", vec.shape)
print("First 5 numbers:", vec[:5])