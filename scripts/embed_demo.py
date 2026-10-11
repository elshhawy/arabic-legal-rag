from legalrag.embeddings import embed_query

vec = embed_query("هل يجوز فسخ العقد؟")
print("Shape:", vec.shape)
print("First 5 numbers:", vec[:5])