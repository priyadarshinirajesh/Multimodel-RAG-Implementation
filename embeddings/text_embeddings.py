# embeddings/text_embeddings.py

from fastembed import TextEmbedding

# CLIP text encoder
text_model = TextEmbedding(
    model_name="Qdrant/clip-ViT-B-32-text"
)

def embed_text(text: str):
    """
    Input  : clinical report text
    Output : 512-dim text embedding (list[float])
    """
    return list(text_model.embed([text]))[0]

