import numpy as np
import faiss

img = np.load("outputs/embeddings/image_embeddings_flava.npy")
txt = np.load("outputs/embeddings/text_embeddings_flava.npy")

print(img.shape)   # should be (17496, 768)
print(txt.shape)   # should be (17496, 768)

index_img = faiss.read_index("outputs/embeddings/faiss_image_flava.index")
index_txt = faiss.read_index("outputs/embeddings/faiss_text_flava.index")

print(index_img.ntotal)
print(index_txt.ntotal)
