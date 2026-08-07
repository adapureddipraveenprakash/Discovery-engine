import os

# Must be set before faiss / torch import OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
