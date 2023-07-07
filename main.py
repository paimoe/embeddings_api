from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
MODEL = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

class EmbedRequest(BaseModel):
    text: str

class SimilarRequest(BaseModel):
    text: list[str]

def generate(text):
    embeds = MODEL.encode([text])
    return embeds[0]

def similarity(embed1, embed2):
    return util.pytorch_cos_sim(embed1, embed2)
    
# embeddings = model.encode(sentences)
# print(embeddings)
# s1 = embeddings[0]
# s2 = embeddings[1]
# print(s1.shape)

# print("similarity")
# print(util.pytorch_cos_sim(s1, s2))

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/embed")
async def embed(req: EmbedRequest):
    return {
        "text": req.text,
        "embedding": generate(req.text).tolist()
    }

@app.post("/similar")
async def similar(req: SimilarRequest):
    embed1 = generate(req.text[0])
    embed2 = generate(req.text[1])
    return {
        "text": req.text,
        "similarity": similarity(embed1, embed2).tolist()
    }