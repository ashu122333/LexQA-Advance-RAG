from dotenv import load_dotenv
from .singleHopChunkFinder import get_relevant_chunks

load_dotenv()

TOP_N=3

def relevante_chunks(questions):
    chunks=[]
    for question in questions:
        sub_chunks=(get_relevant_chunks(question,top_n=TOP_N))
        for chunk in sub_chunks:
            chunks.append(chunk)
    return chunks    