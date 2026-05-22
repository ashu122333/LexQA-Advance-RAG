import os
import psycopg2
import re
import requests
import json
from dotenv import load_dotenv
# from embeddingModel import create_embedding
import psycopg2
from psycopg2.extras import RealDictCursor
from .embeddingModel import create_embedding

load_dotenv()

# i will add ,metadata filtering later ==> paper_id


def get_sematic_chunks( question, top_k=10):
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    with conn.cursor() as cur:
        # Get embedding for the question
        
        q_embedding = create_embedding(question)

        # Query for relevant chunks using cosine similarity
        cur.execute("""
            SELECT id,chunk_text, summary
            FROM chunks
            ORDER BY embedding <=>  %s::vector
            LIMIT %s;
        """, (q_embedding, top_k))

        results = cur.fetchall()
    conn.close()
    return results




def get_top_10_chunks_bm25(query_text):
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        with conn.cursor() as cur:
            # 1. Clean the query and convert to OR logic
            # This turns "RAG system" into "RAG | system"
            # 2. Remove special characters that break tsquery
            # This regex keeps alphanumeric characters and spaces
            clean_query = re.sub(r"[^\w\s]", " ", query_text)
            
            # 2. Split and filter out empty strings
            keywords = [word for word in clean_query.split() if word]
            or_query = " | ".join(keywords)

            # 2. Use to_tsquery with OR logic for ranking
            # We use 'simple' config if 'english' is too aggressive with stemming
            search_query = """
            SELECT 
                id, chunk_text,summary,
                ts_rank_cd(to_tsvector('english', chunk_text), query) AS rank
            FROM 
                chunks, 
                to_tsquery('english', %s) query 
            WHERE 
                to_tsvector('english', chunk_text) @@ query
            ORDER BY 
                rank DESC
            LIMIT 10;
            """
            
            cur.execute(search_query, (or_query,))
            result=cur.fetchall()

            return result
    finally:
        conn.close()


def reciprocal_rank_fusion(semantic_results, keyword_results, alpha=0.5):
    # Create a dictionary to hold the combined scores
    combined_scores = {}

    # Process semantic results
    for rank, (chunk_id, chunk_text, summary) in enumerate(semantic_results):
        score = 1 / (rank + 1)  # Reciprocal rank score
        if chunk_id not in combined_scores:
            combined_scores[chunk_id] = 0
        combined_scores[chunk_id] += alpha * score

    # Process keyword results
    for rank, (chunk_id, chunk_text, summary, keyword_rank) in enumerate(keyword_results):
        score = 1 / (rank + 1)  # Reciprocal rank score
        if chunk_id not in combined_scores:
            combined_scores[chunk_id] = 0
        combined_scores[chunk_id] += (1 - alpha) * score

    # Sort chunks by combined score
    sorted_chunks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_chunks



def top10_chunks(question):
    semantic_results = get_sematic_chunks(question)
    keyword_results = get_top_10_chunks_bm25(question)
    combined_results = reciprocal_rank_fusion(semantic_results, keyword_results)
    ids=[]
    for chunk_id, score in combined_results:
        ids.append(chunk_id)
    ids=ids[:10]  # Keep only top 10 IDs
    ids=set(ids)  # Convert to set for faster lookup    
    top10_chunks=[]
    for chunk in keyword_results:
        if chunk[0] in ids: 
            top10_chunks.append(chunk[:3])
            ids.remove(chunk[0])

    for chunk in semantic_results:
        if chunk[0] in ids: 
            top10_chunks.append(chunk[:3])
            ids.remove(chunk[0])        

    return top10_chunks



def rerank_chunks(query, chunks, chunk_summaries, top_n=5):
    """
    Reranks a list of chunks using Cohere via OpenRouter.
    """
    if not chunks:
        return []

    url = "https://openrouter.ai/api/v1/rerank"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": os.getenv("OPENROUTER_RERANK_MODEL"),
        "query": query,
        "documents": chunks,
        "top_n": top_n
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        reranked_data = response.json()

        final_results = []
        for item in reranked_data.get("results", []):
            text=chunks[item["index"]]
            summary=chunk_summaries[item["index"]]
            final_results.append({"text": text, "summary": summary})

        return final_results

    except Exception as e:
        print(f"Reranking failed: {e}")
        # Fallback: return the first top_n results from the original list if API fails
        final_results = []
        for i in range(min(top_n, len(chunks))):
            final_results.append({"text": chunks[i], "summary": chunk_summaries[i]})
        return final_results


def get_relevant_chunks(question,top_n=5):
    # Step 1: Get top 10 chunks using combined BM25 and semantic search
    top_chunks = top10_chunks(question)

    # Step 2: Extract chunk texts and summaries for reranking
    chunk_texts = [chunk[1] for chunk in top_chunks]  # Extract chunk_text
    chunk_summaries = [chunk[2] for chunk in top_chunks]  # Extract summaries

    # Step 3: Rerank the chunks using Cohere via OpenRouter
    reranked_chunks = rerank_chunks(question, chunk_texts, chunk_summaries,top_n=top_n)
    return reranked_chunks
