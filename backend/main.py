from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

from rag.inputValidation import is_valid_input
from rag.inputRewrite import rewrite_query
from rag.checkSingleOrMultiHop import check_single_or_multi_hop
from rag.singleHopChunkFinder import get_relevant_chunks
from rag.multiHopChunkFinder import relevante_chunks
from rag.generateAnswer import generate_response
from rag.outputValidation import isValidOutput
from rag.outputRewrite import rewrite_response



app = FastAPI(title="LexQA", version="1.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    """Render uses this to check if the service is up."""
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    timings = {}
    question = req.question.strip()
    print(f"Received question: {question}")

    # Step 1: query validation, rewrite and planning
    t0 = time.time()
    valid =is_valid_input(question) 
    if valid["is_valid"]==False:
        return{"error":"Invalid query", "reason":valid["reason"]}
    print("Input validation passed.")
    rewrite_result = rewrite_query(question)
    new_question = rewrite_result["new_query"]
    print(f"Rewritten question: {new_question}")
    plan=check_single_or_multi_hop(new_question)
    print(f"Plan: {plan}")
    hops=plan["type"]
    if hops=="multi_hop":
        questions=plan["sub_questions"]
    else:
        questions=[new_question]    
    timings["input_preprocessing_ms"] = round((time.time() - t0) * 1000)


    # Step 2: Embeddings and retrieval
    t0 = time.time()
    if plan["type"] == "single_hop":
        chunks = get_relevant_chunks(questions[0])
    elif plan["type"]=="multi_hop":
        chunks=relevante_chunks(questions)
    print(f"Retrieved {len(chunks)} relevant chunks.")    
        
    context = "\n".join([f"Summary: {c['summary']}\nChunk: {c['text']}" for c in chunks])
    timings["retrieval_ms"] = round((time.time() - t0) * 1000)



    # Step 3: Generate ,Validate and rewrite
    t0 = time.time()
    answer = generate_response(question, context)
    print(f"Generated answer")
    validated_answer=isValidOutput(question,context,answer)

    if validated_answer["grounded"]==False:
        grounded_error= {"error":"Could not generate a valid answer based on the retrieved information."
                ,"reason":validated_answer["grounded_unsupported_claims"]}
    if validated_answer["relevant"]==False:
        relevant_error= {"error":"The generated answer is not relevant to the question."
                ,"reason":validated_answer["relevant_explanation"]}
    print("Output validation passed.")
    # md_answer=rewrite_response(answer)
    timings["generation_ms"] = round((time.time() - t0) * 1000)

    return {
        "response":answer["answer"],
        # "md_response":md_answer,
        "timings":timings,
        "validation":{
            "grounded": validated_answer["grounded"],
            "relevant": validated_answer["relevant"],
            "grounded_unsupported_claims": validated_answer["grounded_unsupported_claims"],
            "relevant_explanation": validated_answer["relevant_explanation"]
        }
    }

class AnswerRequest(BaseModel):
    answer: str

@app.post("/md")
def md(req:AnswerRequest):
    answer=req.answer.strip()
    md_answer=rewrite_response(answer)
    return md_answer



