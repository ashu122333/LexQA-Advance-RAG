from openrouter import OpenRouter
import os
import json
from dotenv import load_dotenv

load_dotenv()


def isValidOutput(query,context,output):
    load_dotenv()
    SYSTEM_PROMPT = """You are a JSON API.
        Return ONLY valid JSON.
        No markdown.
        No explanations.
        No extra text.
        """
    
    PROMPT = """You are a strict but fair auditor for a research paper Q&A system.
    Your only job is to judge an answer — not improve it, not rewrite it.

    ## Your Two Checks

    ### Check 1 — Grounding
    Read every claim in the answer. Check if it is directly supported by the context below.
    A claim is grounded if the context explicitly states it or clearly implies it.
    A claim is NOT grounded if it:
    - facts not present in the context
    - Generalises beyond what the context says
    - Uses the context as a springboard to add outside knowledge

    ### Check 2 — Relevance
    Does the answer actually respond to what the question asked?
    An answer is NOT relevant if it:
    - Answers a different question than what was asked
    - Only addresses part of a multi-part question
    - Goes off-topic even if the content is accurate

    ## Scoring Rules
    - Be strict on grounding. Paraphrasing the context is fine. Adding to it is not.
    - Be fair on relevance. A concise on-topic answer is relevant even if it is short.
    - If the answer is "I could not find enough information" → set grounded=true, relevant=true.
    This is a valid honest response, not a failure.

    ---

    Question:
    {question}

    Retrieved Context:
    {context}

    Answer to Judge:
    {answer}

    ---

    Return ONLY valid JSON. No explanation outside the JSON.

    {{
    "grounded": true or false,
    "grounded_unsupported_claims": [
        "exact phrase or sentence from the answer that is not supported by the context"
    ],
    "relevant": true or false,
    "relevant_explanation": "one sentence explaining the drift — or empty string if relevant is true"
    }}
    """

    
    with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as client:
        response = client.chat.send(
            model=os.getenv("OPENROUTER_GENERATION_MODEL"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT.format(question=query, context=context, answer=output)}
            ],
            temperature=0,
            response_format={"type": "json_object"}  # IMPORTANT
        )
        output=response.choices[0].message.content
        return json.loads(output)
    



# Depriciated prompt
# PROMPT=""" You are a strict judge. Given a question, retrieved context, and an answer:
    #     1. Is every claim in the answer directly supported by the context? (grounded: true/false)
    #     2. Does the answer actually address what the question asked? (relevant: true/false)

    #     If grounded=false, list the unsupported claims.
    #     If relevant=false, explain the drift.

    #     Question: {question}
    #     Context: {context}
    #     Answer: {answer}
       
    # Return ONLY valid JSON in this format:
    # {{
    #     "grounded": true or false,
    #     "grounded_unsupported_claims": ["a list of claims in the answer that are not supported by the context, or an empty list if grounded is true"],
    #     "relevant": true or false,
    #     "relevant_explanation": "a concise sentence describing why the query is valid or not"
    # }}
    #  """    