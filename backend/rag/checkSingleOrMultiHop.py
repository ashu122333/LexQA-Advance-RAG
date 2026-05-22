import os
import json
from dotenv import load_dotenv  
from openrouter import OpenRouter




SYSTEM_PROMPT = """
You are a JSON API.
Return ONLY valid JSON.
No markdown.
No explanations.
No extra text.
"""


PROMPT = """
You are a query planner for a research paper RAG system.

Your task is to decide whether a query requires:
- single_hop retrieval
OR
- multi_hop retrieval.

IMPORTANT:
Decompose queries based on INFORMATION SOURCES that must be retrieved,
NOT based on sentence fragments or wording.

The goal of decomposition is:
- each sub-question should retrieve distinct information
- sub-questions should be independently searchable

Rules:

MULTI_HOP:
Use when the question:
- compares multiple papers, methods, or systems
- asks for relationships between concepts
- requires combining information from multiple sources

For comparison questions:
- create ONE retrieval sub-question per entity/system
- do NOT split by attributes like "cost", "advantages", etc.
- Each sub question shoule be independent and should not require the other sub-questions to be answered.

SINGLE_HOP:
Use when:
- one retrieval is enough
- the question focuses on one concept/system

Good Examples:

Question:
"How do FLARE and Self-RAG perform adaptive retrieval?"

Output:
{{
  "type": "multi_hop",
  "sub_questions": [
    "How does FLARE perform adaptive retrieval?",
    "How does Self-RAG perform adaptive retrieval?"
  ]
}}

Question:
"What dataset did Self-RAG use?"

Output:
{{
  "type": "single_hop"
}}

Return ONLY valid JSON.

Max 3 sub_questions.

Query:
{query}
"""


def check_single_or_multi_hop(question):
    load_dotenv()
    with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as client:
        response = client.chat.send(
            model=os.getenv("OPENROUTER_REASONING_MODEL"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT.format(query=question)}
            ],
            temperature=0,
            response_format={"type": "json_object"}  # IMPORTANT
        )
        return json.loads(response.choices[0].message.content) #give JSON output