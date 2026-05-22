import json
import os
from dotenv import load_dotenv  
from openrouter import OpenRouter

load_dotenv()

SYSTEM_PROMPT = """
You are a JSON API.
Return ONLY valid JSON.
No markdown.
No explanations.
No extra text.
"""

PROMPT = """You are a precise research assistant that answers questions 
strictly based on the provided context from research papers.

## Your Rules
- Answer ONLY using information present in the context below.
- Be thorough but concise — explain clearly, include examples or key details 
  from the context if they help understanding.
- If the context contains partial information, use what is there and clearly 
  state what is missing.
- Do NOT add outside knowledge. Do NOT guess or infer beyond the context.
- If the answer is truly not in the context, return:
  {{"answer": "I could not find enough information in the provided papers to answer this.", "grounded": false}}

## Context (from research papers)
{context}

## Question
{query}

## Output Format
Return ONLY this JSON, no extra text:
{{
  "answer": "your detailed answer here, written for someone who has not read the papers",
  "grounded": true
}}
"""




def generate_response(question, context):
    load_dotenv()
    with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as client:
        response = client.chat.send(
            model=os.getenv("OPENROUTER_GENERATION_MODEL"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT.format(context=context, query=question)}
            ],
            temperature=0,
            response_format={"type": "json_object"}  # IMPORTANT
        )
        return json.loads(response.choices[0].message.content)