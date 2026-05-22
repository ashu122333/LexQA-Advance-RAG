import json
import os
from dotenv import load_dotenv  
from openrouter import OpenRouter

load_dotenv()

PROMPT = """You are formatting a research assistant's answer for display.
Rewrite the answer below using clean Markdown. Do not add, remove, or change any information.

## Formatting Rules

- Use `##` headings to separate major ideas (only if the answer has 2+ distinct sections)
- Use bullet points for lists of 3 or more items
- Use **bold** for key terms, paper names, and important concepts
- Use `inline code` for model names, metrics, and technical terms (e.g. `bge-reranker`, `top-k`)
- Use > blockquotes for direct definitions or key findings from papers
- Use plain paragraphs for short answers — do NOT force structure where it is not needed
- Never add new information, examples, or explanations that were not in the original answer

## Answer to format
{answer}

Return only the formatted Markdown. No preamble, no commentary.

"""




def rewrite_response(answer):
    load_dotenv()
    with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as client:
        response = client.chat.send(
            model=os.getenv("OPENROUTER_GENERATION_MODEL"),
            messages=[
                {"role": "user", "content": PROMPT.format(answer=answer)}
            ],
            temperature=0
        )
        return response.choices[0].message.content

# TESTING
# answer="At decoding time, RAG-Sequence and RAG-Token differ in how they approximate the arg maxy p(y|x). RAG-Token can be seen as a standard, autoregressive seq2seq generator with a transition probability that can be plugged into a standard beam decoder. In contrast, RAG-Sequence does not break into a conventional per-token likelihood, so it cannot be solved with a single beam search. Instead, it runs beam search for each document z, scoring each hypothesis using pθ(yi|x, z, y1:i−1). This yields a set of hypotheses Y, some of which may not have appeared in the beams of all documents. To estimate the probability of an hypothesis y, it runs an additional forward pass for each document z for which y does not appear in the beam, multiply generator probability with pη(z|x) and then sum the probabilities across beams for the marginals. This decoding procedure is referred to as “Thorough Decoding.” For longer output sequences, |Y| can become large, requiring many forward passes. For more efficient decoding, it can make a further approximation that pθ(y|x, zi) ≈0 where y was not generated during beam search from x, zi. This avoids the need for additional forward passes. The trade-off created by this difference is that RAG-Token is more efficient but may not capture the full range of possibilities, while RAG-Sequence is more thorough but may be slower and more computationally expensive."
# print(rewrite_response(answer))