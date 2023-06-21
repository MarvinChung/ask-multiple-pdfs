import os
import openai
openai.api_key = "EMPTY" # Not support yet
openai.api_base = "http://35.221.228.215:8000/v1"
#openai.api_base = "http://localhost:8000/v1"

def run(query):
    template = f"""
    Answer the following: {query}
    """
    completion = openai.ChatCompletion.create(
        model="redpajama-incite-7b-zh-chat",
        messages=[
            {"role": "user", "content": template}
        ],
        temperature=0,
        top_p=0,
        frequency_penalty=1.2,
        max_tokens=16
    )
    return completion.choices[0].message["content"]

query = 'why is the sky blue'
print(run(query))
