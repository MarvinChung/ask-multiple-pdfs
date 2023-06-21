import openai
openai.api_key = "EMPTY"
openai.api_base = "http://35.221.228.215:8000/v1"
model = "redpajama-incite-7b-zh-chat"
prompt = "Once upon a time"
completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=64)
print(prompt + completion.choices[0].text)
