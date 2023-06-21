from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
import os
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

os.environ["OPENAI_API_BASE"]= "http://35.221.228.215:8080/"
os.environ["OPENAI_API_KEY"]= "Empty"

llm = OpenAI(model="redpajama-incite-7b-zh-instruct", max_tokens=10, temperature=0, top_p=0)
resp = llm.predict("妳好啊")
print(resp)
