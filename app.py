import os
os.environ["OPENAI_API_BASE"]= "http://35.189.163.143:8080/v1"
os.environ["OPENAI_API_KEY"]="EMPTY"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from htmlTemplates import css, bot_template, user_template
import pinecone 
from PyPDF2 import PdfReader
import streamlit as st

qa_prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 
{history}

### Instruction:
{question}

### Input:
{context}

### Response:
""" 


# Rewrite the following sentence for clarity.
REDPAJAMA_QA_PROMPT = PromptTemplate(
    template=qa_prompt_template, input_variables=["history", "context", "question"]
)

# condense_prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

# ### Instruction:
# 給定以下對話和後續問題，將後續問題用其原始語言改寫為獨立問題。

# ### Input:
# chat_history:
# {chat_history}

# question: 
# {question}

# ### Response:
# """ 

# condense_prompt_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
# ### Instruction:
# {question}

# ### Input:
# {chat_history}

# ### Response:
# """ 

# REDPAJAMA_CONDENSE_QUESTION_PROMPT = PromptTemplate(
#     template=condense_prompt_template, input_variables=["chat_history", "question"]
# )

# def get_chat_history(inputs) -> str:
#     res = []

#     ct = 0
#     for human, ai in inputs:
#         print(ct," human:", human)
#         print(ct, " ai:", ai)
#         res.append(f"### Instruction:\n{human}\n\n### Response:\n{ai}\n")
#         # res.append(f"人：{human}\n助手：{ai}\n")
#         ct += 1
#     return "".join(res)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=256,
        chunk_overlap=0,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(pinecone_index, pinecone_api_key, pinecone_env, text_chunks = None):
    # embeddings = OpenAIEmbeddings(model="multilingual-e5-base")
    # embeddings = OpenAIEmbeddings(deployment="multilingual-e5-base", model="multilingual-e5-base")
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs=model_kwargs
    )
    # vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    if text_chunks == None:
        vectorstore = Pinecone.from_existing_index(pinecone_index, embeddings)
    else:
        # OpenAI Embedding (Mtk endpoint) needs to set batch_size = 1
        # vectorstore = Pinecone.from_texts(text_chunks, embeddings, index_name=pinecone_index, batch_size=1)
        vectorstore = Pinecone.from_texts(text_chunks, embeddings, index_name=pinecone_index)

    return vectorstore


def get_conversation_chain(vectorstore, model_name="redpajama-incite-7b-zh-instruct", temperature=0.0):
    llm = ChatOpenAI(
        model_name=model_name, 
        temperature=temperature,
        model_kwargs={"top_p":0.0, "frequency_penalty": 1.1})

    # question_generator = LLMChain(llm=llm, prompt=REDPAJAMA_CONDENSE_QUESTION_PROMPT)
    # doc_chain = load_qa_chain(llm, chain_type="map_reduce")

    # conversation_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=vectorstore.as_retriever(),
    #     condense_question_prompt=REDPAJAMA_CONDENSE_QUESTION_PROMPT,
    #     combine_docs_chain_kwargs={"prompt": REDPAJAMA_QA_PROMPT},
    #     verbose=True,
    #     get_chat_history=get_chat_history
    # )
    # return conversation_chain

    conversation_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={
            "prompt": REDPAJAMA_QA_PROMPT,
            "verbose": True,
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question",
                human_prefix="\n### Instruction",
                ai_prefix="\n### Response"),
        },
    )

    return conversation_chain


def handle_userinput(user_question):
    print("user_question:", user_question)
    response = st.session_state.conversation.run({'query': user_question})
    print("response:", response)

    st.session_state.chat_history.append((user_question, response))

    # print("get_relevant_documents:", st.session_state.conversation._get_docs(user_question, None))
    print("memory:", st.session_state.conversation.combine_documents_chain.memory)
    
    for i, message in enumerate(st.session_state.chat_history):
        st.write(user_template.replace(
            "{{MSG}}", message[0]), unsafe_allow_html=True)
        st.write(bot_template.replace(
            "{{MSG}}", message[1]), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if 'init' not in st.session_state:
        st.session_state.init = True
        default_pinecone_api_key = "d04018e5-4201-44cb-8141-3253264fc4af"
        default_pinecone_env = "us-west4-gcp"
        default_pinecone_index = "pinecone-demo"
        vectorstore = get_vectorstore(default_pinecone_index, default_pinecone_api_key, default_pinecone_env)

        # create conversation chain
        st.session_state.conversation = get_conversation_chain(vectorstore)

    st.header("Chat with multiple PDFs :books:")

    # user_question = st.text_input("Ask a question about your documents:", key='widget', on_change=submit)
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("parameters")
        model_name = st.selectbox(
            'Model Name',
            ('redpajama-incite-7b-zh-instruct', 'chinese-alpaca-plus-7b-hf'))
        temperature = st.slider("temperature", 0.0, 2.0, 0.0)

        st.subheader("Pinecone")
        pinecone_api_key     = st.text_input("Pinecone API key", type="password", value="d04018e5-4201-44cb-8141-3253264fc4af")
        pinecone_env         = st.text_input("Pinecone environment", value="us-west4-gcp")
        pinecone_index       = st.text_input("Pinecone index name", value="pinecone-demo")

        if st.button("Connect Pinecone"):
            vectorstore = get_vectorstore(pinecone_index, pinecone_api_key, pinecone_env)

            # create conversation chain
            st.session_state.conversation = get_conversation_chain(
                    vectorstore, model_name, temperature)

        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(pinecone_index, pinecone_api_key, pinecone_env, text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore, model_name, temperature)


if __name__ == '__main__':
    main()
