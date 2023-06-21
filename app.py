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

# qa_prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that exactly answer the question from the context, or if the answer is not inferrable by the context, state so. Write as concise as possible:

# ### Instruction:
#  請依據文章回答：陶朱隱園誰設計的?

# ### Input:
#  陶朱隱園（英語：Tao Zhu Yin Yuan）是位於臺灣臺北市信義計畫區的豪宅大樓，位在松勇路及松高路口，土地面積約為2468坪，由威京集團的中華工程與亞太工商聯合建，建築師為比利時籍的文森·卡利博。該建築樓高93.2公尺，地上樓層21樓，地下樓層4樓，總樓地板面積為12,962坪，於2018年第三季竣工，第一戶的正式交易日期為次年7月3日。

# 建築外觀特殊，屬於少有的旋轉建築，每層樓旋轉4.5度，頂樓設有直升機停機坪，內部則共有七部電梯，其中一部可承載超跑或救護車；其中每戶約三百坪，訴求垂直森林建築，種植超過 2.3 萬顆的喬灌木，裡面甚至還有叢林水瀑布，陽台部分的面積超過五十坪。2018年即完工取得使用執照，直到2019年出現首筆交易，由相關企業公司承耀公司買下7樓戶。過去一兩年實品屋完工後一直沒開放賞屋，市場傳出陶朱隱園價格太高乾脆封盤不賣。

# ### Response:
#  文森·卡利博

# -----
#  Below is an instruction that describes a task, paired with an input that provides further context. Write a response that exactly answer the question from the context, or if the answer is not inferrable by the context, state so. Write as concise as possible:

# ### Instruction:
#  請依據文章回答：{question}

# ### Input:
#  {context}

# ### Response:
# """ 

# qa_prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that exactly answer the question from the context, or if the answer is not inferrable by the context, state so. Write as concise as possible:
 
# ### Instruction:
#  請依據以下文章，只截取文章中的一段文字來回答這個問題：{question}

# ### Input:
#  {context}

# ### Response:
# """ 

qa_prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that exactly answer the question from the context, or if the answer is not inferrable by the context, state so. Write as concise as possible:
 
### Instruction:
 請依據文章，只摘要文章中的一段文字來回答這個問題：{question}

### Input:
 {context}

### Response:
""" 

# Rewrite the following sentence for clarity.
REDPAJAMA_QA_PROMPT = PromptTemplate(
    template=qa_prompt_template, input_variables=["context", "question"]
)
# REDPAJAMA_QA_PROMPT = PromptTemplate(
#     template=qa_prompt_template, input_variables=["history", "context", "question"]
# )

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

    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={
            "prompt": REDPAJAMA_QA_PROMPT,
            "verbose": True
        },
    )

    return llm_chain


def handle_userinput(user_question):
    print("user_question:", user_question)
    response = st.session_state.conversation.run({'query': user_question})
    print("response:", response)

    section = {"user_question": user_question, "response": response, "docs": []}

    # print("memory:", st.session_state.conversation.combine_documents_chain.memory)
    print("get_relevant_documents:")
    for ct, doc in enumerate(st.session_state.conversation._get_docs(user_question)):
        print("=================")
        print("ct:", ct)
        print(doc)
        section["docs"].append(doc.page_content)

    st.session_state.chat_history.append(section)

    nl = "\n"
    for i, section in enumerate(st.session_state.chat_history[-1:]):
        st.write(user_template.replace(
            "{{MSG}}", section["user_question"]), unsafe_allow_html=True)
        for ct, doc in enumerate(section["docs"]):
            intro = f"以下內容為與本問題最相關的文件 {ct+1}:"
            st.write(bot_template.replace(
                "{{MSG}}", intro), unsafe_allow_html=True)
            st.write(bot_template.replace(
                "{{MSG}}", doc), unsafe_allow_html=True)

        robot_ans = f"根據相關文件，" + section["response"]
        st.write(bot_template.replace(
            "{{MSG}}", robot_ans), unsafe_allow_html=True)


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
    # st.header("王品永續報告書 機器人 :book:")

    # user_question = st.text_input("Ask a question about your documents:", key='widget', on_change=submit)
    st.subheader("目前的pinecone 存放了王品永續報告書，你可以嘗試詢問'王品的理念為何'")
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
