import os
os.environ["OPENAI_API_BASE"]= "http://35.221.228.215:8000/v1"
os.environ["OPENAI_API_KEY"]="EMPTY"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#OPENAI_API_BASE = "https://api.openai.com/v1"
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from htmlTemplates import css, bot_template, user_template
import pinecone 
from PyPDF2 import PdfReader
import streamlit as st

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

# def get_text_chunks():
#     loader = TextLoader("./fake_esg.txt")
#     documents = loader.load()
#     text_splitter = CharacterTextSplitter(chunk_size=32, chunk_overlap=0)
#     texts = text_splitter.split_documents(documents)
#     return texts


def update_vectorstore(texts):
    # model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        # model_kwargs=model_kwargs
    )
    # embeddings = OpenAIEmbeddings()

    # embeddings = OpenAIEmbeddings(openai_api_key="")
    st.session_state.vectorstore = Chroma.from_texts(texts, embeddings)

def update_llm():
    model_name = st.session_state.model_name
    print("model_name:", model_name)
    st.session_state.llm = ChatOpenAI(
            model_name=model_name, #"redpajama-incite-7b-zh-instruct", 
            streaming=True, callbacks=[StreamingStdOutCallbackHandler()],
            temperature=0.0,
            model_kwargs={"top_p":0.0, "frequency_penalty": 1.1})

def update_conversation_chain():

    st.session_state.conversation = RetrievalQA.from_chain_type(
        llm=st.session_state.llm,
        chain_type="stuff",
        retriever=st.session_state.vectorstore.as_retriever(),
        chain_type_kwargs={
            "prompt": REDPAJAMA_QA_PROMPT,
            "verbose": True
        },
    )



def handle_userinput(user_question):
    print("user_question:", user_question)
    print("st.session_state.conversation:", st.session_state.conversation)
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

        robot_ans = f"根據相關文件，" + section["response"]
        st.write(bot_template.replace(
            "{{MSG}}", robot_ans), unsafe_allow_html=True)

        for ct, doc in enumerate(section["docs"]):
            intro = f"以下內容為與本問題最相關的段落 {ct+1}:"
            st.write(bot_template.replace(
                "{{MSG}}", intro), unsafe_allow_html=True)
            st.write(bot_template.replace(
                "{{MSG}}", doc), unsafe_allow_html=True)

        


def main():
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "llm" not in st.session_state:
        st.session_state.llm = None

    # if "init" not in st.session_state:
    #     texts = get_text_chunks()
    #     vectorstore = get_vectorstore(texts)
    #     st.session_state.conversation = get_conversation_chain(vectorstore)
    #     st.session_state.init = True

    st.header("Chat with multiple PDFs :books:")
    
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        update_conversation_chain()
        handle_userinput(user_question)
    elif user_question:
        st.write("pdf is not uploaded yet", unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("parameters")

        model_name = st.selectbox(
            'Model Name',
            ('redpajama-incite-7b-zh-instruct', 'chinese-alpaca-plus-7b-hf'), key='model_name', on_change=update_llm)
        # temperature = st.slider("temperature", 0.0, 2.0, 0.0)

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
                update_vectorstore(text_chunks)

                # create llm
                update_llm()

                # create conversation chain
                update_conversation_chain()


if __name__ == '__main__':
    main()
