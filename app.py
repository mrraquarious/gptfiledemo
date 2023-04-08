import base64
import os
from io import BytesIO
import re
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain.document_loaders import PyPDFLoader
from tempfile import NamedTemporaryFile
import openai
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS, VectorStore
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

from langchain import OpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
import pinecone
# from fpdf import FPDF

st.set_page_config(page_title="ChatGPT", page_icon="🌐")

MAIN = st.empty()

PINECONE_API_KEY = '17dbbc05-f3bc-4cc2-ad40-0bf6d0b13958'
PINECONE_API_ENV = 'eu-west1-gcp'


@st.cache
def parse_pdf(file: BytesIO) -> List[str]:
    with NamedTemporaryFile(dir='.', suffix='.pdf') as f:
        f.write(file.getbuffer())
    loader = PyPDFLoader(file.name)
    output = loader.load()
#     pdf = PdfReader(file)
#     output = []
#     for page in pdf.pages:
#         text = page.extract_text()
#         # Merge hyphenated words
#         text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
#         # Fix newlines in the middle of sentences
#         text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
#         # Remove multiple newlines
#         text = re.sub(r"\n\s*\n", "\n\n", text)

#         output.append(text)

    return output

def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks

def init_openai_settings():
    openai.api_key = os.getenv("OPENAI_API_KEY")


def init_session():
    if not st.session_state.get("params"):
        st.session_state["params"] = dict()
    if not st.session_state.get("chats"):
        st.session_state["chats"] = {}


def new_chat(chat_name):
    if not st.session_state["chats"].get(chat_name):
        st.session_state["chats"][chat_name] = {
            "answer": [],
            "question": [],
            "messages": [
                {"role": "system", "content": st.session_state["params"]["prompt"]}
            ],
        }
    return chat_name


def switch_chat(chat_name):
    if st.session_state.get("current_chat") != chat_name:
        st.session_state["current_chat"] = chat_name
        init_chat(chat_name)
        st.stop()


def switch_chat2(chat_name):
    if st.session_state.get("current_chat") != chat_name:
        st.session_state["current_chat"] = chat_name
        init_sidebar()
        init_chat(chat_name)
        st.stop()

def init_sidebar():
    st.sidebar.title("ChatWithFile")
    chat_name_container = st.sidebar.container()
    chat_config_expander = st.sidebar.expander('Chat configuration')
    # export_pdf = st.sidebar.empty()

    # chat config
    st.session_state["params"] = dict()
    # st.session_state['params']["api_key"] = chat_config_expander.text_input("API_KEY", placeholder="Please input openai key")
    st.session_state["params"]["model"] = chat_config_expander.selectbox(
        "Please select a model",
        ["gpt-3.5-turbo"],  # , "text-davinci-003"
        help="ID of the model to use",
    )
    st.session_state["params"]["temperature"] = chat_config_expander.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=1.2,
        step=0.1,
        format="%0.2f",
        help="""What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.""",
    )
    st.session_state["params"]["max_tokens"] = chat_config_expander.slider(
        "MAX_TOKENS",
        value=2000,
        step=1,
        min_value=100,
        max_value=4000,
        help="The maximum number of tokens to generate in the completion",
    )

    st.session_state["params"]["prompt"] = chat_config_expander.text_area(
        "Prompts",
        "You are a helpful assistant that answer questions as possible as you can.",
        help="The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.",
    )
    
    chat_config_expander.caption('Looking for help at https://platform.openai.com/docs/api-reference/chat')

    new_chat_button = chat_name_container.button(
        label="➕ New Chat"
    )  # , use_container_width=True
    if new_chat_button:
        new_chat_name = f"Chat{len(st.session_state['chats'])}"
        st.session_state["current_chat"] = new_chat_name
        new_chat(new_chat_name)

    with st.sidebar.container():

        for chat_name in st.session_state.get("chats", {}).keys():
            if chat_name == st.session_state.get('current_chat'):
                chat_name_container.button(
                    label='💬 ' + chat_name,
                    on_click=switch_chat2,
                    key=chat_name,
                    args=(chat_name,),
                    type='primary',
                    # use_container_width=True,
                )
            else:
                chat_name_container.button(
                    label='💬 ' + chat_name,
                    on_click=switch_chat2,
                    key=chat_name,
                    args=(chat_name,),
                    # use_container_width=True,
                )

    if new_chat_button:
        switch_chat(new_chat_name)




def init_chat(chat_name):
    chat = st.session_state["chats"][chat_name]
    uploaded_file = st.file_uploader(
            "Upload a pdf file",
            type=["pdf"],
        )
    docsearch = None
    qa = None
    chat_history = []
    if uploaded_file is not None:
            with NamedTemporaryFile(dir='.', suffix='.pdf') as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(uploaded_file.name)
            docs = loader.load()
            texts = text_to_docs(docs)
            with st.spinner("Indexing document... This may take a while⏳"):
                embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
                docsearch = FAISS.from_documents(texts, embeddings)
                qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), docsearch.as_retriever())
    # with MAIN.container():
    answer_zoom = st.container()
    ask_form = st.empty()

#     if len(chat['messages']) == 1 and st.session_state["params"]["prompt"]:
#         chat["messages"][0]['content'] = st.session_state["params"]["prompt"]

    if chat['messages']:
        answer_zoom.caption(f"""ℹ️ Prompt: {chat["messages"][0]['content']}""")
    if chat["question"]:
        for i in range(len(chat["question"])):
            answer_zoom.markdown(f"""😃 **YOU:** {chat["question"][i]}""")
            if i < len(chat["answer"]):
                answer_zoom.markdown(f"""🤖 **AI:** {chat["answer"][i]}""")

    with ask_form.form(chat_name):
        col1, col2 = st.columns([10, 1])
        input_text = col1.text_area("😃 You: ", "Hello, how are you?", key="input", max_chars=2000,
                                     label_visibility='collapsed')

        submitted = col2.form_submit_button("🛫")

        if submitted and input_text:
            
            result = qa({"question": input_text, "chat_history": chat_history})
            chat_history += [(input_text, result["answer"])]
            
            chat["messages"].append({"role": "user", "content": input_text})
            answer_zoom.markdown(f"""😃 **YOU:** {input_text}""")

            with st.spinner("Wait for responding..."):
                answer = result["answer"]
                answer_zoom.markdown(f"""🤖 **AI:** {answer}""")
            chat["messages"].append({"role": "assistant", "content": answer})
            if answer:
                chat["question"].append(input_text)
                chat["answer"].append(answer)




def init_css():
    """try to fixed input field"""
    st.markdown(
        """
    <style>
div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] > [data-testid="stForm"]  {
    border: 20px groove red;
    position: fixed;
    width: 100%;
    
    flex-direction: column;
    flex-grow: 5;
    overflow: auto;
}        
    </style>
    """,
        unsafe_allow_html=True,
    )


def ask(messages):
    if st.session_state["params"]["model"] == 'gpt-3.5-turbo':
        response = openai.ChatCompletion.create(
            model=st.session_state["params"]["model"],
            temperature=st.session_state["params"]["temperature"],
            messages=messages,
            max_tokens=st.session_state["params"]["max_tokens"],
        )
        answer = response["choices"][0]["message"]["content"]
    else:
        raise NotImplementedError('Not implemented yet!')
    return answer


if __name__ == "__main__":
    print("loading")
    init_openai_settings()
    # init_css()
    init_session()
    init_sidebar()
    if st.session_state.get("current_chat"):
        print("current_chat: ", st.session_state.get("current_chat"))
        init_chat((st.session_state["current_chat"]))
    if len(st.session_state["chats"]) == 0:
        switch_chat(new_chat(f"Chat{len(st.session_state['chats'])}"))
