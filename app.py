import base64
import os

import openai
import streamlit as st



from langchain import OpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import pinecone
# from fpdf import FPDF

st.set_page_config(page_title="ChatGPT", page_icon="🌐")

MAIN = st.empty()


PINECONE_API_KEY = '17dbbc05-f3bc-4cc2-ad40-0bf6d0b13958'
PINECONE_API_ENV = 'eu-west1-gcp'

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

@st.experimental_memo()
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)

        output.append(text)

    return output

@st.cache
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
    st.sidebar.title("ChatGPT")
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Upload a pdf, docx, or txt file",
            type=["pdf"],
        )

        if uploaded_file is not None:
            texts = parse_pdf(uploaded_file)
            with st.spinner("Indexing document... This may take a while⏳"):
                embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
                pinecone.init(
                    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
                    environment=PINECONE_API_ENV  
                )
                index_name = "new"
                docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

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

    # Download pdf
    # if st.session_state.get('current_chat'):
    #     chat = st.session_state["chats"][st.session_state['current_chat']]
    #     pdf = FPDF('p', 'mm', 'A4')
    #     pdf.add_page()
    #     pdf.set_font(family='Times', size=16)
    #     # pdf.cell(40, 50, txt='abcd.pdf')
    #
    #     if chat["answer"]:
    #         for i in range(len(chat["answer"]) - 1, -1, -1):
    #             # message(chat["answer"][i], key=str(i))
    #             # message(chat['question'][i], is_user=True, key=str(i) + '_user')
    #             pdf.cell(40, txt=f"""YOU: {chat["question"][i]}""")
    #             pdf.cell(40, txt=f"""AI: {chat["answer"][i]}""")
    #
    #     export_pdf.download_button('📤 PDF', data=pdf.output(dest='S').encode('latin-1'), file_name='abcd.pdf')


def init_chat(chat_name):
    chat = st.session_state["chats"][chat_name]

    # with MAIN.container():
    answer_zoom = st.container()
    ask_form = st.empty()

    if len(chat['messages']) == 1 and st.session_state["params"]["prompt"]:
        chat["messages"][0]['content'] = st.session_state["params"]["prompt"]

    if chat['messages']:
        # answer_zoom.markdown(f"""🤖 **Prompt:** {chat["messages"][0]['content']}""")
        # answer_zoom.info(f"""Prompt: {chat["messages"][0]['content']}""", icon="ℹ️")
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
            docs = docsearch.similarity_search(query, include_metadata=True)

            context = [doc.page_content for doc in docs]
            context = ".".join(context)

            chat["messages"]=[({"role":"system","content":"You can refer to following context to answer questions:" + context})]+chat["messages"]
            chat["messages"].append({"role": "user", "content": input_text})
            answer_zoom.markdown(f"""😃 **YOU:** {input_text}""")

            with st.spinner("Wait for responding..."):
                answer = ask(chat["messages"])
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
