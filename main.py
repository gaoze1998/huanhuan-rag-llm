import os

from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

files = os.listdir("data")
all_splits = []

for file in files:
    if ".txt" in file:
        loader = TextLoader(os.path.join("data", file), encoding="utf-8")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits += text_splitter.split_documents(docs)

huggingface_embeddings = HuggingFaceEmbeddings(model_name="bert-base-chinese", model_kwargs={"device": "cuda"})

if os.path.exists("./persist_dir/"):
    vectorstore = Chroma(embedding_function=huggingface_embeddings,
                         persist_directory="./persist_dir/")
else:
    vectorstore = Chroma.from_documents(documents=all_splits,
                                        embedding=huggingface_embeddings,
                                        persist_directory="./persist_dir/")

retriever = vectorstore.as_retriever(search_type="similarity")

llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)

contextualize_q_system_prompt = """你的输入是chat_histories和用户输入。 \
你要帮我将这两个合并转化为一个上下文输入到另一个promote作为上下文环境。"""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """你是角色扮演游戏里的主持人兼职NPC。\
你可以扮演除了主角以外的任何角色或环境。 \
你可以利用的信息处理用户输入还有一下上下文环境。 \
请你以“角色名：话语”的格式来响应。 \
请以符合上下文环境且复合角色特点的方式来回复。 \

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

while True:
    t = input("请输入：")
    print(conversational_rag_chain.invoke(
        {"input": t},
        config={
            "configurable": {"session_id": "abc123"}
        },
    )["answer"])
