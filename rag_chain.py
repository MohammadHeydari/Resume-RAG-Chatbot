from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage

# add history to the chat
chat_history = []

# embedding (same as DB)
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# load DB
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

# LLM
llm = OllamaLLM(model="gemma3:4b")

def ask(question):

    # 1. retrieve docs
    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # create prompt wit history
    history_text = "\n".join(
        [f"User: {m.content}" if isinstance(m, HumanMessage)
         else f"Assistant: {m.content}" for m in chat_history]
    )

    prompt = f"""
You are a helpful assistant.

Use chat history + context.

Chat History:
{history_text}

Context:
{context}

Question:
{question}
"""

    # 3. call LLM
    answer = llm.invoke(prompt)

    # 4. update memory
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    return answer

print(ask("Who are you?"))
print(ask("What did he study?"))