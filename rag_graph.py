from typing import TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver


# STATE
class State(TypedDict):
    input: str
    # chat_history: Sequence[BaseMessage]
    context: str
    answer: str


# EMBEDDING + DB
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

retriever = db.as_retriever(search_kwargs={"k": 3})


# LLM
llm = OllamaLLM(model="gemma3:4b")


# MAIN NODE
def call_model(state: State):

    question = state["input"]

    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    You are a friendly and casual AI assistant.

    - Talk like a normal human (casual, natural, slightly informal)
    - DO NOT say things like "Based on the context" or "According to the provided information"
    - Just answer directly
    - Keep it simple and conversational

    Context:
    {context}

    Question:
    {question}

    Answer:
"""

    answer = llm.invoke(prompt)

    return {
        "context": context,
        "answer": answer
    }


# GRAPH
workflow = StateGraph(State)

workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

memory = MemorySaver()

app = workflow.compile(checkpointer=memory)


# ASK FUNCTION
def ask(question):

    config = {"configurable": {"thread_id": "user-1"}}

    result = app.invoke(
        {"input": question},
        config=config
    )

    return result["answer"]


# TEST RUN
if __name__ == "__main__":
    print(ask("What is your experience?"))
    print(ask("What did he study?"))