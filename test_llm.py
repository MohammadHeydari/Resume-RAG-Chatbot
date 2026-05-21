from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="gemma3:4b")

print(llm.invoke("سلام خودت رو معرفی کن"))