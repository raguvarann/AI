from langchain_community.llms import Ollama

llm = Ollama(model="llama3")

response = llm.invoke("Explain LangChain in 3 simple bullet points.")
print(response)
