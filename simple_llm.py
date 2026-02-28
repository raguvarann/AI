from langchain_community.llms import Ollama

llm = Ollama(model="llama3")

response = llm.invoke("Explain Ragu in 3 simple bullet points.")
print(response)
