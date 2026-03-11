import chromadb

# Initialize the  client and collection
client = chromadb.PersistentClient(path="./my_chroma_data")
collection = client.get_or_create_collection(name="pdf_knowledge_base")

print("--- ChromaDB Query Tool ---")
print("Type 'exit' to quit.")

while True:
    query = input("\nAsk a question: ")
    if query.lower() == 'exit':
        break
    
    # Run the query
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    # Display results
    if results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            print(f"\nResult {i+1}: {doc[:300]}...") # Show snippet
            print(f"Source: {results['metadatas'][0][i].get('source', 'Unknown')}")
    else:
        print("No results found.")