import chromadb  # Import the actual library, not the file name

# local persistent client
client = chromadb.PersistentClient(path="./my_chroma_data")

# # Delete the collection if it exists to start fresh
# client.delete_collection("my_common")


# collection
collection = client.get_or_create_collection(name="my_common")

# Add common content
collection.upsert(
    documents=["Doc 1 text_v1", "Doc 1 text", "Doc 2 text", "Doc 3 text"],
    ids=["id1_v1", "id1","id2", "id3"],
    metadatas=[
        {"category": "version1"}, 
        {"category": "legal"}, 
        {"category": "hr"},
        {"category": "tech"}
    ]  
)

#as table view
all_data = collection.get()

# Zip the lists together to see the exact state of each record
for i in range(len(all_data['ids'])):
    print(f"ID: {all_data['ids'][i]} | Content: {all_data['documents'][i]} | Meta: {all_data['metadatas'][i]}")


# # Query the database
# results = collection.query(
#     query_texts=["How to automate cloud?"],
#     n_results=1,
#     where={"category": "tech"} # This is the filter
# )
# print(results)


# # Retrieve all records
# all_data = collection.get()

# # Print IDs, Content, and Metadata separately
# print("IDs:", all_data['ids'])
# print("Content:", all_data['documents'])
# print("Metadata:", all_data['metadatas'])



# # data = collection.get(ids=["id1", "id1_v1", "id2", "id3"])
# # print(data['documents'])






# # Update metadata for id1_v1 to the exact dictionary you want
# collection.update(
#     ids=["id1_v1"],
#     metadatas=[{"category": "version1"}] 
# )

# # Verify the update
# check = collection.get(ids=["id1_v1"])
# print("Updated Metadata:", check['metadatas'])


# data = collection.get(ids=["id1_v1"])
# print(data["metadatas"])


# collection.update(
#     ids=["id1_v1"],
#     metadatas=[{"category": "version1"}]
# )

# check = collection.get(ids=["id1_v1"])
# print(check["metadatas"])