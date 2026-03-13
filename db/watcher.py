import time
import chromadb
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pypdf import PdfReader

# Setup
client = chromadb.PersistentClient(path="./my_chroma_data")
collection = client.get_or_create_collection(name="pdf_knowledge_base")

class PDFHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(".pdf"):
            return
        
        print(f"New file detected: {event.src_path}")
        # Logic to read and add content to ChromaDB
        reader = PdfReader(event.src_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                collection.upsert(
                    documents=[text],
                    ids=[f"{event.src_path}_{i}"],
                    metadatas=[{"source": event.src_path}]
                )
        print("File processed and added to database!")

# 3. Start watching
path = "./incoming_pdfs" 
event_handler = PDFHandler()
observer = Observer()
observer.schedule(event_handler, path, recursive=False)
observer.start()

print(f"Watching folder: {path}...")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()