import os
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain

# Make sure your OpenAI API key is set
os.environ['OPENAI_API_KEY'] = 'your-openai-api-key'  # Replace with your OpenAI API key

# Load your document (Change the file path accordingly)
loader = TextLoader("your_document.txt")  # Replace with your actual document file
documents = loader.load()

# Create embeddings for the documents
embeddings = OpenAIEmbeddings()

# Create a FAISS vector store to store embeddings
faiss_db = FAISS.from_documents(documents, embeddings)

# Initialize the LLM (OpenAI GPT)
llm = OpenAI(temperature=0.7)

# Load the QA chain for conversational retrieval
qa_chain = load_qa_chain(llm, chain_type="map_reduce")

# Create the conversational retriever using the FAISS vector store
conversational_retriever = ConversationalRetrievalChain(
    combine_docs_chain=qa_chain, 
    retriever=faiss_db.as_retriever()
)

# Function to interact with the chatbot
def chatbot_conversation(query):
    response = conversational_retriever.ask(query)
    return response

# Start the conversation loop
def start_chatbot():
    print("Welcome to the chatbot! Type 'exit' to end the conversation.")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        response = chatbot_conversation(query)
        print(f"Bot: {response}")

# Run the chatbot
if __name__ == "__main__":
    start_chatbot()
