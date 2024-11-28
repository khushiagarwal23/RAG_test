import os
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
import datetime

os.environ['OPENAI_API_KEY'] = 'your-openai-api-key'  # Replace with your OpenAI API key

loader = TextLoader("your_document.txt")  # Replace with your actual document file
documents = loader.load()

embeddings = OpenAIEmbeddings()

faiss_db = FAISS.from_documents(documents, embeddings)

llm = OpenAI(temperature=0.7)

qa_chain = load_qa_chain(llm, chain_type="map_reduce")

conversational_retriever = ConversationalRetrievalChain(
    combine_docs_chain=qa_chain, 
    retriever=faiss_db.as_retriever()
)

def chatbot_conversation(query):
    response = conversational_retriever.ask(query)
    return response


def dynamic_greeting():
    """Generate a personalized greeting based on the time of day."""
    current_hour = datetime.datetime.now().hour
    if current_hour < 12:
        return "Good morning! I'm here to assist you."
    elif 12 <= current_hour < 18:
        return "Good afternoon! How can I help you today?"
    else:
        return "Good evening! How can I help you tonight?"

def start_chatbot():
    """Start a more dynamic chatbot conversation."""
    print(dynamic_greeting())
    print("Type 'exit' anytime to end the conversation.")
    print("Feel free to ask me anything!")
    
    while True:
        query = input("User: ").strip()
        
        if not query:
            print("Bot: Ayo you need to ask a question or type 'exit' to quit.")
            continue
        
        if query.lower() == "exit":
            print("Bot: Goodbye! It was a pleasure assisting you mate. Ahoi!")
            break
        
        response = chatbot_conversation(query)
        print(f"Bot: {response}")


if __name__ == "__main__":
    start_chatbot()
