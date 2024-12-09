from flask import Flask, jsonify, render_template, request
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os


# Fetch the API key from the environment
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise EnvironmentError("Missing `GOOGLE_API_KEY` environment variable. Please set it in the environment.")

# Use the API key in your initialization
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.1,
    api_key=api_key
)

embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def ask_question(query: str, customer_context: str = None) -> str:
    """Get response from Gemini API"""
    if customer_context:
        template = """
        You are an AI-powered customer support assistant, designed to address customer queries and provide information based on the context and data available. Follow these rules:

        1. Respond politely to greetings and maintain a professional tone.
        2. Answer questions based on the customer support context provided.
        3. If the context does not cover the question, inform the user politely and suggest they contact customer support through alternative channels for further assistance.

        Customer Support Context:
        {customer_context}

        Customer Question: {question}

        Answer:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["customer_context", "question"]
        )

    response = llm.predict(prompt.format(customer_context=customer_context, question=query))

    
    return response

# vector_store.save_local("faiss_index")

vectordb = FAISS.load_local(
    "faiss_index", embeddings
)
# docs = new_vector_store.similarity_search("qux")


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ask_question', methods=['POST'])
def answer():
    global vectordb  # Access the global variable
    if vectordb is None:
        return jsonify({'response': 'No document uploaded yet.'})
    
    data = request.get_json()
    question = data.get('question', '')

    results = vectordb.similarity_search_with_score(question, k=2)
    context = "\n".join([doc.page_content for doc, _ in results])
    response = ask_question(question, context)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run()
    
