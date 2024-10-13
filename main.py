import streamlit as st
#from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import openai
import os
import pickle


# Load environment variable which has API Key
#load_dotenv()
hf_key = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Sidebar for PDF upload and query input
st.sidebar.title("Upload & Query")
file = st.sidebar.file_uploader("Upload a PDF file", type='pdf')

# Display a logo or title for the app
st.title("DeepStudy: AI-powered Research Mentor")

# Chat history for multiple interactions
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []


if file is not None:
    reader = PdfReader(file)
    file_name = file.name[:-4]
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text=text) 
    st.write(f"Total characters in all text chunks: {sum(len(chunk) for chunk in chunks)}")
    
    pickle_file = f"{file_name}.pkl"
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as file:
            vector_store = pickle.load(file)
        st.write(f"The existing pickle file {pickle_file} is loaded from the disk.")
    else:
        # Using HuggingFace Instruct model
        response = SentenceTransformer("hkunlp/instructor-xl")
        embeddings = response.encode(chunks)
        text_embeddings = list(zip(chunks, embeddings))
        # Storing in the FAISS vector database and also into a pickle file with GPU support
        vector_store = FAISS.from_embeddings(text_embeddings=text_embeddings,embedding=response.encode)
        with open(pickle_file, "wb") as file:
            pickle.dump(vector_store, file)

    # User query input with a submit button
    query = st.sidebar.text_input(f"Query about {file_name}.pdf:")
    submit_query = st.sidebar.button("Submit Query")


    if submit_query and query:
        docs = vector_store.similarity_search(query=query, k=3)
        llm = HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            model_kwargs={
                "max_new_tokens": 512,
                "top_k": 30,
                "temperature": 0.6,
                "repetition_penalty": 1.03,
            },
        )
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        response = chain.run(input_documents=docs,question=query) 
        
        # Save query and response to chat history
        st.session_state['chat_history'].append((query, response))

        # Display chat history
        if st.session_state['chat_history']:
            st.subheader("Chat History")
            for i, (q, r) in enumerate(st.session_state['chat_history']):
                st.markdown(f"**User:** {q}")
                st.markdown(f"**Model:** {r}")
                st.markdown("---")  # Divider for clarity
else:
    st.write("Upload a PDF file to extract text.")
