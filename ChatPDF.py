import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain import FAISS
from langchain.llms import OpenAI
import re
#import json
import openai


load_dotenv() # to get the key
# Define a function to get the text from pdf
def get_pdf_text(pdf_reader, pages=None):
    text_dict = {}
    text = ""
    if pages:
        # Iterate through the pages of the PDF and extract text from each page
        for i, page in enumerate(pdf_reader.pages):
            if i+1 in pages:
                text += f"#### ----- Page {i+1} ----- :\n\n" + page.extract_text()
                text_dict[i+1] = page.extract_text()
    else:
        # Iterate through the pages of the PDF and extract text from each page
        for i, page in enumerate(pdf_reader.pages):
            text += f"#### ----- Page {i+1} ----- :\n\n" + page.extract_text()
            text_dict[i+1] = page.extract_text()
    return text, text_dict

# Define a function to process the extracted text
def get_chunks_dict(text_dict):
    chunks_dict = {}
    for page, content in text_dict.items():
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size = 4000,
            chunk_overlap=100,
            length_function=len
            )
        chunks = text_splitter.split_text(content)
        #print(chunks)
        chunks_dict[page] = chunks 
    return chunks_dict

def get_full_chunks(chunks_dict):
    full_chunks =[]
    for page, chunks in chunks_dict.items():
        for chunk in chunks:
            full_chunks.append(chunk)
    return full_chunks

def find_page(doc, chunks_dict):
    for page, chunks in chunks_dict.items():
        for chunk in chunks:
            if doc == chunk:
                return page

# Define a function to ask question with advanced options 
def answer(question, retrieve_new_page=True, retrieve_page_id=False, retrieve_external_knowledge=False):
            
    if "docs" not in st.session_state: # initialise current_docs
        st.session_state.docs = st.session_state.knowledgeBase.similarity_search(query=question)
        
    if retrieve_new_page: # Process the extracted text to create a new knowledge
        st.session_state.docs = st.session_state.knowledgeBase.similarity_search(query=question)
        st.session_state.page_content = st.session_state.docs[0].page_content
        st.session_state.current_page = find_page(st.session_state.page_content, st.session_state.chunks_dict)
    
    else: # just stay on the current page
        st.session_state.page_content = st.session_state.docs[0].page_content
        
        #find the page what has content needed
        st.session_state.current_page = find_page(st.session_state.page_content, st.session_state.chunks_dict) 
        # use the current page as page_content
        st.session_state.page_content = st.session_state.text_dict[st.session_state.current_page] 
           
          
    if retrieve_page_id: # Process the extract text to create a new knowledge from specific pages
        # Check if retrieve_new_page is selected, then this option will be priortity.
        if retrieve_new_page: # Process the extracted text to create a new knowledge
            st.session_state.docs = st.session_state.knowledgeBase.similarity_search(query=question)
            st.session_state.page_content = st.session_state.docs[0].page_content
            st.session_state.current_page = find_page(st.session_state.page_content, st.session_state.chunks_dict)
        else:
            # Get new text according to new pages
            st.session_state.new_text, st.session_state.new_text_dict = get_pdf_text(st.session_state.pdf_reader, retrieve_page_id)

            # Use new selected pages as content
            st.session_state.page_content = st.session_state.new_text
    
    # Run the question-answering chain to find the answer to the question
    response = st.session_state.chain.run(question = question, docs= st.session_state.page_content)
    response = response.replace("\n", "")
    
            
    if retrieve_external_knowledge: # Get information from external sources
        if retrieve_new_page or retrieve_page_id:
            if "I don't know! You can use Advanced Options to enhance searching." in response:
                #offering users the ability to request information from external sources, extending beyond the confines of the current PDFs
                result = openai.ChatCompletion.create(model = "gpt-3.5-turbo",
                                              messages = [dict(role="user", content = question)])

                response = result.choices[0].message.content
        else:
            #offering users the ability to request information from external sources, extending beyond the confines of the current PDFs
            result = openai.ChatCompletion.create(model = "gpt-3.5-turbo",
                                          messages = [dict(role="user", content = question)])

            response = result.choices[0].message.content
                
    return response

# Set the app title
st.set_page_config(page_title="Chat with PDF", page_icon=":books:", layout="wide")
st.header("Chat with PDF :books:")


# Create the side bar for Help and Upload the documents
with st.sidebar:
    
    # Define the help message
    help_message = """
    Welcome to the Chat with PDF app!
    1. Upload your PDFs in the sidebar.
    2. Click the 'Process' button to extract text from the PDFs.
    3. You can interact with the chatbot by asking questions in the chat interface.
    4. Use the Advanced options to control question answering.
    - Retrieve new page: don't tick it if you want to stay on the current page.
    - Retrieve from pages: provide specific pages for searching (e.g: 2, 3). 
    - Use external knowledge: Tick on this option if you want to retrieve information outside of the pdf document.
    5. Click on Clear Conversation button if you want to reset the chat history.
    6. To close this help message, click the 'Close' button.

    Enjoy your chat with the PDF!
    """
    
    
    show_help = False
    # Create a 'Help' button
    if st.button("Help!"):
        show_help = True

    # Display the help message if the 'Help' button is clicked
    if show_help:
        st.info(help_message)

        # Create a 'Close' button to hide the help message
        if st.button("Close"):
            show_help = False    

        
    
    # Define the PDF uploader    
    pdf_docs = st.file_uploader(
        "Upload your PDF here and click on 'Process'")
    
    if st.button("Process"):
        if pdf_docs:
            st.session_state.pdf_reader = PdfReader(pdf_docs)
            with st.spinner("Processing.This may take a while⏳"):
                # get pdf text
                st.session_state.full_text, st.session_state.text_dict = get_pdf_text(st.session_state.pdf_reader)
                
                st.session_state.chunks_dict = get_chunks_dict(st.session_state.text_dict)

                # get the text chunks
                st.session_state.full_chunks = get_full_chunks(st.session_state.chunks_dict)

                # Print processing done
                st.write("Processing done!")

                # Reset knowledge base
                st.session_state.knowledgeBase = None
        else:
            st.write("Please upload your document!")
     

    # Display the PDF content for reference
    if "full_text" in st.session_state:
        st.write("Your document contents:")
        for page, content in st.session_state.text_dict.items():
            st.write(f" *** Page {page} ***\n")
            st.write(content)

    add_vertical_space()

# Create the conversation place 
col1, col2 = st.columns([1,5])

# Reset button
with col1:
        reset_chat = st.button("Clear Conversation")

# Advanced Options
with col2:
    advanced_options = st.checkbox(
        "Advanced Options", help="Use these options to enhance get better answers"
    )
    retrieve_new_page = True
    retrieve_page_id = False
    retrieve_external_knowledge = False

    if advanced_options:
        #with st.expander("Advanced Options", expanded=False):

        # Create advanced options for query         
        subcol1, subcol2, subcol3 = st.columns(3)
        with subcol1:
            retrieve_new_page = st.checkbox("Retrieve new page", value=True)

        with subcol2:
            retrieve_page_id = st.checkbox("Retrieve from pages (e.g: 3, 4):", value=False)
            if retrieve_page_id:
                numbers = st.text_input("Enter page numbers (e.g: 3, 4)")
                # Use regular expression to find numbers
                numbers = re.findall(r'\d+', numbers)

                # Convert the matched numbers to integers
                retrieve_page_id = [int(num) for num in numbers]


        with subcol3:
            retrieve_external_knowledge = st.checkbox("Use external knowledge", value=False)



# Define the system prompt
st.session_state.system_prompt = PromptTemplate(
    input_variables = ["question", "docs"],
    template = """
        You are a heplful assistant that can answer question about the uploaded document.
        Answer the following question: {question}
        By searching the following document: {docs}
        Only use the factual information from the document to answer the question.
        Your answer should be detailed.
        If you don't have enough information to answer the question, the answer is ALWAYS "I don't know! You can use Advanced Options to enhance searching." No other answer is allowed.
    """)
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if reset_chat: # Reset chat if clicked
    st.session_state.messages = []
    
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        
        
# Accept user input
if "full_text" in st.session_state:
    if prompt := st.chat_input("Ask any question about the document?"):

        # Initialize embeddings
        if "embeddings" not in st.session_state:
            st.session_state.embeddings= OpenAIEmbeddings()
            

        # Create a knowledge base from the uploaded text
        if st.session_state.knowledgeBase == None:
            st.session_state.knowledgeBase = FAISS.from_texts(st.session_state.full_chunks, st.session_state.embeddings)   

        # Initialize LLM
        if "llm" not in st.session_state:
            st.session_state.llm = OpenAI()
            

        # Create chain
        if "chain" not in st.session_state:
            st.session_state.chain = LLMChain(llm = st.session_state.llm, prompt = st.session_state.system_prompt)    

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            # Get the response
            full_response = answer(prompt, retrieve_new_page, retrieve_page_id, retrieve_external_knowledge)

            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.write("Please upload your document and press Process before asking any questions.")
