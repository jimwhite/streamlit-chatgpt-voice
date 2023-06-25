import streamlit as st
import openai
import os
import base64
import glob
import json
import mistune
import pytz
import math
import requests
import time

from datetime import datetime
from openai import ChatCompletion
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup
from collections import deque
from audio_recorder_streamlit import audio_recorder

def generate_filename(prompt, file_type):
    central = pytz.timezone('US/Central')
    safe_date_time = datetime.now(central).strftime("%m%d_%I%M")  
    safe_prompt = "".join(x for x in prompt if x.isalnum())[:45]
    return f"{safe_date_time}_{safe_prompt}.{file_type}"

def transcribe_audio(openai_key, file_path, model):
    OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {openai_key}",
    }
    with open(file_path, 'rb') as f:
        data = {'file': f}
        response = requests.post(OPENAI_API_URL, headers=headers, files=data, data={'model': model})
    if response.status_code == 200:
        st.write(response.json())
        
        response2 = chat_with_model(response.json().get('text'), '') # *************************************
        st.write('Responses:')
        #st.write(response)
        st.write(response2)
        return response.json().get('text')
    else:
        st.write(response.json())
        st.error("Error in API call.")
        return None

def save_and_play_audio(audio_recorder):
    audio_bytes = audio_recorder()
    if audio_bytes:
        filename = generate_filename("Recording", "wav")
        with open(filename, 'wb') as f:
            f.write(audio_bytes)
        st.audio(audio_bytes, format="audio/wav")
        return filename
    return None

def create_file(filename, prompt, response):
    if filename.endswith(".txt"):
        with open(filename, 'w') as file:
            file.write(f"{prompt}\n{response}")
    elif filename.endswith(".htm"):
        with open(filename, 'w') as file:
            file.write(f"{prompt}   {response}")
    elif filename.endswith(".md"):
        with open(filename, 'w') as file:
            file.write(f"{prompt}\n\n{response}")
            
def truncate_document(document, length):
    return document[:length]
def divide_document(document, max_length):
    return [document[i:i+max_length] for i in range(0, len(document), max_length)]

def get_table_download_link(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    b64 = base64.b64encode(data.encode()).decode()  
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1]  # get the file extension
    if ext == '.txt':
        mime_type = 'text/plain'
    elif ext == '.py':
        mime_type = 'text/plain'
    elif ext == '.xlsx':
        mime_type = 'text/plain'
    elif ext == '.csv':
        mime_type = 'text/plain'
    elif ext == '.htm':
        mime_type = 'text/html'
    elif ext == '.md':
        mime_type = 'text/markdown'
    else:
        mime_type = 'application/octet-stream'  # general binary data type
    href = f'<a href="data:{mime_type};base64,{b64}" target="_blank" download="{file_name}">{file_name}</a>'
    return href

def CompressXML(xml_text):
    root = ET.fromstring(xml_text)
    for elem in list(root.iter()):
        if isinstance(elem.tag, str) and 'Comment' in elem.tag:
            elem.parent.remove(elem)
    return ET.tostring(root, encoding='unicode', method="xml")
    
def read_file_content(file,max_length):
    if file.type == "application/json":
        content = json.load(file)
        return str(content)
    elif file.type == "text/html" or file.type == "text/htm":
        content = BeautifulSoup(file, "html.parser")
        return content.text
    elif file.type == "application/xml" or file.type == "text/xml":
        tree = ET.parse(file)
        root = tree.getroot()
        xml = CompressXML(ET.tostring(root, encoding='unicode'))
        return xml
    elif file.type == "text/markdown" or file.type == "text/md":
        md = mistune.create_markdown()
        content = md(file.read().decode())
        return content
    elif file.type == "text/plain":
        return file.getvalue().decode()
    else:
        return ""


def chat_with_model(prompt, document_section, model_choice='gpt-3.5-turbo'):
    model = model_choice
    conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    conversation.append({'role': 'user', 'content': prompt})
    if len(document_section)>0:
        conversation.append({'role': 'assistant', 'content': document_section})
    #response = openai.ChatCompletion.create(model=model, messages=conversation)

    # streaming response
    #result_textarea = st.empty()
    #results=[]
    #for responses in openai.Completion.create(model=model, prompt=conversation, stream=True):
#    for responses in openai.ChatCompletion.create(model=model, messages=conversation, stream=True):
        #results.append(str(responses.choices[0]))
     #   results.append(responses.choices[0].text)
        #st.markdown(f'*{results}*')
     #   result = "".join(results).strip()
     #   result = result.replace('\n','')
     #   result_textarea.markdown(f'*{result}*')

        
    # record the time before the request is sent
    start_time = time.time()

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=conversation,
        temperature=0.5,
        stream=True  # again, we set stream=True
    )
    
    # create variables to collect the stream of chunks
    collected_chunks = []
    collected_messages = []
    # iterate through the stream of events
    for chunk in response:
        chunk_time = time.time() - start_time  # calculate the time delay of the chunk
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk['choices'][0]['delta']  # extract the message
        collected_messages.append(chunk_message)  # save the message
        print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message}")  # print the delay and text

    # print the time delay and text received
    print(f"Full response received {chunk_time:.2f} seconds after request")
    full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
    print(f"Full conversation received: {full_reply_content}")




        
    #return response
    #return response['choices'][0]['message']['content']
    return results
    

def chat_with_file_contents(prompt, file_content, model_choice='gpt-3.5-turbo'):
    conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    conversation.append({'role': 'user', 'content': prompt})
    if len(file_content)>0:
        conversation.append({'role': 'assistant', 'content': file_content})
    response = openai.ChatCompletion.create(model=model_choice, messages=conversation)
    return response['choices'][0]['message']['content']
        
    
def main():
    # Sidebar and global
    openai.api_key = os.getenv('OPENAI_KEY')
    st.set_page_config(page_title="GPT Streamlit Document Reasoner",layout="wide")
    menu = ["htm", "txt", "xlsx", "csv", "md", "py"]  #619
    choice = st.sidebar.selectbox("Output File Type:", menu)
    model_choice = st.sidebar.radio("Select Model:", ('gpt-3.5-turbo', 'gpt-3.5-turbo-0301'))
    
    # Audio, transcribe, GPT:
    filename = save_and_play_audio(audio_recorder)
    if filename is not None:
        transcription = transcribe_audio(openai.api_key, filename, "whisper-1")
        st.write(transcription)
        gptOutput = chat_with_model(transcription, '', model_choice) # *************************************
        filename = generate_filename(transcription, choice)
        create_file(filename, transcription, gptOutput)
        st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)


    user_prompt = st.text_area("Enter prompts, instructions & questions:", '', height=100)

    collength, colupload = st.columns([2,3])  # adjust the ratio as needed
    with collength:
        #max_length = 12000 - optimal for gpt35 turbo. 2x=24000 for gpt4.  8x=96000 for gpt4-32k.
        max_length = st.slider("File section length for large files", min_value=1000, max_value=128000, value=12000, step=1000)
    with colupload:
        uploaded_file = st.file_uploader("Add a file for context:", type=["xml", "json", "xlsx","csv","html", "htm", "md", "txt"])
    
    document_sections = deque()
    document_responses = {}

    if uploaded_file is not None:
        file_content = read_file_content(uploaded_file, max_length)
        document_sections.extend(divide_document(file_content, max_length))

    if len(document_sections) > 0:
        
        if st.button("👁️ View Upload"):
            st.markdown("**Sections of the uploaded file:**")
            for i, section in enumerate(list(document_sections)):
                st.markdown(f"**Section {i+1}**\n{section}")
        
        st.markdown("**Chat with the model:**")
        for i, section in enumerate(list(document_sections)):
            if i in document_responses:
                st.markdown(f"**Section {i+1}**\n{document_responses[i]}")
            else:
                if st.button(f"Chat about Section {i+1}"):
                    st.write('Reasoning with your inputs...')
                    response = chat_with_model(user_prompt, section, model_choice) # *************************************
                    st.write('Response:')
                    st.write(response)
                    document_responses[i] = response
                    filename = generate_filename(f"{user_prompt}_section_{i+1}", choice)
                    create_file(filename, user_prompt, response)
                    st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)

    if st.button('💬 Chat'):
        st.write('Reasoning with your inputs...')
        response = chat_with_model(user_prompt, ''.join(list(document_sections,)), model_choice) # *************************************
        st.write('Response:')
        st.write(response)
        
        filename = generate_filename(user_prompt, choice)
        create_file(filename, user_prompt, response)
        st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)

    all_files = glob.glob("*.*")
    all_files = [file for file in all_files if len(os.path.splitext(file)[0]) >= 20]  # exclude files with short names
    all_files.sort(key=lambda x: (os.path.splitext(x)[1], x), reverse=True)  # sort by file type and file name in descending order

    # sidebar of files
    file_contents=''
    next_action=''
    for file in all_files:
        col1, col2, col3, col4, col5 = st.sidebar.columns([1,6,1,1,1])  # adjust the ratio as needed
        with col1:
            if st.button("🌐", key="md_"+file):  # md emoji button
                with open(file, 'r') as f:
                    file_contents = f.read()
                    next_action='md'
        with col2:
            st.markdown(get_table_download_link(file), unsafe_allow_html=True)
        with col3:
            if st.button("📂", key="open_"+file):  # open emoji button
                with open(file, 'r') as f:
                    file_contents = f.read()
                    next_action='open'
        with col4:
            if st.button("🔍", key="read_"+file):  # search emoji button
                with open(file, 'r') as f:
                    file_contents = f.read()
                    next_action='search'
        with col5:
            if st.button("🗑", key="delete_"+file):
                os.remove(file)
                st.experimental_rerun()
                
    if len(file_contents) > 0:
        if next_action=='open':
            file_content_area = st.text_area("File Contents:", file_contents, height=500)
        if next_action=='md':
            st.markdown(file_contents)
        if next_action=='search':
            file_content_area = st.text_area("File Contents:", file_contents, height=500)
            st.write('Reasoning with your inputs...')
            response = chat_with_file_contents(user_prompt, file_contents)
            st.write('Response:')
            st.write(response)
            filename = generate_filename(file_content_area, choice)
            create_file(filename, file_content_area, response)
            st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)
                
if __name__ == "__main__":
    main()