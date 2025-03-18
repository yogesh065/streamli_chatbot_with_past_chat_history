import time
import os
import joblib
import streamlit as st
from groq import Groq
from collections import deque
api_key=os.getenv("GROQ_API_KEY")
api_key=st.secrets["key_api"]["GROQ_API_KEY"]
st.set_page_config(page_title="Chat with AI-Yogesh", layout='wide')


new_chat_id = f'{time.time()}'
MODEL_ROLE = 'assistant'
AI_AVATAR_ICON = '✨'

# Create a data/ folder if it doesn't already exist
os.makedirs('data/', exist_ok=True)

# Load past chats (if available)
try:
    past_chats: dict = joblib.load('data/past_chats_list')
except FileNotFoundError:
    past_chats = {}

# Sidebar allows a list of past chats
with st.sidebar:
    st.write('# Past Chats')
    if st.session_state.get('chat_id') is None:
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id] + list(past_chats.keys()),
            format_func=lambda x: past_chats.get(x, 'New Chat'),
            placeholder='_',
        )
    else:
        # This will happen the first time AI response comes in
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()),
            index=1,
            format_func=lambda x: past_chats.get(x, 'New Chat' if x != st.session_state.chat_id else st.session_state.chat_title),
            placeholder='_',
        )
    # Save new chats after a message has been sent to AI
    # TODO: Give user a chance to name chat
    st.session_state.chat_title = f'ChatSession-{st.session_state.chat_id}'

st.write('# Chat with Yogesh')

# Function to load chat history safely
def load_chat_history(chat_id):
    try:
        messages = joblib.load(f'data/{chat_id}-st_messages')
        groq_history = joblib.load(f'data/{chat_id}-groq_messages')
        return messages, groq_history
    except FileNotFoundError:
        return [], []

# Load chat history (allows to ask multiple questions)
st.session_state.messages, st.session_state.groq_history = load_chat_history(st.session_state.chat_id)

# Initialize the model
st.session_state.model = Groq(api_key=api_key)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(name=message['role'], avatar=message.get('avatar')):
        st.markdown(message['content'])

# React to user input
def main():
    if prompt := st.chat_input('Your message here...'):
        # Save this as a chat for later
        if st.session_state.chat_id not in past_chats.keys():
            # Use the first 10 characters of user input as chat title
            chat_title = prompt[:50].strip() or "ChatSession"
            past_chats[st.session_state.chat_id] = chat_title
            joblib.dump(past_chats, 'data/past_chats_list')
        # Display user message in chat message container
        with st.chat_message('user'):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append(dict(role='user', content=prompt))
        
        # Prepare the messages for the model by combining the session history and new message
        model_messages = [{"role": message['role'], "content": message['content']} for message in st.session_state.messages]
        
        # Send message to Groq
        client = st.session_state.model
        completion = client.chat.completions.create(
            model="llama-3.3-70b-specdec",
            messages=model_messages,
            temperature=1,
            max_tokens=8000,
            top_p=1,
            stream=True,
            stop=None,
        )
        message_placeholder = st.empty()
        full_response = ''

        for chunk in completion:
            if chunk.choices[0].delta.content:
                for ch in chunk.choices[0].delta.content:
                    full_response += ch
                    time.sleep(0.00001)
                    message_placeholder.write(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append(
            dict(role=MODEL_ROLE, content=full_response, avatar=AI_AVATAR_ICON)
        )
        st.session_state.groq_history = deque(completion)  # Assign a serializable object
        
        # Save to file (only serializable parts)
        joblib.dump(st.session_state.messages, f'data/{st.session_state.chat_id}-st_messages')
        joblib.dump(list(st.session_state.groq_history), f'data/{st.session_state.chat_id}-groq_messages')  # Save as list

if __name__ == "__main__":
    main()
