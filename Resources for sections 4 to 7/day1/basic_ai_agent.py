### Basic Ai Agent with WEB UI

import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Load AI Model
llm = OllamaLLM(model="mistral")  # Change to "llama3" or another Ollama model

# Initialize Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()  # Stores user-AI conversation history

# Define AI Chat Prompt
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous conversation: {chat_history}\nUser: {question}\nAI:"
)

# Function to run AI chat with memory
def run_chain(question):
    # Retrieve past chat history
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in st.session_state.chat_history.messages])

    # Run the AI response generation
    response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))

    # Store new user input and AI response in memory
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)

    return response

# Streamlit UI
st.title("🤖 AI Chatbot with Memory ")
st.write("Ask me anything!")

user_input = st.text_input("📝 Your Question:")
if user_input:
    response = run_chain(user_input)
    st.write(f"**You:** {user_input}")
    st.write(f"**AI:** {response}")

# Show full chat history
st.subheader("📜 Chat History")
for msg in st.session_state.chat_history.messages:
    st.write(f"**{msg.type.capitalize()}**: {msg.content}")






### Basic AI Agent with Memory


# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.prompts import PromptTemplate
# from langchain_ollama import OllamaLLM

# # Load AI Model
# llm = OllamaLLM(model="mistral")  # Change model to "llama3" or another if needed

# # Initialize Memory
# chat_history = ChatMessageHistory()  # Stores user-AI conversation history

# # Define AI Chat Prompt
# prompt = PromptTemplate(
#     input_variables=["chat_history", "question"],
#     template="Previous conversation: {chat_history}\nUser: {question}\nAI:"
# )

# # Function to run AI chat with memory
# def run_chain(question):
#     # Retrieve chat history manually
#     chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history.messages])

#     # Run the AI response generation
#     response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))

#     # Store new user input and AI response in memory
#     chat_history.add_user_message(question)
#     chat_history.add_ai_message(response)

#     return response

# # Interactive CLI Chatbot
# print("\n🤖 AI Chatbot with Memory")
# print("Type 'exit' to stop.")

# while True:
#     user_input = input("\n📝 You: ")
#     if user_input.lower() == "exit":
#         print("\n👋 Goodbye!")
#         break

#     ai_response = run_chain(user_input)
#     print(f"\n🤖 AI: {ai_response}")




### Basic AI Agent

# from langchain_ollama import OllamaLLM

# # Load AI Model from Ollama
# llm = OllamaLLM(model="mistral")

# print("\n Welcome to your AI Agent! Ask me anything.")
# while True:
# 	question = input("Your Question (or type 'exit' to stop): ")
# 	if question.lower() == 'exit':
# 		print("Goodbye!")
# 		break
# 	response = llm.invoke(question)
# 	print("\n AI Response: ", response)