import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Load AI Model
llm = OllamaLLM(model="mistral")  # Change to "llama3" or another Ollama model

# Initialize Memory (LangChain v1.0+)
chat_history = ChatMessageHistory()

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty("rate", 160)  # Adjust speaking speed

# Speech Recognition
recognizer = sr.Recognizer()

# Function to Speak
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to Listen
def listen():
    with sr.Microphone() as source:
        print("\n🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        print(f"👂 You Said: {query}")
        return query.lower()
    except sr.UnknownValueError:
        print("🤖 Sorry, I couldn't understand. Try again!")
        return ""
    except sr.RequestError:
        print("⚠️ Speech Recognition Service Unavailable")
        return ""

# AI Chat Prompt
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous conversation: {chat_history}\nUser: {question}\nAI:"
)

# Function to Process AI Responses
def run_chain(question):
    # Retrieve past chat history manually
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history.messages])

    # Run AI response generation
    response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))

    # Store new user input and AI response in memory
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response)

    return response

# Main Loop
speak("Hello! I am your AI Assistant. How can I help you today?")
while True:
    query = listen()
    if "exit" in query or "stop" in query:
        speak("Goodbye! Have a great day.")
        break
    if query:
        response = run_chain(query)
        print(f"\n🤖 AI Response: {response}")
        speak(response)








