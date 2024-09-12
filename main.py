import ollama
import chromadb
import os
from langchain_ollama import OllamaLLM
from datetime import datetime

# Initialise models and database
def initialise_models():
    client = chromadb.Client()
    conversation_collection = client.create_collection(name="conversation_embeddings")
    response_model = OllamaLLM(model="llama3.1")
    embedding_model = "mxbai-embed-large"
    return conversation_collection, response_model, embedding_model

# Generate time prefix in "hhmmss_" format
def generate_time_prefix():
    return datetime.now().strftime("%H%M%S_")

# Summarise context for memory optimisation (every 10 responses)
def summarise_context(context, limit=20):
    context_lines = context.split("\n")
    return "\n".join(context_lines[-limit:]) if len(context_lines) > limit else context

# Store embeddings in the vector database
def store_embeddings(text, conversation_id, embedding_model, collection):
    embedding = ollama.embeddings(model=embedding_model, prompt=text)["embedding"]
    collection.add(ids=[conversation_id], embeddings=[embedding], documents=[text])

# Retrieve the relevant conversation context using the vector database
def retrieve_relevant_context(user_input, embedding_model, collection, n_results=1):
    user_embedding = ollama.embeddings(model=embedding_model, prompt=user_input)["embedding"]
    results = collection.query(query_embeddings=[user_embedding], n_results=n_results)
    return "\n".join([doc[0] for doc in results["documents"]])

# Append text to a log file with UTF-8 encoding
def log_to_file(file_path, text):
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(text)

# Initialise log files if not present
def init_log_files(prefix, conversation_log, summarised_log):
    conversation_log = f"{prefix}{conversation_log}"
    summarised_log = f"{prefix}{summarised_log}"
    for file in [conversation_log, summarised_log]:
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as log_file:
                log_file.write(f"{os.path.basename(file)}\n\n")
    return conversation_log, summarised_log

# Chatbot conversation logic
def handle_conversation():
    # Initialise models
    conversation_collection, response_model, embedding_model = initialise_models()

    # Generate time prefix for unique log file names
    time_prefix = generate_time_prefix()
    conversation_log, summarised_log = init_log_files(time_prefix, "conversation_log.txt", "summarised_conversation_log.txt")

    context, interaction_count = "", 0
    template = "Answer the question below.\n\nHere is the relevant conversation history: {context}\n\nQuestion: {question}\n\nAnswer:\n"

    print("Welcome to the AI Chatbot! Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        try:
            interaction_count += 1

            # Store user input embeddings
            store_embeddings(user_input, f"user_{interaction_count}", embedding_model, conversation_collection)

            # Retrieve relevant context and generate response
            relevant_context = retrieve_relevant_context(user_input, embedding_model, conversation_collection)
            full_prompt = template.format(context=relevant_context, question=user_input)
            result = response_model.invoke(full_prompt)

            # Store bot response embeddings
            store_embeddings(result, f"bot_{interaction_count}", embedding_model, conversation_collection)

            # Log the conversation and display bot response
            log_to_file(conversation_log, f"User: {user_input}\nAI: {result}\n")
            print("Bot:", result)

            # Update conversation context
            context += f"\nUser: {user_input}\nAI: {result}"

            # Summarise and log every 10 interactions
            if interaction_count % 10 == 0:
                summarised_context = summarise_context(context)
                log_to_file(summarised_log, f"\nSummarised after {interaction_count} interactions:\n{summarised_context}\n")
                context = summarised_context

        except Exception as e:
            print(f"Sorry, I encountered an error: {str(e)}. Let me try again.")

if __name__ == "__main__":
    handle_conversation()
