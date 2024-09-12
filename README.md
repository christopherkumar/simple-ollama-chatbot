# Chatbot with Contextual Memory and Vector Database

This project implements an AI-powered chatbot that utilizes a vector database for storing conversation embeddings and retrieving relevant context for each conversation. The chatbot is built using the `ollama` and `chromadb` libraries and features contextual memory summarisation, as well as persistent logging for each session with time-stamped logs.

## Features

- **Contextual Memory:** The chatbot stores user inputs and AI responses in a vector database to retrieve relevant context in future interactions.
- **Conversation Summarisation:** After every 10 interactions, the conversation history is summarised to maintain efficient memory usage.
- **Vector Database Storage:** Conversations are stored as embeddings in the ChromaDB database using the `mxbai-embed-large` embedding model.
- **Unique Log Files:** Each conversation session generates two separate log files with a unique time-stamped prefix to avoid overwriting previous logs.
  - `hhmmss_conversation_log.txt`: Stores the full conversation.
  - `hhmmss_summarized_conversation_log.txt`: Stores summarised conversation history after every 10 interactions.

## Requirements

- Python 3.8+
- [ollama](https://github.com/ollama/ollama)
- `chromadb`
- `langchain_ollama`

## Installation

1. Clone the repository:
   ```bash
   git clone <[repository-url](https://github.com/christopherkumar/simple-ollama-chatbot.git)>
   cd chatbot-with-contextual-memory
2. Install the required libraries:
   ```bash
   pip install ollama chromadb langchain_ollama

## Usage

1. Run the chatbot:
  `python main.py`
2. Interact with the chatbot in your terminal:
- The chatbot will respond to your questions and utilise the conversation history to generate relevant responses.
- Type `exit` to end the conversation.
3. Each conversation session generates two log files with a time-stamped prefix (in `hhmmss` format):
- `conversation_log.txt`: Logs the entire conversation.
- `summarised_conversation_log.txt`: Logs the summarised history every 10 interactions.
