# WhatsApp MedEncyclop Chatbot

This is a WhatsApp chatbot based on LLAMA-2-7B model. The chatbot has Retrieval Augmented Generation (RAG) that is based on [A-Z Family Medical Encyclopedia by The British Medical Association](https://archive.org/details/azfamilymedicalencyclopedia) turned into a Pinecone vector database. This allows the chatbot to gather medical information from the encyclopedia and puse it as a context for the conversation.

The WhatsApp chatbot is based on the [python-whatsapp-bot](https://github.com/daveebbelaar/python-whatsapp-bot) repo. Refer to it for WhatsApp API, Meta Business and ngrok setup information.

## LLM setup

Install the required python packages:

```
pip install -r requirements.txt
```

Install the GPU version of Llama.cpp using the following command. Use the `FORCE_CMAKE=1` environment variable to force the use of cmake:

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

Download the quantized LLAMA-2-7B model from Hugging Face:
```
wget https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf
```

Add the Pinecone API key to the .env file:
```
PINECONE_API_KEY=<your-api-key>
```