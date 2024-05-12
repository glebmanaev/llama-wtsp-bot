import os
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

import pinecone
from pinecone import Pinecone, ServerlessSpec

from sentence_transformers import SentenceTransformer

from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationChain
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as LC_Pinecone
from langchain.memory import ConversationBufferWindowMemory



load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


#create LlamaCpp model
llama_path = 'llama-2-7b-chat.Q4_0.gguf'
llama = LlamaCpp(
    model_path=llama_path,
    n_gpu_layers = -1,
    n_batch = 2048,
    n_ctx=4096,
    verbose=False
    )

#create conversation chain
prompt = PromptTemplate(input_variables=['history', 'input'], 
template='The following is a conversation between a human and an AI. The AI provides relatively short but informative answers using details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.\n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:'
)
chain = ConversationChain(
    llm=llama,
    memory=ConversationBufferWindowMemory(k=3),
    prompt=prompt,
    verbose=False
    )

memories_store = {} # user_id : ConversationBufferWindowMemory(k=3) for each user

#init pinecone and create index
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = 'medical-encyclopedia-768-100'
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768, #1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )
index = pc.Index(index_name)

#load the medical encyclopedia data 
file_path = 'parsed_AZ_Family_Medical_encyclopedia.txt'
chunk_size = 100
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()
# Split text into chunks
words = text.split()
chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
data = pd.DataFrame({
    'chunk': chunks,
    'chunk-id': range(len(chunks)),
    'source': 'medical_encyclopedia',
    'title': 'Medical Encyclopedia'
    })

#Embed the mdeical data
model = SentenceTransformer('sentence-t5-large')
batch_size = 100

for i in tqdm(range(0, len(data), batch_size), desc="Embedding medical data"):
    i_end = min(len(data), i + batch_size)
    batch = data.iloc[i:i_end]
    ids = [f"{x['source']}-{x['chunk-id']}" for _, x in batch.iterrows()] # generate unique ids for each chunk
    texts = batch['chunk'].tolist()
    embeds = model.encode(texts, show_progress_bar=False)
    metadata = [{'text': x['chunk']} for _, x in batch.iterrows()]


#upsert the data
embeddings = HuggingFaceEmbeddings(model_name='sentence-t5-large')
vectorstore = LC_Pinecone(
    index=index,
    embedding=embeddings,
    text_key="chunk"
)


def add_medical_context(query, k=3):
    """Finds the top k most relevant medical context for a given query and appends it to the query."""
    query_vector = model.encode([query])[0].tolist()
    medical_context_raw = index.query(vector=query_vector, top_k=k, include_metadata=True)
    medical_context = '.'.join([match['metadata']['text'] for match in medical_context_raw['matches']])
    query_w_context = f"Context:{medical_context}\nQuestion:{query}"
    return query_w_context


def generate_response(message):
    global memories_store
    
    name = message.split(":")[0]

    if name not in memories_store:
        memories_store[name] = ConversationBufferWindowMemory(k=3)
    memory = memories_store[name]
    chain.memory = memory

    message_body = message.split(":")[1]
    message_body = add_medical_context(message_body)
    new_message = chain.run(message_body)
    memories_store[name] = chain.memory

    new_message = f"Hey, {name}!" + new_message if name not in memories_store else new_message

    return new_message


def main():
    while True:
        message = input("Enter your name and message: ")
        response = generate_response(message)
        print(response)
        time.sleep(1)


if __name__ == "__main__":
    main()

