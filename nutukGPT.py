import os
import re
import chromadb
import gradio as gr
import nltk
nltk.download('punkt')
nltk.download('punkt_tab') 

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from openai import OpenAI
from agents import Agent, Runner, trace, ModelSettings
from agents.mcp import MCPServerStdio
from pydantic import BaseModel, Field

MODEL_NAME = "gpt-5-nano"
PDF_PATH = "Nutuk_modern.pdf"
DB_PATH = "./nutuk_chroma_db"
CHUNK_SIZE = 1000
OVERLAP = 100
MAX_TURNS = 10

openai = OpenAI()

from pydantic import BaseModel, Field

# ... existing imports ...

class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )

def rerank(question, documents, metadatas):
    """
    Re-ranks retrieved documents based on relevance to the question.
    """
    print("--- Reranking Chunks ---")
    
    user_prompt = f"The user has asked the following question:\n\n{question}\n\n"
    user_prompt += "Order all the chunks of text by relevance to the question, from most relevant to least relevant.\n"
    user_prompt += "Here are the chunks:\n\n"
    
    for index, doc in enumerate(documents):
        # Using 1-based indexing for the LLM to make it easier
        user_prompt += f"# CHUNK ID {index + 1}:\n{doc}\n\n"
    
    user_prompt += "Reply only with the list of ranked chunk ids."

    messages = [
        {"role": "system", "content": "You are a document re-ranker. You must rank the provided chunks based on relevance to the question."},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # Call OpenAI with Structured Outputs
        response = openai.chat.completions.create(
            model=MODEL_NAME, 
            messages=messages,
            response_format=RankOrder,
        )
        
        reply = response.choices[0].message.content
        order = RankOrder.model_validate_json(reply).order

        final_indices = [i - 1 for i in order]
        
        ranked_docs = [documents[i] for i in final_indices]
        ranked_metas = [metadatas[i] for i in final_indices]
        
        print(f"Reranked Order: {rank_response.order}")
        return ranked_docs, ranked_metas

    except Exception as e:
        print(f"Reranking failed: {e}. Returning original order.")
        return documents, metadatas

def clean_text(text):
    """
    Cleans PDF artifacts: removes newlines, handles hyphenated words 
    at line breaks, and strips extra whitespace.
    """
    # Join words split by hyphens at the end of a line (common in PDFs)
    text = re.sub(r'(\w+)-\s*\n(\w+)', r'\1\2', text)
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=1000, overlap=100):
    """
    Using NLTK Turkish sentence tokenization for better sentence splitting (handles "Gen.", "Prof.", etc.)
    """
    sentences = sent_tokenize(text, language='turkish')
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        
        # If a single sentence is longer than chunk_size, force-split it
        if sentence_len > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            # Add the long sentence as its own chunk or split it
            chunks.append(sentence[:chunk_size])
            continue

        if current_length + sentence_len > chunk_size:
            chunks.append(" ".join(current_chunk))
            
            overlap_text = ""
            overlap_len = 0
            new_start = []
            for s in reversed(current_chunk):
                if overlap_len + len(s) < overlap:
                    new_start.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current_chunk = new_start
            current_length = overlap_len
            
        current_chunk.append(sentence)
        current_length += sentence_len
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def rag_pipeline():
    encoder = SentenceTransformer("selmanbaysan/turkish_embedding_model_fine_tuned")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name="nutuk_collection")

    print("Reading and Cleaning PDF...")
    reader = PdfReader(PDF_PATH)
    
    documents = []
    metadatas = []
    ids = []
    id_count = 0

    for page_num, page in enumerate(reader.pages):
        raw_text = page.extract_text()
        if not raw_text:
            continue
        
        cleaned_text = clean_text(raw_text)        
        page_chunks = chunk_text(cleaned_text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        
        for chunk in page_chunks:
            documents.append(chunk)
            metadatas.append({"source": "Nutuk", "page": page_num + 1})
            ids.append(f"id_{id_count}")
            id_count += 1
            
    print(f"Generated {len(documents)} chunks.")

    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_embeddings = encoder.encode(batch_docs).tolist()

        collection.add(
            embeddings=batch_embeddings, 
            documents=batch_docs,        
            metadatas=batch_metas,
            ids=batch_ids
        )
    print(f"Database saved to {DB_PATH}")

def testing_rag(query):
    client = chromadb.PersistentClient(path=DB_PATH)
    encoder = SentenceTransformer("selmanbaysan/turkish_embedding_model_fine_tuned") # model finetuned on Turkish datasets
    collection = client.get_collection(name="nutuk_collection")

    query_embedding = encoder.encode(query).tolist()

    results = collection.query(query_embeddings=[query_embedding], n_results=5)

    print("--- Retrieving Context ---")
    for i, doc in enumerate(results['documents'][0]):
        page_num = results['metadatas'][0][i]['page']
        print(f"[Page {page_num}]: {doc}" + "\n\n")

async def chat(message, history):

    client = chromadb.PersistentClient(path=DB_PATH)
    encoder = SentenceTransformer("selmanbaysan/turkish_embedding_model_fine_tuned")
    collection = client.get_collection(name="nutuk_collection")

    query_embedding = encoder.encode(message).tolist()
    # results = collection.query(query_embeddings=[query_embedding], n_results=5)

    # Increase n_results slightly (e.g., to 10) to give the reranker more candidates to choose from
    results = collection.query(query_embeddings=[query_embedding], n_results=10)
    initial_docs = results['documents'][0]
    initial_metas = results['metadatas'][0]

    # --- RERANKING STEP ---
    ranked_docs, ranked_metas = rerank(message, initial_docs, initial_metas)
    
    # Slice to keep only the top 5 after reranking if you want to save context window
    ranked_docs = ranked_docs[:5]
    ranked_metas = ranked_metas[:5]

    print("Retrieving Context (Post-Rerank)")
    retrieved_context = ""
    for i, doc in enumerate(ranked_docs):
        page_num = ranked_metas[i]['page']
        retrieved_context += f"[Sayfa {page_num}]: {doc}" + "\n"

    # print("--- Retrieving Context ---")
    # retrieved_context = ""
    # for i, doc in enumerate(results['documents'][0]):
    #     page_num = results['metadatas'][0][i]['page']
    #     retrieved_context += f"[Sayfa {page_num}]: {doc}" + "\n"

    system_prompt = f"""

    Sen, Mustafa Kemal Atatürk'ün ölümsüz eseri "Nutuk" üzerine uzmanlaşmış bir asistansın. Görevin, sana \
    sağlanan metin parçalarını (bağlamı) kullanarak kullanıcı sorularına yanıt vermek.

    Buna ek olarak "Web Search" aracını kullanarak internetteki bilgileri de kullan. 
    Arama yaparken sayfa numarası, saat veya çok spesifik metin parçalarını sorguya dahil etme. \
    Sorgularını "olay adı + kişi" gibi genel anahtar kelimelerle oluştur.

    ### Temel İlkelerin:
    1. **Sadakat ve Hiyerarşi:** Yanıtlarını öncelikle sana verilen bağlam (context) içindeki bilgilere dayandır. \
    2. **Üslup:** Resmi, saygılı, net ve Cumhuriyet vizyonuna uygun bir dil kullan. Nutuk'taki olayları anlatırken \
    Atatürk'ün perspektifini yansıt (Örn: "Metne göre, Paşa bu durumu şöyle aktarıyor...").
    3. **Atıf Yapma:** Nutuk metninden aldığın bilgilerin sayfa numarasını mutlaka belirt (Örn: Sayfa 444). Web aramasından \
    gelen bilgiler için ise "Web aramasına göre..." ifadesini kullan.
    4. **Çelişki Yönetimi:** Eğer kullanıcı sorusu, bağlamdaki bilgiler ve web sonuçları çelişiyorsa, Nutuk metnini esas al ve \
    "Nutuk metnine göre durum şöyledir:" diyerek açıkla.

    ### Yanıt Formatı:
    - Yanıtlarını maddeler halinde veya kısa, öz paragraflarla yapılandır.
    - Alıntı yaparken çift tırnak kullan ve kronolojik sırayı takip et.

    Sana sunulan metin parçaları aşağıdadır:
    ---------------------
    {retrieved_context}
    ---------------------
    """
    
    print(system_prompt)


    processed_history = []
    for msg in history:
        processed_history.append({"role": msg['role'], "content": msg['content']})

    # Append the new user message
    processed_history.append({"role": "user", "content": message})

    env = {"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")}
    params = {"command": "npx", "args": ["-y", "@brave/brave-search-mcp-server"], "env": env}

    async with MCPServerStdio(params=params, client_session_timeout_seconds=30) as mcp_server:
        agent = Agent(name="agent", instructions=system_prompt, model=MODEL_NAME, mcp_servers=[mcp_server], model_settings=ModelSettings(tool_choice="required"))
        with trace("nutukgpt"): # tracing and monitoring the agent
            result = await Runner.run(agent, processed_history)
        return result.final_output

if __name__ == "__main__":
    # rag_pipeline() 
    # chat("cumhuriyetin ilanı nasıl oldu")
    gr.ChatInterface(chat, type="messages").launch()