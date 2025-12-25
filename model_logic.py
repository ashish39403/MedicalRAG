import torch
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
DB_FAISS_PATH = "vectorstore/db_faiss"

def get_qa_chain():
    # 1. GPU Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.float16, quantization_config=bnb_config
    )

    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, 
        max_new_tokens=512, temperature=0.1, repetition_penalty=1.15, return_full_text=False
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # 2. Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    # 3. Fixed Prompt (Leakage Proof)
    template = """
    
<s>[INST] You are an expert Medical AI Assistant specialized in Oncology.
STRICT RULES:
1. Answer ONLY from the provided Context. Do NOT use outside knowledge.
2. FORMATTING: If the user asks for points (e.g., "in 5 points"), you MUST provide exactly that format.
3. DEEP EXPLANATION: If the user provides a passage and asks to explain/elaborate, focus ONLY on that specific text.
4. If the answer is not in the context, say: "I am sorry, but the provided medical documents do not contain specific information about this."

Context:
{context}

History:
{chat_history}

Question:
{question} [/INST]
Answer:


"""

    prompt = PromptTemplate(template=template, input_variables=["context", "chat_history", "question"])

    # 4. Memory Integration
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )