import torch
from langchain_core.prompts import ChatPromptTemplate , PromptTemplate
from langchain_huggingface import HuggingFacePipeline ,HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM , BitsAndBytesConfig , pipeline
from langchain_community.vectorstores import FAISS
import io
from gtts import gTTS
import base64

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

def load_llm():
  
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config
    )

    # Create Transformers Pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1, 
        top_p=0.95,
        repetition_penalty=1.15,
        return_full_text=False 
    )

    return HuggingFacePipeline(pipeline=pipe)
llm = load_llm()


DB_FAISS_PATH = "vectorstore/db_faiss"
# Setting up the prompt

custom_prompt_template = """

You are a Medical AI Assistant specialized in Oncology.

IMPORTANT:
If the user provides a passage and asks to explain it,
you MUST explain ONLY that passage.
Do NOT switch to any other topic.

Use retrieved medical context ONLY to support or clarify
the SAME concept mentioned in the user text.

----------------------------
RULES:
----------------------------

1. Use ONLY the provided context.
2. Do NOT use outside knowledge.
3. If information is missing, reply exactly:
"I am sorry, but the provided medical documents do not contain specific information about this."

4. Default response style:
- Clear, structured paragraphs.
- Bullet points ONLY if the user explicitly asks.

5. Do NOT diagnose cancer or suggest treatment.
If asked, say:
"A cancer diagnosis or treatment decision can only be made by a qualified oncologist after proper medical evaluation."

6. End EVERY response with:
"Disclaimer: This is AI-generated information for educational purposes. Please consult an oncologist for medical decisions."

----------------------------
Context:
{context}

User Text / Question:
{question}

Answer:

"""

def set_custom_prompt():
    template = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return template


# Storing the above created emeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"

# Creating the embedding
embedding_model = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH , embedding_model , allow_dangerous_deserialization=True)


# Creating the QAchain

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", 
    retriever=db.as_retriever(search_kwargs={'k': 3}), 
    return_source_documents=True, 
    chain_type_kwargs={"prompt": set_custom_prompt()}
)


# 4. Function to get response
def get_medical_response(query):
    response = qa_chain.invoke({"query": query})
    answer = response["result"]
    sources = response["source_documents"]
    return answer, sources





# --- Testing the Chatbot ---
if __name__ == "__main__":
    question = "MOLECULAR GENETICS OF HEREDITARY CANCER PREDISPOSITION SYNDROMES"
    print(f"\nQuestion: {question}")
    
    answer, metadata = get_medical_response(question)
    
    print("\n--- AI Response ---")
    print(answer)