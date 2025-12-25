from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_logic import get_qa_chain

app = FastAPI(title="Medical RAG API")
qa_bot = None

class QueryRequest(BaseModel):
    question: str

@app.on_event("startup")
async def startup_event():
    global qa_bot
    qa_bot = get_qa_chain()

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        response = qa_bot.invoke({"question": request.question})
        return {
            "answer": response["answer"].strip(),
            "sources": [doc.metadata.get('source', 'Unknown') for doc in response["source_documents"]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))