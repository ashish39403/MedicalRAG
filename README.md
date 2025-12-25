MedicalRAG

Medical AI assistant with FastAPI backend and Streamlit frontend for interactive medical insights.

Features

Interactive web interface built with Streamlit.

FastAPI backend for handling API requests and ML model processing.

Supports retrieval-augmented generation for medical queries.

Displays results as text, tables, or charts.

Modular design: easily extendable for new AI models or datasets.

Tech Stack

Frontend: Streamlit

Backend: FastAPI

AI/ML: Python-based models (LLM or custom models)

Dependencies: Listed in requirements_ui.txt and requirements_backend.txt

Optional GPU support for faster model inference


# SetUp Instructions...

1.) Clone the repository

git clone https://github.com/ashish39403/MedicalRAG.git

cd MedicalRAG


2.) Build the docker image

docker build -t medicalrag .

3.) Run the container

docker run -p 8501:8501 -p 8000:8000 medicalrag
