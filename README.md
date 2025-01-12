# EveWorksAssignment

## Question Answering System with PubMed Dataset, Fine-Tuned Model, and FAISS for Efficient Retrieval

This project implements a question-answering (QA) system using the PubMed dataset, a fine-tuned LLaMA model, and FAISS for efficient similarity-based retrieval of relevant contexts. The goal is to build a system that can provide answers to questions based on medical contexts retrieved from PubMed articles. It uses a fine-tuned LLaMA model, knowledge stored in FAISS indices, and Flask for serving the application as a web service.

## Features

- **Dataset**: The PubMed QA dataset (`pubmed_qa`) is used for fine-tuning the LLaMA model.
- **Model**: Fine-tuned LLaMA-3.2-1B-Instruct model with LoRA applied for efficient training.
- **FAISS Indexing**: Contexts from the dataset are embedded using `sentence-transformers`, and the embeddings are stored in FAISS for efficient similarity search.
- **Web Application**: Flask-based web application allowing users to input questions and receive answers.
- **Deployment**: Flask app is deployed using `ngrok` for external access.

## Prerequisites

Make sure you have the following installed:
- Python 3.7 or higher
- `pip` for package management
- Hugging Face account for model and tokenizer storage
- CUDA-enabled GPU (optional but recommended for training and embedding)


##
**The model is fine-tuned using the following steps:**

Loading the Dataset: The PubMed QA dataset (pubmed_qa) is loaded.
Preprocessing: The dataset is tokenized, with special tokens added for question and context.
Model and Tokenizer: The LLaMA model is loaded with 4-bit quantization using BitsAndBytesConfig, and LoRA is applied for efficient fine-tuning.
Training: The model is fine-tuned using the SFTTrainer class from the trl library with specified training arguments.
The model is fine-tuned and pushed to the Hugging Face Hub for future use.

**FAISS Indexing**

The contexts from the PubMed dataset are embedded using the sentence-transformers model (all-MiniLM-L6-v2). These embeddings are used to create a FAISS index for efficient retrieval of the most relevant context based on a user query.
The context embeddings are stored in a .npy file.
The FAISS index is saved for use in retrieving relevant context for each query.

**Query Answering**

Query Processing: When a user submits a query, it is embedded using the same model and compared to the context embeddings stored in the FAISS index.
Answer Generation: The most relevant context is selected, and the model generates an answer based on the question and context.

**Model Deployment**

The Flask application is deployed locally and can be accessed via ngrok. The app serves as an interactive interface for users to query the fine-tuned LLaMA model.
