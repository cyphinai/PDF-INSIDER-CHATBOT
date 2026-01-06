# RAG ChatBot - Conversational PDF Question Answering System

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based exclusively on the content of a PDF document. Built with Streamlit, LangChain, and Groq API, featuring **conversational memory** for handling follow-up questions.

## 🎯 What Does This Agent Do?

This chatbot:
- **Reads and understands PDF documents** using advanced text processing
- **Answers questions ONLY from the PDF content** - it won't use outside knowledge
- **Retrieves relevant sections** from the document to provide accurate answers
- **Maintains conversation history** for seamless chat experience and follow-up questions
- **Handles contextual references** like "it", "that", "the previous example"
- **Refuses to answer** questions not covered in the PDF to prevent hallucinations

## ✨ Key Features

- 📄 **PDF document processing** with intelligent text chunking
- 🔍 **Semantic search** using FAISS vector store with HuggingFace embeddings
- 💬 **Conversational memory** - remembers previous questions and answers
- 🔄 **Context-aware responses** - understands follow-up questions and references
- 🚫 **Strict context-based responses** (no external knowledge)
- ⚡ **Fast responses** using Groq's LLM API with Llama 3.1
- 🧹 **Memory management** - automatically manages conversation history
- 🔍 **Debug view** - inspect conversation history in sidebar

## 🛠️ Technology Stack

- **Streamlit** - Interactive web interface
- **LangChain** - RAG and conversation management framework
- **FAISS** - Vector database for semantic search
- **HuggingFace Embeddings** - Text embeddings (all-MiniLM-L12-v2)
- **Groq API** - LLM inference (llama-3.1-8b-instant)
- **PyPDF** - PDF processing library
- **Conversation Buffer Memory** - Maintains chat history context

## 📋 Prerequisites

- Python 3.8 or higher
- Groq API key ([Get it here](https://console.groq.com/))
- Basic knowledge of Python and command line

## 🚀 Installation & Setup

### 1. Clone and Set Up
```bash
git clone https://github.com/yourusername/rag-conversational-chatbot.git
cd rag-conversational-chatbot
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install streamlit langchain langchain-groq langchain-community
pip install faiss-cpu sentence-transformers pypdf python-dotenv
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Configure PDF Path
In the main Python file, update the PDF file path to your document:
```python
pdf_file = "/path/to/your/document.pdf"  # Line 40 in the code
```

## 🎮 Running the Application

### 1. Start the Application
```bash
streamlit run app.py
```

### 2. Access the Interface
- Open your browser and go to `http://localhost:8501`

### 3. Start Chatting
- Type your question in the chat input at the bottom
- The bot will provide answers based on your PDF document
- Use follow-up questions naturally (see examples below)

## 🗣️ Conversation Flow Examples

### Example 1: Follow-up Questions
```
You: What is Python?
Assistant: Python is a high-level, interpreted programming language...

You: What are its main features?
Assistant: (Understands "its" refers to Python and answers based on context)
```

### Example 2: References
```
You: Explain functions in Python
Assistant: Functions are reusable blocks of code...

You: Give me an example of that
Assistant: (Understands "that" refers to functions and provides example)
```

### Example 3: Questions Outside PDF Scope
```
You: What's the capital of France?
Assistant: I cannot answer this question based on the provided document.
```

## 🔧 Configuration Options

You can customize these parameters in the code:

### Text Processing
- **Chunk size**: `chunk_size=1000` (adjust for document complexity)
- **Chunk overlap**: `chunk_overlap=100` (ensures context continuity)
- **Embedding model**: `model_name="all-MiniLM-L12-v2"` (alternatives: `all-mpnet-base-v2`)

### Retrieval Settings
- **Number of chunks**: `search_kwargs={'k': 3}` (increase for complex topics)
- **Similarity search**: FAISS with cosine similarity

### Conversation Settings
- **History length**: Last 5 Q&A pairs (prevents context overflow)
- **Memory type**: Conversation buffer memory with chat history

### LLM Settings
- **Model**: `model="llama-3.1-8b-instant"` (Groq model)
- **Temperature**: Default (controlled via Groq API)

## 📂 Project Structure

```
rag-conversational-chatbot/
│
├── app.py                      # Main application file
├── requirements.txt           # Python dependencies
├── .env                      # Environment variables (not committed)
├── README.md                 # This documentation
└── assets/                   # Optional: Screenshots, icons
    ├── demo.gif
    └── architecture.png
```

## 📦 Requirements

Create a `requirements.txt` file with:
```txt
streamlit>=1.28.0
langchain>=0.1.0
langchain-groq>=0.1.0
langchain-community>=0.0.10
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
pypdf>=3.17.0
python-dotenv>=1.0.0
```

## 🔍 How It Works

### 1. **Document Processing**
- PDF is loaded and split into manageable chunks
- Text embeddings are created using HuggingFace models
- Vector store (FAISS) is built for efficient similarity search

### 2. **Conversation Management**
- Each Q&A pair is stored in conversation history
- History is automatically managed (keeps last 5 exchanges)
- Context is passed to LLM for understanding follow-up questions

### 3. **Question Answering**
- User question triggers semantic search in vector store
- Relevant document chunks are retrieved
- LLM generates answer using both document context and conversation history
- Response is validated against document content only

### 4. **Memory Management**
- Session state maintains conversation across interactions
- Automatic cleanup prevents context overflow
- Clear conversation button for resetting context

## 🚨 Error Handling

The application includes error handling for:
- Missing API keys
- PDF loading failures
- Vector store initialization errors
- Network connectivity issues
- Invalid questions or queries

## 🧪 Testing

Test the system with different types of questions:

1. **Direct questions** about document content
2. **Follow-up questions** using pronouns
3. **Multi-step queries** requiring context
4. **Out-of-scope questions** to test refusal capability
5. **Technical terminology** specific to your document

## 📊 Performance Tips

1. **For large PDFs**: Increase `chunk_size` to 1500-2000
2. **For technical documents**: Use `k=4` or `k=5` for more context
3. **For faster responses**: Consider using smaller embedding models
4. **For better accuracy**: Fine-tune chunk overlap based on document structure

## 🔄 Updating the Document

To use a different PDF:
1. Update the file path in the code
2. Restart the Streamlit application
3. Clear conversation history (sidebar button)
4. The new document will be processed automatically

## 🛡️ Security Notes

- API keys are stored in `.env` file (add to `.gitignore`)
- Session data is stored locally in browser
- No external data storage or tracking
- Document processing happens locally

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### Areas for Improvement:
- Add support for multiple PDFs
- Implement document upload feature
- Add citation of source sections
- Include confidence scores for answers
- Add export conversation feature

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors & Contributors

- **Cyphr AI** - Initial work - [cyphrai](https://github.com/cyphrai)
- **Contributors** - See contributors list

## 🙏 Acknowledgments

- **LangChain Team** for the comprehensive RAG framework
- **Groq** for high-performance LLM inference
- **Streamlit** for making web apps accessible
- **HuggingFace** for open-source embedding models
- **Meta** for the Llama models

## 📚 Resources & References

- [LangChain Documentation](https://python.langchain.com/docs/)
- [Groq Cloud Documentation](https://console.groq.com/docs)
- [FAISS Documentation](https://faiss.ai/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [HuggingFace Models](https://huggingface.co/models)

## 🔗 Related Projects

- [LangChain RAG Templates](https://github.com/langchain-ai/langchain-templates)
- [Streamlit Chat Elements](https://github.com/streamlit/streamlit-chat)
- [FAISS Examples](https://github.com/facebookresearch/faiss/wiki)

---

**Note**: This chatbot is designed for educational and research purposes. Always verify critical information from the original source documents. The system is optimized for text-based PDFs; scanned/image PDFs may require OCR preprocessing.

---

*Last Updated: December 2023*  
*Version: 2.0 (Conversational Edition)*
