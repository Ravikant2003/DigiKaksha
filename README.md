
# **DigiKaksha (AI-Based Smart Classroom Management)**

**(The YOUTUBE link for the working of this project : https://youtu.be/s1tvwx_CCRs)**


DigiKaksha is a machine learning-based smart solution designed to assist and streamline tasks in educational institutions. It consists of three main components:

## **Face Detection-Based Attendance Monitoring**
Built using OpenCV, this module enables automatic attendance tracking through facial recognition. A camera setup (either inbuilt or external) is required to capture and identify students’ faces.

## **Student Performance Analysis**
The platform provides a graphical analysis of student performance by visualizing data such as marks and attendance. Graphs are generated using Matplotlib, offering insights into both individual and overall performance.

## **LangChain-Based Study Material Chatbot**
Utilizing LangChain, a retriever-augmented generation (RAG) model, and the Google Gemini API, the chatbot helps students comprehend their notes or study materials. It supports querying uploaded PDFs and generates responses based on their content.

**PDF Processing Pipeline:**
- **Docling Conversion:** Uploaded PDFs and Office documents are converted to Markdown using [Docling](https://github.com/DS4SD/docling), preserving document structure and semantic headers.
- **Markdown Header-Aware Splitting:** Uses LangChain's `MarkdownHeaderTextSplitter` to split by headers (#, ##, ###) for semantic chunking.
- **Recursive Character Splitting:** Further splits long sections into chunks of 1000 characters with 200-character overlap using `RecursiveCharacterTextSplitter`.
- **Embeddings:** Text chunks are embedded using the HuggingFace model **`Qwen/Qwen3-Embedding-0.6B`**, a lightweight yet powerful embedding model optimized for semantic search.
- **Vector Storage:** Embeddings are stored in a Chroma vector database (cosine similarity, persisted in `./VectorDB/`) for efficient retrieval.

**Customization:**
- The embedding model can be overridden via the `EMBED_MODEL_ID` environment variable in your `.env` file.
- **Important:** If you change the embedding model, delete the `./VectorDB/` directory and re-upload your PDFs to regenerate embeddings with the new model.

---

## **Key Technologies Used**
- **LangChain and RAG:** Retrieval-augmented generation for context-aware question answering.
- **Docling:** Advanced PDF-to-Markdown converter preserving document structure.
- **Qwen3 Embeddings:** HuggingFace `Qwen/Qwen3-Embedding-0.6B` model for semantic text embeddings.
- **Chroma DB:** Vector database for efficient similarity search and retrieval.
- **Google Gemini:** LLM (`gemini-2.5-flash`) for generating natural language responses.
- **OpenCV:** For face detection and attendance monitoring.
- **Matplotlib:** For creating graphs and analyzing student performance.
- **MongoDB:** As the database backend to store student data.
- **FastAPI:** High-performance web framework for the backend API.

---

## **Key Components**

### 1. **Face Detection for Attendance:**
- Uses face recognition to mark attendance automatically.
- Stores known student faces and compares them with real-time input to log attendance.
- The attendance is marked with the time in a CSV file.

### 2. **Student Data Analysis:**
- Analyzes student marks and attendance across subjects.
- Provides bar and pie charts for visual insights, including individual student performance or an overall comparison of class averages.

### 3. **LangChain-Powered Chatbot:**
- **Document Conversion:** Uses Docling to convert PDF study materials to Markdown format, preserving structure.
- **Intelligent Chunking:** Applies header-aware splitting followed by recursive character text splitting (chunk_size=1000, chunk_overlap=200).
- **Semantic Embeddings:** Embeds text chunks using the HuggingFace model `Qwen/Qwen3-Embedding-0.6B` (configurable via `EMBED_MODEL_ID` env variable).
- **Vector Storage:** Stores embeddings in Chroma vector database for fast similarity-based retrieval.
- **Context-Aware Responses:** Provides accurate answers to students' questions via a chatbot interface powered by Google Gemini (`gemini-2.5-flash`).

---

## **Requirements**
- Install necessary modules using `pip install -r requirements.txt`.
- Python 3.11.X or below is recommended, as Python 3.12.X may cause compatibility issues.
- Ensure a camera setup is available for face recognition.

### **API Keys Required:**
Create a `.env` file in the project root with the following keys:
```
LANGCHAIN_TRACING_V2=<your-langsmith-api-key>
GOOGLE_API_KEY=<your-google-gemini-api-key>
 
```
- **LANGCHAIN_TRACING_V2:** For LangChain tracing and debugging via LangSmith.
- **GOOGLE_API_KEY:** For Google Gemini model integration (`gemini-2.5-flash`).

---

## **Setup Instructions**
1. Install dependencies via the `requirements.txt` file.
2. Ensure you have a valid `LANGCHAIN_TRACING_V2` and `GOOGLE_API_KEY` to use LangChain and Google APIs.
3. Set up MongoDB for storing and retrieving student data.
4. Run the app using FastAPI, accessible at `http://localhost:8000/`.

---

## **Additional Notes**
- Face detection and attendance logging is handled via `face.py`.
- **PDF Processing:** Uploaded study materials are converted to Markdown using Docling, then chunked and embedded with `Qwen/Qwen3-Embedding-0.6B` into a Chroma vector store.
- **RAG Retrieval:** A retriever system fetches relevant context from the vector store to answer student queries accurately.
- **Performance Graphs:** Graphs for student performance analysis can be viewed via the web interface, allowing individual or class-wide analysis.
- **Evaluation Metrics:** 
  - RAGAS evaluation (faithfulness, answer relevancy, context precision/recall) available at `/evaluate/run`
  - Retrieval quality metrics (Precision@k, Recall@k, F1@k) available at `/evaluate/retrieval`

