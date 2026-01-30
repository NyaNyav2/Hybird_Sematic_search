# Hybrid Semantic Search API (Law Documents)

API tìm kiếm **Hybrid Search (BM25 + Vector Semantic Search)** trên **OpenSearch 2.x**, kết hợp **OpenAI Embedding** và **Cohere Rerank**, tối ưu cho dữ liệu văn bản pháp luật đã được chunk và embedding sẵn trong index `embed_van_ban`.

---

## ✨ Tính năng chính

* 🔍 **Hybrid Search**: kết hợp keyword search (BM25) + semantic search (vector)
* ⚖️ **Weighted Normalization Pipeline** trên OpenSearch
* 🧠 **OpenAI Embedding** (`text-embedding-3-large`, 3072 dims)
* 🔁 **Cohere Rerank** để cải thiện độ liên quan kết quả
* 📄 Tìm kiếm toàn bộ dữ liệu hoặc **theo từng `doc_id`**
* 🚀 REST API viết bằng **Flask**, sẵn sàng chạy Docker

---

## 🏗️ Kiến trúc tổng quan

```
Client
  │
  │  POST /search_hybrid_data
  ▼
Flask API
  │
  ├─ OpenAI Embedding (query)
  ├─ OpenSearch Hybrid Query (BM25 + KNN)
  ├─ Normalization Pipeline (min-max + weighted mean)
  └─ Cohere Rerank (top-k)
  ▼
JSON Response
```

---

## 📦 Yêu cầu hệ thống

* Python >= 3.10
* OpenSearch >= 2.15 (đã có index `embed_van_ban`)
* Docker / Docker Compose (khuyến nghị)
* API Keys:

  * OpenAI
  * Cohere

---

## 📁 Cấu trúc project

```
.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── tool_search_hybrid.py
├── .env            # KHÔNG commit
├── .env.example    # File mẫu
└── README.md
```

---

## ⚙️ Biến môi trường (.env)

```env
# OpenSearch
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_PROTOCOL=http
OPENSEARCH_USER=
OPENSEARCH_PASS=

# AI Keys
OPENAI_API_KEY=sk-xxxx
COHERE_API_KEY=xxxx
```

> ⚠️ **Không commit `.env` lên GitHub**

---

## ▶️ Chạy bằng Docker Compose (Khuyến nghị)

```bash
docker-compose up --build
```

API mặc định chạy tại:

```
http://localhost:5031
```

---

## ▶️ Chạy local (không Docker)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
python tool_search_hybrid.py
```

---

## 🔌 API Endpoints

### 1️⃣ Hybrid Search toàn bộ dữ liệu

**POST** `/search_hybrid_data`

```json
{
  "text": "điều kiện khởi kiện tranh chấp đất đai"
}
```

**Response**:

```json
{
  "query": "...",
  "index": "embed_van_ban",
  "total_results": 10,
  "rerank_success": true,
  "results": [
    {
      "chunk_id": "...",
      "doc_id": "...",
      "chunk": "...",
      "chunk_type": "article",
      "index": 3,
      "score": 1.23,
      "rerank_score": 0.98,
      "search_method": "embed_van_ban_hybrid + rerank"
    }
  ]
}
```

---

### 2️⃣ Tìm kiếm trong một văn bản cụ thể

**POST** `/search_by_doc_id`

```json
{
  "text": "thời hiệu khởi kiện",
  "doc_id": "66b9c07d3ab9c4ae3d5f793f"
}
```

---

### 3️⃣ Health check

**GET** `/health`

```json
{
  "status": "healthy",
  "opensearch": {
    "connected": true,
    "version": "2.15.0"
  },
  "index": {
    "name": "embed_van_ban",
    "document_count": 123456
  },
  "cohere_rerank": true
}
```

---

## ⚖️ Cấu hình Hybrid Search

```python
BM25_WEIGHT = 0.3
VECTOR_WEIGHT = 0.7
TOP_K_HYBRID_DATA = 100
TOP_K_RERANK = 10
```

Normalization pipeline:

* **Min-Max normalization**
* **Arithmetic mean** với trọng số

---

## 🧠 Index OpenSearch yêu cầu

Index `embed_van_ban` cần có các field:

* `chunk_id`
* `doc_id`
* `chunk` (text)
* `chunk_type`
* `index`
* `original_p`
* `embedding` (vector 3072 dims)

---

## 🔐 Bảo mật

* Không commit `.env`
* Rotate API key nếu từng lộ
* GitHub Push Protection đã bật (khuyến nghị)

---

## 🚀 Hướng phát triển

* [ ] Query expansion
* [ ] Multi-index search
* [ ] Streaming response
* [ ] Cache embedding
* [ ] Auth / Rate limit

---

## 👨‍💻 Tác giả

Developed for **Hybrid Semantic Search on Vietnamese Law Documents**.

---

## 📄 License

Private / Internal use
