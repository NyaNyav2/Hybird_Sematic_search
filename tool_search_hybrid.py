import os
import json
import openai
from flask import Flask, request, jsonify, abort
from opensearchpy import OpenSearch, RequestsHttpConnection
from typing import List
import re
import numpy as np
import tiktoken
import time
from dotenv import load_dotenv
import cohere 

# Load biến môi trường
load_dotenv()

# ===================== CONFIG =====================
# 1. Cấu hình OpenSearch
OPENSEARCH_HOST = os.getenv('OPENSEARCH_HOST', '113.190.241.201')
OPENSEARCH_PORT = int(os.getenv('OPENSEARCH_PORT', '9200'))
OPENSEARCH_PROTOCOL = os.getenv('OPENSEARCH_PROTOCOL', 'http')
OPENSEARCH_USER = os.getenv('OPENSEARCH_USER', None)
OPENSEARCH_PASS = os.getenv('OPENSEARCH_PASS', None)

# Tên Index - KHỚP VỚI INDEX CỦA BẠN
INDEX_NAME_DATA = "embed_van_ban"

# Pipeline name
SEARCH_PIPELINE_NAME = "hybrid-norm-pipeline"

# 2. Cấu hình AI Models
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
TOKEN_LIMIT = 8192
encoding = tiktoken.encoding_for_model("text-embedding-3-large")

# 3. Search Params
TOP_K_HYBRID_DATA = 100
TOP_K_FALLBACK = 10
TOP_K_RERANK = 10

# 4. Hybrid Search Weights
BM25_WEIGHT = 0.3  # Trọng số cho keyword search
VECTOR_WEIGHT = 0.7  # Trọng số cho vector search

# ===================== Flask init =====================
app = Flask(__name__)

# ===================== OpenSearch Client =====================
auth = (OPENSEARCH_USER, OPENSEARCH_PASS) if OPENSEARCH_USER else None

client = OpenSearch(
    hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
    http_auth=auth,
    scheme=OPENSEARCH_PROTOCOL,
    use_ssl=False,
    verify_certs=False,
    ssl_show_warn=False,
    connection_class=RequestsHttpConnection,
    timeout=60
)

print(f"🔗 Connecting to OpenSearch at {OPENSEARCH_HOST}:{OPENSEARCH_PORT}")

# ===================== Setup Search Pipeline =====================
def setup_search_pipeline():
    """
    Tạo normalization pipeline cho hybrid search
    Chỉ chạy 1 lần khi khởi động
    """
    pipeline_body = {
        "description": "Hybrid search with normalization and weighted combination",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {
                        "technique": "min_max"
                    },
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {
                            "weights": [BM25_WEIGHT, VECTOR_WEIGHT]
                        }
                    }
                }
            }
        ]
    }
    
    try:
        try:
            client.http.get(f"/_search/pipeline/{SEARCH_PIPELINE_NAME}")
            print(f"✅ Pipeline '{SEARCH_PIPELINE_NAME}' đã tồn tại")
        except:
            client.http.put(
                f"/_search/pipeline/{SEARCH_PIPELINE_NAME}",
                body=pipeline_body
            )
            print(f"✅ Đã tạo pipeline '{SEARCH_PIPELINE_NAME}'")
            print(f"   📊 Weights: BM25={BM25_WEIGHT}, Vector={VECTOR_WEIGHT}")
    except Exception as e:
        print(f"⚠️  Không thể tạo pipeline: {e}")
        print(f"   💡 Search vẫn hoạt động nhưng không có normalization")

setup_search_pipeline()

# ===================== Utils Functions =====================
def preprocess_text(text):
    """Tiền xử lý text cho search"""
    text = re.sub(r"[^\w\s]", " ", str(text).lower())
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_vector(vector):
    """Chuẩn hóa vector về unit length"""
    vector_array = np.array(vector)
    norm = np.linalg.norm(vector_array)
    if norm == 0:
        return vector_array.tolist()
    return (vector_array / norm).tolist()

def safe_text_for_embedding(text):
    """Đảm bảo text không vượt quá token limit"""
    processed_text = preprocess_text(text)
    tokens = encoding.encode(processed_text)
    if len(tokens) > TOKEN_LIMIT:
        tokens = tokens[:TOKEN_LIMIT]
        processed_text = encoding.decode(tokens)
    return processed_text

def get_embedding(text, max_retries=3):
    """Tạo embedding với retry logic"""
    safe_text = safe_text_for_embedding(text)
    for attempt in range(max_retries):
        try:
            response = openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=safe_text
            )
            raw_embedding = response.data[0].embedding
            return raw_embedding
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"⚠️  Lỗi embedding (thử lại sau {wait_time}s): {e}")
                time.sleep(wait_time)
            else:
                print(f"❌ Lỗi tạo embedding sau {max_retries} lần thử: {e}")
                return None

def validate_request():
    """Validate request body"""
    if not request.is_json:
        abort(400, "Request body must be JSON")
    data = request.get_json()
    text = data.get("text")
    if not text:
        abort(400, "Missing 'text' field")
    return text

# ===================== Init Cohere Rerank =====================
def init_cohere_reranker():
    """Khởi tạo Cohere reranker"""
    try:
        co = cohere.ClientV2(api_key=COHERE_API_KEY)  # ✅ Dùng ClientV2
        print("✅ Cohere Reranker initialized")
        return co
    except Exception as e:
        print(f"⚠️  Failed to initialize Cohere reranker: {e}")
        return None

cohere_client = init_cohere_reranker()  # ✅ Đổi tên biến

# ===================== CORE SEARCH LOGIC =====================
def execute_opensearch_hybrid(index_name, query_text, query_vector, top_k):
    """
    Thực hiện Hybrid Search trên OpenSearch 2.15
    ĐÃ ĐIỀU CHỈNH CHO INDEX embed_van_ban
    """
    
    body = {
        "size": top_k,
        # ✅ CẬP NHẬT: Lấy đúng các trường có trong embed_van_ban
        "_source": [
            "chunk_id", "doc_id", "chunk", "chunk_type", 
            "index", "original_p"
        ],
        "query": {
            "hybrid": {
                "queries": [
                    # Query 1: BM25 keyword search
                    # ✅ CẬP NHẬT: Search trong trường "chunk" thay vì "content"
                    {
                        "match": {
                            "chunk": {
                                "query": query_text
                            }
                        }
                    },
                    # Query 2: Vector semantic search
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_vector,
                                "k": top_k
                            }
                        }
                    }
                ]
            }
        }
    }

    try:
        response = client.search(
            index=index_name, 
            body=body,
            params={"search_pipeline": SEARCH_PIPELINE_NAME}
        )
        return response['hits']['hits']
    except Exception as e:
        print(f"⚠️  Search với pipeline thất bại, thử không pipeline: {e}")
        try:
            response = client.search(index=index_name, body=body)
            return response['hits']['hits']
        except Exception as e2:
            print(f"❌ OpenSearch Error on {index_name}: {e2}")
            raise e2

def parse_hits_to_documents(hits):
    """
    ✅ CẬP NHẬT: Chuyển đổi kết quả cho cấu trúc embed_van_ban
    Returns: (documents_list, response_data_list)
    """
    documents = []
    response_data = []
    
    for hit in hits:
        source = hit['_source']
        score = hit['_score']
        
        # ✅ Lấy các trường từ embed_van_ban
        chunk_id = source.get('chunk_id', hit.get('_id', ''))
        doc_id = source.get('doc_id', '')
        chunk = source.get('chunk', '')
        chunk_type = source.get('chunk_type', '')
        chunk_index = source.get('index', 0)
        original_p = source.get('original_p', '')
        
        # Chuẩn bị text cho Rerank
        # Ưu tiên chunk (đã được phân tích), fallback về original_p
        if chunk:
            doc_text = chunk
        elif original_p:
            doc_text = original_p
        else:
            doc_text = chunk_id  # Worst case
            
        documents.append(preprocess_text(doc_text))
        
        # ✅ Chuẩn bị object kết quả theo format embed_van_ban
        response_data.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "chunk": chunk,
            "chunk_type": chunk_type,
            "index": chunk_index,
            "original_p": original_p,
            "score": score,
            # Thêm markdown để dễ hiển thị
            "markdown": f"[{chunk_type.upper()}] {chunk[:100]}..." if len(chunk) > 100 else f"[{chunk_type.upper()}] {chunk}"
        })
        
    return documents, response_data

def apply_rerank(query_text, documents, response_data, search_method):
    """
    ✅ SỬA LẠI: Áp dụng Cohere rerank với SDK mới
    Returns: (final_data, rerank_success)
    """
    final_data = []
    rerank_success = False
    
    if cohere_client and documents:
        try:
            print(f"🔄 Reranking {len(documents)} documents...")
            
            # ✅ Gọi API Cohere rerank
            rerank_response = cohere_client.rerank(
                model="rerank-multilingual-v3.0",
                query=query_text,
                documents=documents,
                top_n=min(TOP_K_RERANK, len(documents))
            )
            
            # ✅ Parse kết quả
            for result in rerank_response.results:
                final_data.append({
                    **response_data[result.index],
                    "rerank_score": float(result.relevance_score),
                    "search_method": f"{search_method} + rerank"
                })
            
            rerank_success = True
            print(f"✅ Rerank thành công, trả về {len(final_data)} kết quả")
            
        except Exception as e:
            print(f"⚠️  Rerank failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Fallback nếu rerank không thành công
    if not rerank_success:
        final_data = [{
            **item,
            "search_method": search_method
        } for item in response_data[:TOP_K_FALLBACK]]
        print(f"📋 Dùng top {len(final_data)} kết quả gốc (không rerank)")
    
    return final_data, rerank_success

# ===================== API ENDPOINTS =====================

@app.route("/search_hybrid_data", methods=["POST"])
def hybrid_search_data():
    """Search trong embed_van_ban index"""
    text = validate_request()
    new_text = preprocess_text(text)
    print(f"\n{'='*60}")
    print(f"📊 HYBRID SEARCH - embed_van_ban")
    print(f"Query: {new_text}")
    print(f"{'='*60}")

    try:
        # 1. Tạo embedding
        dense_embedding = get_embedding(new_text)
        if not dense_embedding:
            abort(500, "Failed to create embedding")
        
        # 2. Search OpenSearch
        hits = execute_opensearch_hybrid(
            INDEX_NAME_DATA, 
            new_text, 
            dense_embedding, 
            TOP_K_HYBRID_DATA
        )
        print(f"🔍 Tìm thấy {len(hits)} chunks từ OpenSearch")
        
        # 3. Parse kết quả
        documents, response_data = parse_hits_to_documents(hits)
        
        # 4. Rerank
        final_data, rerank_success = apply_rerank(
            new_text, 
            documents, 
            response_data, 
            "embed_van_ban_hybrid"
        )

        return jsonify({
            "query": text,
            "index": INDEX_NAME_DATA,
            "total_results": len(final_data),
            "rerank_success": rerank_success,
            "results": final_data
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        abort(500, str(e))


@app.route("/search_by_doc_id", methods=["POST"])
def search_by_doc_id():
    """
    ✅ MỚI: Tìm kiếm trong một văn bản cụ thể
    Body: {"text": "...", "doc_id": "66b9c07d3ab9c4ae3d5f793f"}
    """
    if not request.is_json:
        abort(400, "Request body must be JSON")
    
    data = request.get_json()
    text = data.get("text")
    doc_id = data.get("doc_id")
    
    if not text or not doc_id:
        abort(400, "Missing 'text' or 'doc_id' field")
    
    new_text = preprocess_text(text)
    print(f"\n{'='*60}")
    print(f"📄 SEARCH IN DOCUMENT: {doc_id}")
    print(f"Query: {new_text}")
    print(f"{'='*60}")
    
    try:
        # 1. Tạo embedding
        dense_embedding = get_embedding(new_text)
        if not dense_embedding:
            abort(500, "Failed to create embedding")
        
        # 2. Search với filter doc_id
        body = {
            "size": TOP_K_HYBRID_DATA,
            "_source": [
                "chunk_id", "doc_id", "chunk", "chunk_type", 
                "index", "original_p"
            ],
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "bool": {
                                "must": [
                                    {"match": {"chunk": {"query": new_text}}}
                                ],
                                "filter": [
                                    {"term": {"doc_id": doc_id}}
                                ]
                            }
                        },
                        {
                            "bool": {
                                "must": [
                                    {
                                        "knn": {
                                            "embedding": {
                                                "vector": dense_embedding,
                                                "k": TOP_K_HYBRID_DATA
                                            }
                                        }
                                    }
                                ],
                                "filter": [
                                    {"term": {"doc_id": doc_id}}
                                ]
                            }
                        }
                    ]
                }
            }
        }
        
        response = client.search(
            index=INDEX_NAME_DATA,
            body=body,
            params={"search_pipeline": SEARCH_PIPELINE_NAME}
        )
        
        hits = response['hits']['hits']
        print(f"🔍 Tìm thấy {len(hits)} chunks trong doc {doc_id}")
        
        # 3. Parse và rerank
        documents, response_data = parse_hits_to_documents(hits)
        final_data, rerank_success = apply_rerank(
            new_text, 
            documents, 
            response_data,
            f"doc_filter_{doc_id}"
        )
        
        return jsonify({
            "query": text,
            "doc_id": doc_id,
            "index": INDEX_NAME_DATA,
            "total_results": len(final_data),
            "rerank_success": rerank_success,
            "results": final_data
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        abort(500, str(e))


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        info = client.info()
        
        # Lấy thông tin index
        index_stats = client.indices.stats(index=INDEX_NAME_DATA)
        doc_count = index_stats['_all']['primaries']['docs']['count']
        
        return jsonify({
            "status": "healthy",
            "opensearch": {
                "connected": True,
                "version": info.get("version", {}).get("number", "unknown"),
                "cluster": info.get("cluster_name", "unknown")
            },
            "index": {
                "name": INDEX_NAME_DATA,
                "document_count": doc_count
            },
            "cohere_rerank": cohere_client is not None,
            "pipeline": SEARCH_PIPELINE_NAME
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


# ===================== MAIN =====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 STARTING HYBRID SEARCH API - embed_van_ban")
    print("="*60)
    print(f"📌 Index       : {INDEX_NAME_DATA}")
    print(f"📌 Pipeline    : {SEARCH_PIPELINE_NAME}")
    print(f"📌 BM25 Weight : {BM25_WEIGHT}")
    print(f"📌 Vector Weight: {VECTOR_WEIGHT}")
    print(f"📌 Top-K       : {TOP_K_HYBRID_DATA}")
    print(f"📌 Rerank Top-K: {TOP_K_RERANK}")
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=5031, threaded=True)