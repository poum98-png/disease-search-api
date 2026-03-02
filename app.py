import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from openai import OpenAI
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
DEFAULT_TOPK = int(os.getenv("DEFAULT_TOPK", "5"))

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY 환경변수를 설정하세요.")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경변수를 설정하세요.")

app = Flask(__name__)
CORS(app)  # 아임웹에서 호출할 거라 CORS 허용(초기엔 전체 허용)

sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
oa = OpenAI(api_key=OPENAI_API_KEY)

def embed_text(text: str) -> list[float]:
    text = (text or "").strip()
    if not text:
        return []
    res = oa.embeddings.create(model=EMBED_MODEL, input=text)
    return res.data[0].embedding

def vec_to_pgvector_literal(vec: list[float]) -> str:
    # pgvector는 "[0.1,0.2,...]" 형태 문자열 캐스팅이 가장 안전합니다.
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/admin/embed-all")
def admin_embed_all():
    # 초기 1회: diseases 테이블에서 embedding이 NULL인 행들만 임베딩 채움
    token = request.args.get("token", "")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        return jsonify({"error": "unauthorized"}), 401

    # embedding이 비어있는 행만 가져오기
    resp = (
        sb.table("diseases")
        .select("id, embed_text")
        .is_("embedding", "null")
        .limit(2000)
        .execute()
    )

    rows = resp.data or []
    updated = 0
    failed = []

    for r in rows:
        disease_id = r.get("id")
        text = r.get("embed_text") or ""
        try:
            vec = embed_text(text)
            if not vec:
                failed.append({"id": disease_id, "reason": "empty embed_text"})
                continue

            vec_literal = vec_to_pgvector_literal(vec)

            sb.table("diseases").update({"embedding": vec_literal}).eq("id", disease_id).execute()
            updated += 1
        except Exception as e:
            failed.append({"id": disease_id, "reason": str(e)})

    return jsonify({
        "total_candidates": len(rows),
        "updated": updated,
        "failed": failed
    })

@app.get("/search")
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "q is required"}), 400

    k = request.args.get("k", str(DEFAULT_TOPK))
    try:
        k = int(k)
    except:
        k = DEFAULT_TOPK
    k = max(1, min(10, k))

    qvec = embed_text(q)
    qvec_literal = vec_to_pgvector_literal(qvec)

    # Supabase RPC: match_diseases(query_embedding, match_count)
    resp = sb.rpc("match_diseases", {
        "query_embedding": qvec_literal,
        "match_count": k
    }).execute()

    results = resp.data or []
    # score 높은 순으로 이미 정렬됨(함수에서 order by 거리)
    return jsonify({
        "query": q,
        "topk": k,
        "results": results
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
