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

# "증상 검색"이 아닌 문의(가격/예약/보험/위치 등)를 빠르게 차단하기 위한 키워드
NON_SYMPTOM_KEYWORDS = [
    "가격", "비용", "얼마", "원", "만원", "견적",
    "보험", "실비", "청구", "적용", "서류",
    "예약", "진료시간", "시간", "영업", "휴무",
    "위치", "주소", "주차", "길찾기",
    "이벤트", "할인", "쿠폰",
    "후기", "리뷰",
    "원장", "의사", "스태프",
    "상담", "문의",
    "교정", "라미네이트", "미백", "스케일링"
]

# 너무 짧거나 애매한 입력(“아파요”, “시려요” 등) → 추가 질문 유도
VAGUE_PHRASES = {
    "아파요", "아픈데", "통증", "시려요", "시림", "불편해요",
    "피나요", "붓기", "붓어요", "냄새", "입냄새", "고름"
}

# 증상 관련 단서(이 단어가 하나도 없고 너무 짧으면 애매하다고 판단)
SYMPTOM_HINT_WORDS = [
    "치아", "이", "잇몸", "턱", "관자", "사랑니", "임플란트",
    "씹", "저작", "양치", "치실",
    "찬", "뜨거", "단", "차가", "온도",
    "욱신", "찌릿", "쑤시", "시큰", "뻐근",
    "붓", "출혈", "피", "구취", "입냄새"
]

def looks_non_symptom(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return False
    return any(k in q for k in NON_SYMPTOM_KEYWORDS)

def looks_too_vague(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return True

    # 너무 짧으면(2~3글자 수준) 거의 항상 애매함
    if len(q) <= 3:
        return True

    # "아파요/시려요" 같은 포괄 표현만 있고 단서가 부족하면 애매함
    if q in VAGUE_PHRASES:
        return True

    # 5글자 이하인데(짧은데) 증상 단서 단어가 하나도 없으면 애매함
    if len(q) <= 5 and not any(h in q for h in SYMPTOM_HINT_WORDS):
        return True

    return False


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

    # 1) 비증상 질문 차단 (가격/예약/보험/위치 등)
    if looks_non_symptom(q):
        return jsonify({
            "query": q,
            "blocked": True,
            "message": "증상(불편감) 기반 질문에 대해서만 안내가 가능합니다. 예) 치아가 시려요, 잇몸이 붓고 피나요, 씹을 때 아파요",
            "results": []
        }), 200

    # 2) 너무 짧거나 애매한 질문은 추가 정보 요청(재질문)
    if looks_too_vague(q):
        return jsonify({
            "query": q,
            "blocked": True,
            "message": "어느 부위가 언제부터, 어떤 상황에서 더 불편한가요? (예: 오른쪽 아래 잇몸이 3일 전부터 붓고 씹을 때 아파요 / 찬물 마시면 특정 치아가 찌릿해요)",
            "results": []
        }), 200

    # k 파싱 (UI에서 많이 요청할 수 있게 상한 50으로)
    k = request.args.get("k", str(DEFAULT_TOPK))
    try:
        k = int(k)
    except:
        k = DEFAULT_TOPK
    k = max(1, min(50, k))

    qvec = embed_text(q)
    qvec_literal = vec_to_pgvector_literal(qvec)

    resp = sb.rpc("match_diseases", {
        "query_embedding": qvec_literal,
        "match_count": k
    }).execute()

    results = resp.data or []
    return jsonify({
        "query": q,
        "topk": k,
        "blocked": False,
        "results": results
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)


