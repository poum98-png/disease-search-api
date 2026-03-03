import os
import json

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from openai import OpenAI
from supabase import create_client
from collections import defaultdict

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

def build_query_text_from_units(units: list[dict], key: str) -> str:
    """
    units 배열에서 특정 카테고리(key)의 값들을 모아 임베딩용 텍스트를 만든다.
    - 중복 제거
    - 너무 길면 일부만 사용
    """
    vals = []
    seen = set()

    for u in units or []:
        items = u.get(key, [])
        if not isinstance(items, list):
            continue
        for it in items:
            s = str(it).strip()
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            vals.append(s)

    # 너무 길면 앞부분만 사용(임베딩 안정/비용 절감)
    MAX_ITEMS = int(os.getenv("SMART_MAX_ITEMS_PER_CAT", "20"))
    vals = vals[:MAX_ITEMS]

    if not vals:
        return ""

    # 포맷 고정(가벼운 정리): "key: a, b, c"
    return f"{key}: " + ", ".join(vals)

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

@app.post("/classify")
def classify():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "text is required"}), 400

    # 너무 긴 입력은 비용/품질 위해 컷(원하면 길이 조정 가능)
    MAX_CHARS = int(os.getenv("CLASSIFY_MAX_CHARS", "800"))
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    system_prompt = """
너는 치과 증상 문장을 구조화하는 정보추출기다.
입력은 사용자가 자유롭게 쓴 한국어 텍스트이며, 마침표/줄바꿈이 없을 수 있다.

너의 작업은 2단계다:

1) 입력을 "의미 단위"로 1~8개로 분리한다.
   - 의미 단위는 증상/위치/자극요인/시간경과/부정/배경 등이 바뀌는 지점이다.
   - 마침표나 줄바꿈이 없어도 적절히 분리하라.
   - 지나치게 잘게 쪼개지 말고, 각 단위가 의미를 가지게 하라.
   - 분리한 각 단위의 원문을 text 필드에 그대로 넣어라.

2) 각 의미 단위마다 아래 9개 카테고리만 추출해, "반드시 JSON만" 출력하라.

카테고리(반드시 이 키만 사용):
- location: 해부학적 위치/부위(치아, 어금니, 앞니, 치아 사이, 잇몸, 턱관절, 혀, 볼 안쪽 등)
- laterality: 방향/측면(왼쪽, 오른쪽, 위, 아래, 한쪽, 양쪽 등)  ※ 위치의 방향 정보만
- pain_pattern: 통증 양상/감각(찌릿, 욱신, 시림, 둔통, 날카로움, 박동성, 자발통, 지속/간헐, 저림 등)
- trigger: 유발/악화 요인(씹을 때, 찬 것/찬물/찬바람, 뜨거운 것, 단 것, 양치, 눌렀을 때, 입 벌릴 때, 밤에 심해짐 등)
- associated_signs: 동반 증상/징후(붓기, 출혈, 고름, 입냄새, 음식물 끼임, 검은 반점, 치아 깨짐, 흔들림, 열감 등)
- time_course: 시간/경과(갑자기, 며칠째, 몇 주째, 점점 심해짐, 밤/아침에 더 심함, 반복 등)
- severity_urgency: 강도/응급 신호(참기 힘듦, 잠 못 잠, 얼굴 붓기, 발열, 삼키기/호흡 어려움 등)
- negation: 부정/없음 정보("~없어요/~아니에요/~괜찮아요" 등)
- context: 배경/원인 단서(최근 치료/시술, 임플란트/교정, 외상(딱딱한 거 씹다), 이갈이, 스케일링 후 등)

구어체/은유 표현 인식 강화 규칙(중요):
- 환자들이 사용하는 구어체/은유적 표현도 "통증 또는 감각 이상"으로 적극 인식하라.
- 문맥상 아프다/불편하다/이상하다를 표현하는 말이면 pain_pattern에 포함하라.
- 특히 아래 표현들은 pain_pattern 후보로 간주하라(의학적 진단을 추측하지 말고, 감각 표현으로만 분류):
  - "솟아오른 느낌", "솟구친 느낌", "확 올라오는 느낌", "치받치는 느낌"
  - "우리해요/우리하다" (둔하고 은근한 통증/불편감 표현으로 취급)
  - "찌르르", "저릿저릿", "찌릿찌릿", "지릿", "뻐근", "묵직", "땡김", "당김"
  - "화끈", "따끔", "얼얼", "감각이 이상", "신경 쓰임"
- 위 표현이 들어오면 가능하면 원문 표현을 그대로 pain_pattern에 넣어라.
  (예: pain_pattern: ["우리해요", "솟아오른 느낌"])
- 단, 추측하지 말고 입력에 있는 표현만 사용하라.

추가 규칙:
- 추측하지 말고, 입력에 명시된 정보만 추출하라.
- 각 카테고리는 문자열 배열(list of strings)로 반환하라. 없으면 [].
- JSON 외의 어떤 텍스트도 출력하지 말라.
- 반드시 아래 출력 형식만 사용하라.

출력 형식(반드시 준수):
{
  "units": [
    {
      "text": "...",
      "location": [],
      "laterality": [],
      "pain_pattern": [],
      "trigger": [],
      "associated_signs": [],
      "time_course": [],
      "severity_urgency": [],
      "negation": [],
      "context": []
    }
  ]
}
"""

    try:
        response = oa.chat.completions.create(
            model=os.getenv("CLASSIFY_MODEL", "gpt-4.1-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0
        )

        content = response.choices[0].message.content.strip()

        # 혹시 ```json ... ``` 형태면 제거
        if content.startswith("```"):
            content = content.strip("`")
            content = content.replace("json", "", 1).strip()

        obj = json.loads(content)

        # 안전 보정
        units = obj.get("units", [])
        if not isinstance(units, list):
            units = []

        # 최대 8개로 제한
        MAX_UNITS = int(os.getenv("CLASSIFY_MAX_UNITS", "8"))
        units = units[:MAX_UNITS]

        keys = [
            "location", "laterality", "pain_pattern", "trigger",
            "associated_signs", "time_course", "severity_urgency",
            "negation", "context"
        ]

        normalized_units = []
        for u in units:
            if not isinstance(u, dict):
                continue
            nu = {"text": str(u.get("text", "")).strip()}
            for k in keys:
                v = u.get(k, [])
                nu[k] = v if isinstance(v, list) else []
            # text가 비어있으면 제외
            if nu["text"]:
                normalized_units.append(nu)

        return jsonify({
            "ok": True,
            "input": text,
            "units": normalized_units
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/smart-search")
def smart_search():
    """
    입력 텍스트 → /classify(의미 단위 분리+분류) → 4개 카테고리 임베딩 →
    match_disease_vectors RPC를 카테고리별로 호출 → 질환별 점수 합산 → TopK 반환
    """
    data = request.get_json(silent=True) or {}
    q = (data.get("text") or "").strip()
    if not q:
        return jsonify({"error": "text is required"}), 400

    # (선택) 기존 차단/재질문 로직도 그대로 재사용 가능
    if looks_non_symptom(q):
        return jsonify({
            "query": q,
            "blocked": True,
            "message": "증상(불편감) 기반 질문에 대해서만 안내가 가능합니다. 예) 치아가 시려요, 잇몸이 붓고 피나요, 씹을 때 아파요",
            "results": []
        }), 200

    if looks_too_vague(q):
        return jsonify({
            "query": q,
            "blocked": True,
            "message": "어느 부위가 언제부터, 어떤 상황에서 더 불편한가요? (예: 오른쪽 아래 잇몸이 3일 전부터 붓고 씹을 때 아파요 / 찬물 마시면 특정 치아가 찌릿해요)",
            "results": []
        }), 200

    # TopK
    topk = int(data.get("k") or DEFAULT_TOPK)
    topk = max(1, min(50, topk))

    # 카테고리별로 가져올 벡터 수
    fetch_k = int(data.get("fetch_k") or os.getenv("SMART_FETCH_K", "30"))
    fetch_k = max(10, min(100, fetch_k))

    # 카테고리 가중치(후보 제시형 추천값)
    W_LOCATION = float(os.getenv("W_LOCATION", "1.0"))
    W_PAIN     = float(os.getenv("W_PAIN", "0.9"))
    W_TRIGGER  = float(os.getenv("W_TRIGGER", "0.8"))
    W_ASSOC    = float(os.getenv("W_ASSOC", "0.6"))

    # 1) LLM 분류 호출: 내부 함수처럼 /classify 로직을 재사용하기 위해
    #    현재 app.py에 /classify가 있으니, HTTP로 다시 부르지 말고 "직접 호출"이 가장 깔끔하지만
    #    구조를 크게 바꾸지 않으려면 여기서는 oa.chat을 한 번 더 호출하지 않고,
    #    /classify가 이미 구현되어 있다는 전제에서 "같은 프롬프트"를 함수화해서 재사용하는 게 이상적입니다.
    #
    # 지금은 간단하게: classify()가 반환하는 형식과 동일하게,
    # 프론트에서 먼저 /classify 호출 후 units를 넘겨도 됩니다.
    #
    # → 여기서는 data에 units가 있으면 그걸 사용하고,
    #   없으면 /classify에 해당하는 LLM 호출을 한 번 수행하도록 합니다.

    units = data.get("units")
    if not isinstance(units, list):
        # units가 없으면, 기존 /classify와 동일 프롬프트로 1회 호출해서 units 생성
        # (당신 app.py의 /classify와 동일한 system_prompt를 그대로 쓰는 게 베스트)
        # 여기서는 /classify에서 쓰는 system_prompt를 CLASSIFY_SYSTEM_PROMPT 환경변수로 빼는 방식을 추천하지만,
        # 빠르게 동작하게 하려고 간단히 호출합니다.

        classify_system_prompt = os.getenv("CLASSIFY_SYSTEM_PROMPT", "").strip()
        if not classify_system_prompt:
            return jsonify({
                "error": "units가 없고, CLASSIFY_SYSTEM_PROMPT 환경변수도 비어있습니다. 먼저 /classify 결과 units를 보내거나 CLASSIFY_SYSTEM_PROMPT를 설정하세요."
            }), 400

        try:
            resp = oa.chat.completions.create(
                model=os.getenv("CLASSIFY_MODEL", "gpt-4.1-mini"),
                messages=[
                    {"role": "system", "content": classify_system_prompt},
                    {"role": "user", "content": q}
                ],
                temperature=0
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.strip("`")
                content = content.replace("json", "", 1).strip()
            obj = json.loads(content)
            units = obj.get("units", [])
            if not isinstance(units, list):
                units = []
        except Exception as e:
            return jsonify({"error": f"classify failed: {str(e)}"}), 500

    # 2) 4개 카테고리별 임베딩 텍스트 생성
    q_loc   = build_query_text_from_units(units, "location")
    q_pain  = build_query_text_from_units(units, "pain_pattern")
    q_trig  = build_query_text_from_units(units, "trigger")
    q_assoc = build_query_text_from_units(units, "associated_signs")

    # 전부 비면 검색 불가
    if not any([q_loc, q_pain, q_trig, q_assoc]):
        return jsonify({
            "query": q,
            "blocked": True,
            "message": "증상 정보를 더 알려주세요. (예: 어디가 아픈지 / 어떤 느낌인지 / 언제 더 심한지)",
            "results": [],
            "units": units
        }), 200

    # 3) 카테고리별 임베딩 생성
    vec_loc   = embed_text(q_loc)   if q_loc else None
    vec_pain  = embed_text(q_pain)  if q_pain else None
    vec_trig  = embed_text(q_trig)  if q_trig else None
    vec_assoc = embed_text(q_assoc) if q_assoc else None

    # 4) RPC 호출 helper
    def rpc_match(vec, vtype: str):
        if not vec:
            return []
        vec_literal = vec_to_pgvector_literal(vec)
        resp = sb.rpc("match_disease_vectors", {
            "query_embedding": vec_literal,
            "match_count": fetch_k,
            "filter_vector_type": vtype
        }).execute()
        return resp.data or []

    rows_loc   = rpc_match(vec_loc, "location")
    rows_pain  = rpc_match(vec_pain, "pain_pattern")
    rows_trig  = rpc_match(vec_trig, "trigger")
    rows_assoc = rpc_match(vec_assoc, "associated_signs")

    # 5) 질환별 점수 합산
    scores = defaultdict(float)
    meta = {}  # disease_id -> {title,url}

    def add_rows(rows, weight: float):
        for r in rows:
            did = r.get("disease_id")
            if not did:
                continue
            score = float(r.get("score", 0.0) or 0.0)
            scores[did] += score * weight
            if did not in meta:
                meta[did] = {
                    "disease_id": did,
                    "title": r.get("title", ""),
                    "url": r.get("url", "")
                }

    add_rows(rows_loc,   W_LOCATION)
    add_rows(rows_pain,  W_PAIN)
    add_rows(rows_trig,  W_TRIGGER)
    add_rows(rows_assoc, W_ASSOC)

    # 6) 정렬 + TopK
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked = ranked[:topk]

    results = []
    for did, sc in ranked:
        m = meta.get(did, {"disease_id": did, "title": "", "url": ""})
        results.append({
            "id": did,
            "title": m["title"],
            "url": m["url"],
            "score": round(sc, 6)  # 디버깅용 (원하면 제거)
        })

    return jsonify({
        "query": q,
        "blocked": False,
        "results": results,
        "units": units,
        "debug": {
            "q_location": q_loc,
            "q_pain_pattern": q_pain,
            "q_trigger": q_trig,
            "q_associated_signs": q_assoc,
            "weights": {
                "location": W_LOCATION,
                "pain_pattern": W_PAIN,
                "trigger": W_TRIGGER,
                "associated_signs": W_ASSOC
            },
            "fetch_k": fetch_k
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)











