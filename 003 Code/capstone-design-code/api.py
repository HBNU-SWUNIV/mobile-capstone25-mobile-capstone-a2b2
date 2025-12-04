# ============================================================
#  api.py — 텍스트/음성 + 차량용 RAG + Google STT + gTTS + 알람 기능 통합본
# ============================================================

import base64
import os, sys, io, contextlib, builtins, re
from io import BytesIO
from typing import Optional
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gtts import gTTS
from urllib.parse import quote

# ============================================================
#  public 폴더 import
# ============================================================
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "public")
if PUBLIC_DIR not in sys.path:
    sys.path.insert(0, PUBLIC_DIR)

def _safe_import(name):
    try:
        return __import__(name)
    except Exception as e:
        print(f"[IMPORT FAIL] {name}: {e}")
        return None

rag_mod   = _safe_import("ask_rag")
ask_mod   = _safe_import("ask")
voice_mod = _safe_import("test_voice_2")

# ============================================================
#  유틸: STDOUT 캡처
# ============================================================
def _call_and_capture(func, *args, **kwargs) -> Optional[str]:
    if not callable(func): return None
    buf = io.StringIO()
    orig_input = builtins.input
    try:
        builtins.input = lambda *a, **k: "n"
        with contextlib.redirect_stdout(buf):
            res = func(*args, **kwargs)
    except Exception as e:
        return f"(내부 오류) {e}"
    finally:
        builtins.input = orig_input

    text = res if isinstance(res, str) else buf.getvalue()
    return (text or "").strip() or None


# ============================================================
#  Google STT
# ============================================================
def stt_from_bytes(raw: bytes, content_type: str = "", language="ko-KR"):
    try:
        from google.cloud import speech
    except:
        print("[STT] google-cloud-speech import 실패")
        return None

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except:
        pass

    ct = (content_type or "").lower()
    encoding = None
    sr = None

    from google.cloud import speech
    if "ogg" in ct:
        encoding = speech.RecognitionConfig.AudioEncoding.OGG_OPUS
    elif "webm" in ct:
        encoding = getattr(speech.RecognitionConfig.AudioEncoding, "WEBM_OPUS", None)
        if encoding is None:
            return None
    elif "wav" in ct:
        encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
        sr = 16000
    else:
        encoding = speech.RecognitionConfig.AudioEncoding.OGG_OPUS

    try:
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=raw)

        cfg = dict(
            encoding=encoding,
            language_code=language,
            enable_automatic_punctuation=True
        )
        if sr:
            cfg["sample_rate_hertz"] = sr

        config = speech.RecognitionConfig(**cfg)
        resp = client.recognize(config=config, audio=audio)
        texts = [
            r.alternatives[0].transcript
            for r in resp.results if r.alternatives
        ]

        return (" ".join(texts)).strip() if texts else None
    except Exception as e:
        print("[STT 오류]", e)
        return None


# ============================================================
#  차량용품 키워드 감지
# ============================================================

def detect_accessory_keyword(text: str):
    text_low = text.lower()
    
    kw_map = {
        "엔진오일": ["엔진오일", "오일", "오일갈아", "오일 교체", "오일 교환", "오일필터", "오일 필터", "윤활유"],
        "에어필터": ["에어필터", "캐빈필터", "공기필터", "에어컨필터", "공조필터"],
        "브레이크패드": ["브레이크패드", "패드", "브레이크 패드", "끼익", "브레이크 소리", "덜덜"],
        "브레이크액": ["브레이크액", "브레이크 오일", "dot3", "dot4"],
        "냉각수": ["냉각수", "부동액", "쿨런트", "라디에이터", "과열"],
        "배터리": ["배터리", "방전", "축전지", "시동 안걸림"],
        "타이어": ["타이어", "스노우타이어", "사계절 타이어", "트레드", "공기압", "펑크", "휠"],
        "와이퍼": ["와이퍼", "와이퍼 고무", "유리 닦는"],
        "점화플러그": ["점화플러그", "스파크플러그", "시동불량"],
        "연료첨가제": ["첨가제", "불스원샷", "인젝터 클리너"],
        "OBD": ["obd", "스캐너", "코드리더기"],
        "전조등": ["전조등", "라이트", "램프", "hid", "led"],
        "실내등": ["실내등", "룸램프"],
        "블랙박스": ["블랙박스", "블박", "대시캠", "대쉬캠"],
        "퓨즈": ["퓨즈", "전기 안들어와", "전기 문제"],
        "세차용품": ["세차", "왁스", "광택", "폼건", "카샴푸"],
        "방향제": ["방향제", "탈취", "차 냄새"],
        "충전기": ["충전기", "시거잭", "usb 충전"],
        "체인": ["체인", "스노우체인"]
    }

    for key, words in kw_map.items():
        for w in words:
            if w in text or w.lower() in text_low:
                return key
    return None


# def detect_accessory_keyword(text: str):
#     kw_map = {
#         "타이어": ["타이어", "스노우타이어", "공기압"],
#         "엔진오일": ["엔진오일"],
#         "와이퍼": ["와이퍼"],
#         "배터리": ["배터리"],
#         "블랙박스": ["블랙박스"],
#         "네비게이션": ["네비"],
#         "에어필터": ["에어필터", "캐빈필터"],
#         "체인": ["체인"],
#         "세차용품": ["세차"],
#         "방향제": ["방향제"],
#         "충전기": ["충전기"],
#     }
#     lower = text.lower()
#     for k, arr in kw_map.items():
#         for a in arr:
#             if a in text or a.lower() in lower:
#                 return k
#     return None

# ============================================================
#  쇼핑 의도 감지 (recommend intent)
# ============================================================
def is_recommend_intent(text: str):
    recommend_words = [
        "추천", "추천해줘", "추천해 줘",
        "사야", "사야돼", "사야 돼",
        "사야할까", "사야 할까",
        "사고싶", "사고 싶", 
        "살까", "구매", "뭐 사",
        "골라줘", "고르"
    ]
    for w in recommend_words:
        if w in text:
            return True
    return False

def build_naver_shopping_link(keyword, car):
    q = f"{car} {keyword}" if car else keyword
    return f"https://search.shopping.naver.com/search/all?query={quote(q)}"


# ============================================================
#  FastAPI 설정
# ============================================================
app = FastAPI(title="Capstone Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)


# ============================================================
#  자연어 → 알람 시간 파싱
# ============================================================
def parse_alarm_time(text: str):
    raw = text
    no_space = raw.replace(" ", "")
    now = datetime.now()   # ✅ timezone 없이 로컬 시간

    # --------------------------------------
    # 상대시간
    # --------------------------------------
    m = re.search(r"(\d+)\s*분\s*뒤", raw)
    if m:
        return now + timedelta(minutes=int(m.group(1)))

    m = re.search(r"(\d+)\s*시간\s*뒤", raw)
    if m:
        return now + timedelta(hours=int(m.group(1)))

    # --------------------------------------
    # 절대시간 (오전/오후 포함)
    # --------------------------------------
    m = re.search(r"(오전|오후)\s*(\d+)\s*시\s*(\d*)\s*분?", raw)
    if m:
        ampm = m.group(1)
        hour = int(m.group(2))
        minute = int(m.group(3)) if m.group(3) else 0

        if ampm == "오후" and hour != 12:
            hour += 12
        if ampm == "오전" and hour == 12:
            hour = 0

        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target

    # --------------------------------------
    # 절대시간 (오전/오후 없음 → 24시간 기준)
    # --------------------------------------
    m = re.search(r"(\d+)\s*시\s*(\d*)\s*분?", raw)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0

        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target

    return None


# ============================================================
#  DB 연결
# ============================================================
import psycopg2
import psycopg2.extras

DB_URL = os.getenv("DATABASE_URL")

def db():
    return psycopg2.connect(DB_URL)


# ============================================================
#  AskReq 모델
# ============================================================
class AskReq(BaseModel):
    question: str
    carModel: Optional[str] = None


# ============================================================
#  ask_text — 알람 + RAG + 답변 + TTS
# ============================================================
@app.post("/api/ask")
def ask_text(req: AskReq):

    question_raw = req.question
    question = question_raw.strip()
    no_space = question.replace(" ", "")
    car = req.carModel or "아반떼"

    # -----------------------------------------
    # 1) 알람 문장인지 검사 (음성 문제 해결)
    # -----------------------------------------
    if ("알람" in question) or ("알람" in no_space):

        alarm_at = parse_alarm_time(question)
        if alarm_at:
            try:
                conn = db()
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO alarms(session_id, message, scheduled_at)
                    VALUES (%s, %s, %s)
                    """,
                    ("demo-session", question_raw, alarm_at)
                )
                conn.commit()
                print("[ALARM SAVED]", alarm_at)
            except Exception as e:
                print("[ALARM ERROR]", e)

            local_t = alarm_at.astimezone().strftime("%H시 %M분")
            ans = f"{local_t}에 알람을 설정했습니다."

            # TTS 생성
            audio_b64 = None
            try:
                tts = gTTS(text=ans, lang="ko")
                buf = BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)
                audio_b64 = base64.b64encode(buf.read()).decode()
            except:
                pass

            return {"answer": ans, "carModel": car, "audio": audio_b64}

    # -----------------------------------------
    # 2) 일반 질문 처리 (RAG → ask)
    # -----------------------------------------
    ans = None

    if rag_mod and hasattr(rag_mod, "ask_with_db_context"):
        ans = _call_and_capture(rag_mod.ask_with_db_context, question, car)

    if not ans and ask_mod and hasattr(ask_mod, "ask_question"):
        ans = _call_and_capture(ask_mod.ask_question, question)

    if not ans:
        ans = f"(임시응답) 질문을 받았습니다: {question}"

    # -----------------------------------------
    # 3) 차량용품 키워드
    # -----------------------------------------
    kw = detect_accessory_keyword(question)
    if kw and is_recommend_intent(question):
        link = build_naver_shopping_link(kw, car)
        ans = f"🛒 {kw} 추천 링크입니다:\n{link}"

    # 4) TTS 정제
    tts_text = ans

    # 쇼핑 아이콘은 제거
    tts_text = tts_text.replace("🛒", "")

    # URL 제거 (링크는 읽지 않도록)
    tts_text = re.sub(r"https?://\S+", "", tts_text)

    # 이모지/기타 특수문자 제거
    tts_text = re.sub(r"[^\w\s가-힣.,!?]", "", tts_text).strip()

    audio_b64 = None
    try:
        tts = gTTS(text=tts_text, lang="ko")
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode()
    except:
        pass

    return {"answer": ans, "carModel": car, "audio": audio_b64}


# ============================================================
#  음성 → STT → ask_text
# ============================================================
@app.post("/api/voice")
async def voice(file: UploadFile = File(...), carModel: Optional[str] = None):

    raw = await file.read()
    text = stt_from_bytes(raw, file.content_type)

    if not text:
        text = "(음성인식 실패)"

    print("[STT RESULT]:", text)

    data = ask_text(AskReq(question=text, carModel=carModel))

    return {
        "text": text,
        "answer": data["answer"],
        "carModel": data["carModel"],
        "audio": data["audio"]
    }


# ============================================================
#  알람 관련 API
# ============================================================
class AlarmReq(BaseModel):
    session_id: str
    message: str
    scheduled_at: str


@app.post("/api/alarm/create")
def create_alarm(req: AlarmReq):
    conn = db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO alarms(session_id, message, scheduled_at)
        VALUES (%s, %s, %s)
        RETURNING id
        """,
        (req.session_id, req.message, req.scheduled_at)
    )
    alarm_id = cur.fetchone()[0]
    conn.commit()
    return {"ok": True, "id": alarm_id}


@app.get("/api/alarms")
def list_alarms(session_id: str):
    conn = db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "SELECT * FROM alarms WHERE session_id=%s ORDER BY scheduled_at ASC",
        (session_id,)
    )
    return cur.fetchall()


@app.delete("/api/alarm/{aid}")
def delete_alarm(aid: int):
    conn = db()
    cur = conn.cursor()
    cur.execute("DELETE FROM alarms WHERE id=%s", (aid,))
    conn.commit()
    return {"ok": True}


@app.get("/api/alarm/pending")
def pending_alarm(session_id: str):
    now = datetime.now()

    conn = db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT * FROM alarms
        WHERE session_id=%s
          AND fired=false
          AND scheduled_at <= %s
        ORDER BY scheduled_at ASC
        LIMIT 1
        """,
        (session_id, now)
    )
    row = cur.fetchone()

    if not row:
        return {"alarm": None}

    # 🔻 울린 알람은 바로 삭제 (또는 필요하면 fired=true로만 업데이트)
    cur.execute("DELETE FROM alarms WHERE id=%s", (row["id"],))
    conn.commit()

    return {"alarm": row}
