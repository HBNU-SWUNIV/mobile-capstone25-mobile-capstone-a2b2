import { useEffect, useRef, useState } from "react";
import { Send, Mic, Bot, User } from "lucide-react";

const DEFAULT_CAR_MODEL = "아반떼";
const API_BASE = "";

/** 짧은 문장은 알약(Pill) 형태 */
const isShort = (t) => t.length <= 12 && !t.includes("\n");

/** 말풍선 */
function Bubble({ me, children }) {
  return (
    <div className={`w-full flex ${me ? "justify-end" : "justify-start"}`}>
      <div
        className={[
          "max-w-[76%] rounded-2xl leading-relaxed shadow-sm border border-white/10",
          "px-4 py-3 text-[18px]",
          me ? "bg-accent/20 text-text" : "bg-white/10 text-text",
        ].join(" ")}
      >
        {children}
      </div>
    </div>
  );
}

/** 프로필 + 말풍선 */
function Row({ me, text }) {
  const content = isShort(text) ? (
    <div
      className={[
        "px-5 py-2.5 rounded-full text-[18px] font-medium",
        me ? "bg-accent/30 text-text" : "bg-white/10 text-text",
      ].join(" ")}
    >
      {text}
    </div>
  ) : (
    <Bubble me={me}>
      {text.split("\n").map((line, i) => (
        <p
          key={i}
          className="whitespace-pre-wrap break-words"   // ← break-words 추가
          dangerouslySetInnerHTML={{
            __html: line.replace(
              /(https?:\/\/[^\s]+)/g,
              '<a href="$1" target="_blank" class="underline text-blue-400 break-all">$1</a>'
            ),
          }}
        />
      ))}
    </Bubble>
  );

  return (
    <div className={`w-full flex items-center gap-3 ${me ? "justify-end" : ""}`}>
      {!me && (
        <div className="shrink-0 h-8 w-8 rounded-full bg-panel border border-white/10 flex items-center justify-center mt-[2px]">
          <Bot size={16} className="text-sub" />
        </div>
      )}
      {content}
      {me && (
        <div className="shrink-0 h-8 w-8 rounded-full bg-accent/20 flex items-center justify-center mt-[2px]">
          <User size={16} className="text-text" />
        </div>
      )}
    </div>
  );
}

export default function Chat() {
  const [messages, setMessages] = useState([
    { me: false, text: "무엇을 도와드릴까요?\n예: ‘엔진 경고등이 켜졌어요’" },
  ]);
  const [input, setInput] = useState("");
  const [recording, setRecording] = useState(false);
  const [sending, setSending] = useState(false);

  // 🔊 TTS 재생 여부
  const [ttsPlaying, setTtsPlaying] = useState(false);

  // 🔊 TTS 속도 (A버전: 개발자가 직접 변경)
  const [ttsSpeed] = useState(1.5); // 기본 1.5배속

  // 🔔 알람용 세션 ID (임시 고정값, 필요하면 나중에 UUID/localStorage로 교체)
  const SESSION_ID = "demo-session";

  const endRef = useRef(null);
  const ttsAudioRef = useRef(null);

  // 🔊 TTS 중단
  const stopTTS = () => {
    if (ttsAudioRef.current) {
      ttsAudioRef.current.pause();
      ttsAudioRef.current.currentTime = 0;
      ttsAudioRef.current = null;
    }
    setTtsPlaying(false);
  };

  // 🔊 TTS 재생 (속도 적용)
  const playTTS = (audioBase64) => {
    if (!audioBase64) return;

    stopTTS();

    const bytes = Uint8Array.from(atob(audioBase64), (c) => c.charCodeAt(0));
    const blob = new Blob([bytes], { type: "audio/mpeg" });
    const url = URL.createObjectURL(blob);

    const audio = new Audio(url);
    audio.playbackRate = ttsSpeed; // 속도 적용

    ttsAudioRef.current = audio;
    setTtsPlaying(true);

    audio.onended = () => setTtsPlaying(false);
    audio.onerror = () => setTtsPlaying(false);

    audio.play().catch(() => setTtsPlaying(false));
  };

  // 텍스트 전송
  const send = async () => {
    const text = input.trim();
    if (!text || sending || recording) return;

    stopTTS();

    setMessages((prev) => [
      ...prev,
      { me: true, text },
      { me: false, text: "생각 중…" },
    ]);

    setInput("");
    setSending(true);

    try {
      const res = await fetch(`${API_BASE}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, carModel: DEFAULT_CAR_MODEL }),
      });

      const data = await res.json();

      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = { me: false, text: data.answer };
        return next;
      });

      playTTS(data.audio);
    } catch {
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = { me: false, text: "서버 오류" };
        return next;
      });
    }

    setSending(false);
  };

  // 녹음 관련
  const mediaRecRef = useRef(null);
  const chunksRef = useRef([]);

  const startRecording = async () => {
    stopTTS();

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      mediaRecRef.current = mr;

      chunksRef.current = [];
      setRecording(true);

      mr.ondataavailable = (e) => chunksRef.current.push(e.data);

      mr.onstop = async () => {
        setRecording(false);

        const blob = new Blob(chunksRef.current, { type: "audio/webm" });

        setMessages((prev) => [
          ...prev,
          { me: true, text: "(음성 메시지)" },
          { me: false, text: "음성 인식 중…" },
        ]);

        const fd = new FormData();
        fd.append("file", blob, "voice.webm");

        try {
          const res = await fetch(`${API_BASE}/api/voice`, {
            method: "POST",
            body: fd,
          });

          const data = await res.json();

          setMessages((prev) => {
            const next = [...prev];
            next[next.length - 1] = {
              me: false,
              text: `📝 인식: ${data.text}\n\n${data.answer}`,
            };
            return next;
          });

          playTTS(data.audio);
        } catch {
          setMessages((prev) => [
            ...prev,
            { me: false, text: "음성 전송 실패" },
          ]);
        }
      };

      mr.start();
    } catch {
      setMessages((prev) => [
        ...prev,
        { me: false, text: "마이크 권한을 허용해주세요." },
      ]);
    }
  };

  const stopRecording = () => {
    const mr = mediaRecRef.current;
    if (mr && mr.state !== "inactive") {
      mr.stop();
      mr.stream.getTracks().forEach((t) => t.stop());
    }
    setRecording(false);
  };

  // 🔔 알람 폴링 (5초마다 백엔드에 알림 체크)
  useEffect(() => {
    const timer = setInterval(async () => {
      try {
        const res = await fetch(
          `${API_BASE}/api/alarm/pending?session_id=${encodeURIComponent(
            SESSION_ID
          )}`
        );
        const data = await res.json();

        if (data.alarm) {
          // 1) 채팅창에 알람 메시지 추가
          setMessages((prev) => [
            ...prev,
            { me: false, text: `⏰ 알람: ${data.alarm.message}` },
          ]);

          // 2) 알람 소리 재생 (public/sounds/alarm1.mp3 에 파일 두기)
          const audio = new Audio("/sounds/alarm.mp3");
          audio.play().catch(() => {});
        }
      } catch (e) {
        console.error("알람 체크 실패:", e);
      }
    }, 5000); // 5초마다 서버에 알람 확인

    return () => clearInterval(timer);
  }, []);

  // 🎤 버튼 분기
  const handleMicClick = () => {
    if (ttsPlaying) {
      stopTTS();
      return;
    }

    if (recording) {
      stopRecording();
      return;
    }

    startRecording();
  };

  return (
    <div className="h-full flex flex-col">
      <header className="px-6 py-3 border-b border-white/10 flex items-center gap-3">
        <div className="h-8 w-8 rounded-xl bg-panel border border-white/10 flex items-center justify-center">
          <Bot size={16} className="text-sub" />
        </div>
        <div className="font-semibold">차량용 AI 어시스턴트</div>
      </header>

      <section className="flex-1 overflow-y-auto px-6 py-5 space-y-6">
        {messages.map((m, i) => (
          <Row key={i} me={m.me} text={m.text} />
        ))}
        <div ref={endRef} />
      </section>

      <footer className="px-6 pb-6">
        <div className="bg-panel border border-white/10 rounded-full h-14 px-3 flex items-center gap-1">
          {/* 마이크 버튼 */}
          <button
            onClick={handleMicClick}
            className={`h-10 w-10 rounded-full flex items-center justify-center ${
              recording ? "bg-good/30" : "hover:bg-white/10"
            }`}
          >
            {ttsPlaying ? (
              <span className="text-sub text-xl">🔇</span>
            ) : (
              <Mic size={18} className={recording ? "text-good" : "text-sub"} />
            )}
          </button>

          <input
            value={input}
            disabled={sending || recording}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder={
              recording
                ? "녹음 중..."
                : sending
                ? "응답 대기 중..."
                : "메시지를 입력하세요"
            }
            className="flex-1 bg-transparent outline-none text-[16px] text-text placeholder:text-sub/70 px-2"
          />

          <button
            onClick={send}
            disabled={sending || recording}
            className={`h-10 w-10 rounded-full ${
              sending ? "bg-accent/50" : "bg-accent"
            } text-black flex items-center justify-center`}
          >
            <Send size={16} />
          </button>
        </div>
      </footer>
    </div>
  );
}
