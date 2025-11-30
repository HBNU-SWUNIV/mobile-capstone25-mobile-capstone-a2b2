// src/pages/Home.jsx
import { useEffect, useState, useRef } from "react";
import { Mic, Search, Cloud } from "lucide-react";
import { co2Zone } from "../utils/co2";

// 🔗 API 주소
// - 라즈베리파이에서 UI까지 같이 돌리면: "http://localhost:5000"
// - 노트북에서 UI를 돌리면: "http://라즈베리파이IP:5000"
//   예: const API_BASE = "http://192.168.46.116:5000";
const API_BASE = "http://192.168.46.116:5000"; // 👉 네 환경에 맞게 바꿔도 됨
const DUST_URL = `${API_BASE}/api/dust`;

export default function Home() {
  const [now, setNow] = useState(new Date());
  const [listening, setListening] = useState(false);

  // 실제로는 PM2.5 값이지만, UI에서는 "CO₂ 농도"처럼 보여줌
  const [co2Value, setCo2Value] = useState(null);
  const [dustError, setDustError] = useState("");

  // 🔊 알림 음성 & 이전 zone 상태 기억용 ref
  const alertAudioRef = useRef(null);
  const prevZoneKeyRef = useRef(null);

  // 🔔 경고 배너 표시 여부 & 타이머
  const [showAlertBanner, setShowAlertBanner] = useState(false);
  const bannerTimeoutRef = useRef(null);

  // 시계
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  // 🔊 알림 음성 미리 로드
  useEffect(() => {
    // public/sounds/alert-bad-air.mp3 를 로드
    alertAudioRef.current = new Audio("/sounds/alert-bad-air.mp3");
  }, []);

  // 센서 값 주기적으로 가져오기
  useEffect(() => {
    let isMounted = true;

    async function fetchDust() {
      try {
        const res = await fetch(DUST_URL);
        const data = await res.json();

        if (!isMounted) return;

        if (!data.ok || data.pm25 == null) {
          setDustError("센서에서 값을 가져오지 못했습니다.");
          setCo2Value(null);
        } else {
          setDustError("");
          setCo2Value(data.pm25); // PM2.5 값
        }
      } catch (err) {
        if (!isMounted) return;
        setDustError("센서 서버에 연결할 수 없습니다.");
        setCo2Value(null);
      }
    }

    fetchDust();                             // 처음 1번
    const id = setInterval(fetchDust, 5000); // 5초마다 자동 호출

    return () => {
      isMounted = false;
      clearInterval(id);
    };
  }, []);

  const hh = String(now.getHours()).padStart(2, "0");
  const mm = String(now.getMinutes()).padStart(2, "0");
  const ss = String(now.getSeconds()).padStart(2, "0");

  const zone = co2Zone(co2Value);

  // 공기질 상태가 "나쁨/매우나쁨" 구간으로 처음 진입할 때만
  // ① 알림 음성 재생 + ② 경고 배너를 5초 동안 표시
  useEffect(() => {
    if (!zone) return;
    if (co2Value == null || dustError) return;

    const currentKey = zone.key;            // "good" | "warn" | "mid" | "danger"
    const prevKey = prevZoneKeyRef.current; // 이전 상태 기억

    // 다음 비교를 위해 현재 상태 저장
    prevZoneKeyRef.current = currentKey;

    const isBadNow = currentKey === "mid" || currentKey === "danger";
    const wasBadBefore = prevKey === "mid" || prevKey === "danger";

    // 지금 나쁨이 아니면(좋음/보통이면) 아무 것도 안 함
    if (!isBadNow) return;

    // 이전에도 이미 나쁨 상태였다면(계속 나쁨이면) 새로 재생/표시 안 함
    if (wasBadBefore) return;

    // 👉 여기까지 왔다는 뜻 = good/warn → mid/danger 로 "처음 넘어온 순간"

    // 1) 알림 음성 재생
    const audio = alertAudioRef.current;
    if (audio) {
      audio.currentTime = 0;
      audio
        .play()
        .catch((err) => {
          console.log("알림 음성 재생 오류:", err);
        });
    }

    // 2) 경고 배너 5초 동안 표시
    setShowAlertBanner(true);

    // 기존 타이머 있으면 정리
    if (bannerTimeoutRef.current) {
      clearTimeout(bannerTimeoutRef.current);
    }

    // 5초 후 자동으로 배너 숨기기
    bannerTimeoutRef.current = setTimeout(() => {
      setShowAlertBanner(false);
      bannerTimeoutRef.current = null;
    }, 5000);
  }, [zone.key, co2Value, dustError]);

  return (
    <div className="h-full relative flex items-center justify-center bg-bg text-text">
      <div className="w-full max-w-[980px] px-6 flex flex-col items-center text-center">

        {/* 시계 */}
        <div
          className="font-extrabold leading-none tracking-widest"
          style={{ fontSize: "clamp(48px, 7vw, 80px)" }}
        >
          {hh} : {mm} : {ss}
        </div>

        {/* CO₂ 정보 + 구름 아이콘 */}
        <div className="mt-8 grid grid-cols-[auto_auto] items-center gap-x-6">
          <div className="text-left">
            <div className="text-[30px] font-extrabold leading-tight">
              현재 CO₂ 농도는
            </div>

            {dustError ? (
              <div className="text-[18px] mt-1 text-red-400">
                {dustError}
              </div>
            ) : co2Value == null ? (
              <div className="text-[24px] font-extrabold mt-1 opacity-80">
                측정 중...
              </div>
            ) : (
              <div className="text-[30px] font-extrabold leading-tight">
                {co2Value.toFixed(1)} ppm
              </div>
            )}
          </div>

          {/* 구름 아이콘 (색 = zone.color) */}
          <Cloud
            size={90}
            strokeWidth={2.5}
            style={{ color: zone.color, fill: zone.color }}
          />
        </div>
      </div>

      {/* 좌하단 마이크 버튼 */}
      <button
        onClick={() => setListening(v => !v)}
        className={`absolute bottom-10 left-10 h-16 w-16 rounded-full flex items-center justify-center shadow-lg transition z-20 ${
          listening ? "bg-good/30" : "bg-panel"
        }`}
        aria-label="Microphone"
      >
        <Mic className={listening ? "text-good" : "text-text"} size={28} />
      </button>

      {/* 음성 인식 바 */}
      {listening && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 px-6 py-3 rounded-full bg-panel border border-white/10 text-text shadow-lg flex items-center gap-3 z-20">
          <Mic className="opacity-90" size={18} />
          <span className="opacity-90">궁금한 내용을 질문해주세요!</span>
          <Search size={18} className="opacity-80" />
        </div>
      )}

      {/* 🔔 공기질 경고 배너 (5초간 표시) */}
      {showAlertBanner && (
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-30 flex justify-center">
          <div className="max-w-md w-full rounded-xl bg-red-500/90 text-white px-4 py-3 shadow-lg text-sm flex items-center gap-3">
            <span className="font-semibold">경고</span>
            <span className="flex-1 text-left">
              현재 공기질이 좋지 않습니다. 창문을 열어 환기해 주세요.
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
