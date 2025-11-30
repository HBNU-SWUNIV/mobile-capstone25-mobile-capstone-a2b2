// src/pages/Co2.jsx
import React, { useRef, useState, useEffect } from "react";
import {
  ResponsiveContainer, LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip,
  BarChart, Bar,
} from "recharts";
import { motion } from "framer-motion";
import { Cloud } from "lucide-react";
import { co2Zone } from "../utils/co2";

// 🔗 라즈베리파이 Flask 서버 주소
const API_BASE = "http://192.168.46.116:5000"; // 또는 "http://localhost:5000"
const DUST_URL    = `${API_BASE}/api/dust`;
const RECENT_URL  = `${API_BASE}/api/dust/recent?minutes=120`; // 최근 2시간
const DAILY_URL   = `${API_BASE}/api/dust/daily?days=7`;       // 최근 7일
const MONTHLY_URL = `${API_BASE}/api/dust/monthly?months=6`;   // 최근 6개월;

/* =========================
   스와이프 훅 (모바일/데스크탑)
========================= */
function useSwipe({ onLeft, onRight, threshold = 40 }) {
  const startX = useRef(null);
  const isDown = useRef(false);

  const start = (x) => { startX.current = x; isDown.current = true; };
  const move = (x) => {
    if (!isDown.current || startX.current == null) return;
    const dx = x - startX.current;
    if (Math.abs(dx) > threshold) {
      isDown.current = false;
      dx < 0 ? onLeft?.() : onRight?.();
    }
  };
  const end = () => { startX.current = null; isDown.current = false; };

  return {
    bind: {
      onTouchStart: (e) => start(e.touches[0].clientX),
      onTouchMove: (e) => move(e.touches[0].clientX),
      onTouchEnd: end,
      onMouseDown: (e) => start(e.clientX),
      onMouseMove: (e) => isDown.current && move(e.clientX),
      onMouseUp: end,
      onMouseLeave: end,
    },
  };
}

export default function Co2() {
  const [index, setIndex] = useState(0); // 0: 실시간, 1: 일간, 2: 월간
  const titles = [
    "실시간 이산화탄소 그래프",
    "일간 평균 이산화탄소 그래프",
    "월간 평균 이산화탄소 그래프",
  ];
  const slideCount = titles.length;

  const go = (i) => setIndex(((i % slideCount) + slideCount) % slideCount);
  const { bind } = useSwipe({ onLeft: () => go(index + 1), onRight: () => go(index - 1) });

  // 현재 센서 값 (카드용)
  const [current, setCurrent] = useState(null);
  const [dustError, setDustError] = useState("");

  // 🔊 공기질 알림 음성 & 이전 상태 기억용
  const alertAudioRef = useRef(null);
  const prevZoneKeyRef = useRef(null);

  // 🔔 경고 배너 표시 여부 & 타이머
  const [showAlertBanner, setShowAlertBanner] = useState(false);
  const bannerTimeoutRef = useRef(null);

  // 그래프용 데이터
  const [lineData, setLineData] = useState([]);   // 실시간 (최근 2시간)
  const [dayData, setDayData] = useState([]);     // 일별 평균
  const [monthData, setMonthData] = useState([]); // 월별 평균

  // 현재 값 가져오기 (카드용)
  useEffect(() => {
    let isMounted = true;

    async function fetchDust() {
      try {
        const res = await fetch(DUST_URL);
        const data = await res.json();

        if (!isMounted) return;

        if (!data.ok || data.pm25 == null) {
          setDustError("센서에서 값을 가져오지 못했습니다.");
          setCurrent(null);
        } else {
          setDustError("");
          setCurrent(data.pm25);
        }
      } catch (err) {
        if (!isMounted) return;
        setDustError("센서 서버에 연결할 수 없습니다.");
        setCurrent(null);
      }
    }

    fetchDust();
    const id = setInterval(fetchDust, 5000);

    return () => {
      isMounted = false;
      clearInterval(id);
    };
  }, []);

  // 🔊 알림 음성 미리 로드
  useEffect(() => {
    // Home에서 썼던 것과 동일한 경로
    alertAudioRef.current = new Audio("/sounds/alert-bad-air.mp3");
  }, []);

  // 그래프용 데이터 가져오기
  useEffect(() => {
    let isMounted = true;

    async function fetchRecent() {
      try {
        const res = await fetch(RECENT_URL);
        const data = await res.json();
        if (!isMounted || !data.ok) return;

        // 원본 포인트: [{ timestamp: "YYYY-MM-DD HH:MM:SS", pm25, ... }]
        const points = data.points || [];

        // 1) 5분 단위 버킷으로 묶어서 평균 내기
        const bucketMap = new Map();

        points.forEach((p) => {
          const ts = p.timestamp; // "YYYY-MM-DD HH:MM:SS"
          if (!ts || p.pm25 == null) return;

          const minuteStr = ts.slice(14, 16); // "MM"
          const minute = parseInt(minuteStr, 10);
          if (Number.isNaN(minute)) return;

          // 5분 단위로 내림 (0,5,10,...,55)
          const bucketMinute = Math.floor(minute / 5) * 5;
          const bucketMinuteStr = String(bucketMinute).padStart(2, "0");

          // "YYYY-MM-DD HH:" 까지 잘라서 5분 단위 분 붙이기
          const bucketKey = ts.slice(0, 14) + bucketMinuteStr; // "YYYY-MM-DD HH:MM"

          if (!bucketMap.has(bucketKey)) {
            bucketMap.set(bucketKey, { sum: 0, count: 0 });
          }
          const bucket = bucketMap.get(bucketKey);
          bucket.sum += p.pm25;
          bucket.count += 1;
        });

        // 2) 시간순으로 정렬 + 평균값 배열로 변환
        const keys = Array.from(bucketMap.keys()).sort(); // 문자열 정렬 = 시간순
        let averaged = keys.map((key) => {
          const { sum, count } = bucketMap.get(key);
          const avg = sum / count;
          const timeLabel = key.slice(11, 16); // "HH:MM" 만 X축 라벨로 사용
          return { t: timeLabel, v: avg };
        });

        // 3) 너무 많으면 MAX_POINTS 개수로 줄이기
        const MAX_POINTS = 60;
        if (averaged.length > MAX_POINTS) {
          const step = Math.ceil(averaged.length / MAX_POINTS);
          averaged = averaged.filter((_, idx) => idx % step === 0);
        }

        setLineData(averaged);
      } catch (err) {
        console.log("recent fetch error", err);
      }
    }

    async function fetchDaily() {
      try {
        const res = await fetch(DAILY_URL);
        const data = await res.json();
        if (!isMounted || !data.ok) return;

        let mapped = data.items.map((d) => {
          const label = d.date.slice(5); // "MM-DD"
          return { name: label, v: d.avg_pm25 };
        });

        // 👉 막대는 최대 7개만 보여주기 (최근 7일)
        if (mapped.length > 7) {
          mapped = mapped.slice(-7); // 뒤에서 7개(가장 최신 7일)
        }

        setDayData(mapped);
      } catch (err) {
        console.log("daily fetch error", err);
      }
    }

    async function fetchMonthly() {
      try {
        const res = await fetch(MONTHLY_URL);
        const data = await res.json();
        if (!isMounted || !data.ok) return;

        const mapped = data.items.map((m) => {
          const label = m.month; // "YYYY-MM"
          return { name: label, v: m.avg_pm25 };
        });
        setMonthData(mapped);
      } catch (err) {
        console.log("monthly fetch error", err);
      }
    }

    // 처음 한 번 전부 가져오기
    fetchRecent();
    fetchDaily();
    fetchMonthly();

    // 실시간 그래프는 1분마다 갱신
    const id = setInterval(fetchRecent, 60 * 1000);

    return () => {
      isMounted = false;
      clearInterval(id);
    };
  }, []);

  const zone = co2Zone(current);
  const axisColor = "#6B7280";
  const tickStyle = { fontSize: 12 };

  // 🔊 공기질 상태가 나쁨/매우나쁨으로 처음 들어갈 때
  // ① 알림 음성 재생 + ② 경고 배너를 5초 동안 표시
  useEffect(() => {
    if (!zone) return;
    if (current == null || dustError) return;

    const currentKey = zone.key;            // "good" | "warn" | "mid" | "danger"
    const prevKey = prevZoneKeyRef.current; // 이전 상태 기억

    // 다음 비교를 위해 현재 상태 저장
    prevZoneKeyRef.current = currentKey;

    const isBadNow = currentKey === "mid" || currentKey === "danger";
    const wasBadBefore = prevKey === "mid" || prevKey === "danger";

    // 지금 나쁨/매우나쁨이 아니면 재생 X
    if (!isBadNow) return;

    // 이전에도 이미 나쁨/매우나쁨 상태였다면 또 재생 X
    if (wasBadBefore) return;

    // 👉 good/warn → mid/danger 로 처음 넘어온 순간

    // 1) 알림 음성 재생
    const audio = alertAudioRef.current;
    if (audio) {
      audio.currentTime = 0;
      audio
        .play()
        .catch((err) => {
          console.log("Co2 페이지 알림 음성 재생 오류:", err);
        });
    }

    // 2) 경고 배너 5초 동안 표시
    setShowAlertBanner(true);

    // 기존 타이머 있으면 정리
    if (bannerTimeoutRef.current) {
      clearTimeout(bannerTimeoutRef.current);
    }

    // 5초 후 배너 숨기기
    bannerTimeoutRef.current = setTimeout(() => {
      setShowAlertBanner(false);
      bannerTimeoutRef.current = null;
    }, 5000);
  }, [zone.key, current, dustError]);

  return (
    <div className="h-full flex flex-col gap-6 p-6">
      {/* ===== 그래프 캐러셀 카드 ===== */}
      <section className="bg-panel rounded-2xl p-5 border border-white/5 shadow-lg select-none">
        <motion.h2
          key={index}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
          className="text-white font-semibold text-base mb-2"
        >
          {titles[index]}
        </motion.h2>

        <div className="overflow-hidden rounded-xl" {...bind}>
          <div
            className="flex w-full transition-transform duration-300 ease-out"
            style={{ transform: `translateX(-${index * 100}%)` }}
          >
            {/* Slide 0: 실시간 라인차트 */}
            <div className="w-full shrink-0 h-64 px-1">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={lineData}>
                  <CartesianGrid stroke="#ffffff10" vertical={false} />
                  <XAxis dataKey="t" stroke={axisColor} tick={tickStyle} />
                  <YAxis stroke={axisColor} tick={tickStyle} />
                  <Tooltip />
                  <Line type="monotone" dataKey="v" stroke="#5AC8FA" strokeWidth={2} dot />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Slide 1: 일간 바차트 */}
            <div className="w-full shrink-0 h-64 px-1">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={dayData}>
                  <CartesianGrid stroke="#ffffff10" vertical={false} />
                  <XAxis dataKey="name" stroke={axisColor} tick={tickStyle} />
                  <YAxis stroke={axisColor} tick={tickStyle} />
                  <Tooltip />
                  <Bar dataKey="v" fill="#5AC8FA" radius={[8, 8, 0, 0]} barSize={40} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Slide 2: 월간 바차트 */}
            <div className="w-full shrink-0 h-64 px-1">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={monthData}>
                  <CartesianGrid stroke="#ffffff10" vertical={false} />
                  <XAxis dataKey="name" stroke={axisColor} tick={tickStyle} />
                  <YAxis stroke={axisColor} tick={tickStyle} />
                  <Tooltip />
                  <Bar dataKey="v" fill="#5AC8FA" radius={[8, 8, 0, 0]} barSize={40} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* 도트 인디케이터 */}
        <div className="mt-2 flex items-center justify-center gap-2">
          {Array.from({ length: slideCount }).map((_, i) => (
            <button
              key={i}
              onClick={() => go(i)}
              className={`h-2 w-2.5 rounded-full transition-transform ${
                i === index ? "bg-accent scale-125" : "bg-white/30"
              }`}
              aria-label={`slide-${i + 1}`}
            />
          ))}
        </div>
      </section>

      {/* ===== 현재 농도 카드 ===== */}
      <section className="flex items-center justify-between bg-panel rounded-2xl p-6 border border-white/5 shadow-lg">
        <div>
          <div className="text-white text-base">현재 CO₂ 농도는</div>

          {dustError ? (
            <div className="text-red-400 text-sm mt-1">{dustError}</div>
          ) : current == null ? (
            <div className="text-white text-2xl font-extrabold mt-1 opacity-80">
              측정 중...
            </div>
          ) : (
            <div className="text-white text-4xl font-extrabold mt-1">
              {current.toFixed(1)} ppm
            </div>
          )}

          <span
            className="mt-3 inline-flex items-center gap-2 rounded-full px-4 py-1.5 text-sm font-semibold"
            style={{ backgroundColor: `${zone.color}33`, color: zone.color }}
          >
            {zone.label}
          </span>
        </div>

        {/* 구름 아이콘 */}
        <Cloud
          size={90}
          strokeWidth={2.5}
          style={{ color: zone.color, fill: zone.color }}
          aria-label="CO2 상태"
        />
      </section>

      {/* 🔔 공기질 경고 배너 (5초간 표시) */}
      {showAlertBanner && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-30 flex justify-center">
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
