// CO2 구간/색상 유틸 (Figma 팔레트)
export const CO2_COLORS = {
  good:   "#5E9F5C", // 좋음
  warn:   "#C9B458", // 보통
  mid:    "#D9822B", // 나쁨
  danger: "#C14949", // 매우 나쁨
};

// 미세먼지 센서의 값을 넣어주되 이산화탄소 함수 이름은 그대로 유지
// 인자로 들어오는 값은 실제로는 "PM2.5(㎍/m³)" 값입니다.
export function co2Zone(ppm) {
  const pm25 = Number(ppm);

  // 값이 없거나 숫자가 아닐 때 → 일단 "보통" 상태로 처리
  if (!isFinite(pm25)) {
    return { key: "warn", label: "보통", color: CO2_COLORS.warn };
  }

  // ▼ 미세먼지(PM2.5) 기준 (환경부 일반 기준)
  //  0 ~  15 : 좋음
  // 16 ~  35 : 보통
  // 36 ~  75 : 나쁨
  // 76 이상  : 매우 나쁨

  if (pm25 <= 7) {
    return { key: "good", label: "좋음", color: CO2_COLORS.good };
  }
  if (pm25 <= 10) {
    return { key: "warn", label: "보통", color: CO2_COLORS.warn };
  }
  if (pm25 <= 15) {
    return { key: "mid", label: "나쁨", color: CO2_COLORS.mid };
  }
  return          { key: "danger", label: "매우 나쁨", color: CO2_COLORS.danger };
}
