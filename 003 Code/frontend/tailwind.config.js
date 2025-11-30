/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Figma 팔레트
        bg:     "#121212",
        panel:  "#2E2E2E",
        text:   "#F5F5F5",
        sub:    "#6B7280",
        mute:   "#B0B0B0",

        // CO₂ 상태 색
        good:   "#5E9F5C",
        warn:   "#C9B458",
        mid:    "#D9822B",
        danger: "#C14949",
        accent: "#5AC8FA", // 포인트(블루)
      },
      borderRadius: {
        xl2: "1rem",
      },
    },
  },
  plugins: [],
};
