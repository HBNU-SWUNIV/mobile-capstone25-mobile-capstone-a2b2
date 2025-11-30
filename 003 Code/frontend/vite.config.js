import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// .env 의 VITE_API_URL 우선, 없으면 8000 유지
// 예) VITE_API_URL=http://192.168.0.10:8080
const BACKEND_URL = process.env.VITE_API_URL || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,          // 같은 네트워크(패드/휴대폰)에서 접속 허용
    port: 5173,          // 개발 포트
    proxy: {
      // 핵심 API 묶음: 텍스트 질의(/api/ask), 음성 전송(/api/voice) 등
      '/api':     { target: BACKEND_URL, changeOrigin: true, secure: false, ws: true },

      // 필요 시 함께 쓰는 엔드포인트(센서/QA/쇼핑/비서 등)도 동일 백엔드로 전달
      '/sensors': { target: BACKEND_URL, changeOrigin: true, secure: false, ws: true },
      '/qa':      { target: BACKEND_URL, changeOrigin: true, secure: false, ws: true },
      '/shop':    { target: BACKEND_URL, changeOrigin: true, secure: false, ws: true },
      '/assist':  { target: BACKEND_URL, changeOrigin: true, secure: false, ws: true },
    },
  },
})
