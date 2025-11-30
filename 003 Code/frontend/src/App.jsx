import { Routes, Route, Navigate } from "react-router-dom";
import Sidebar from "./components/Sidebar.jsx";
import Home from "./pages/Home.jsx";
import Chat from "./pages/Chat.jsx";
import Co2 from "./pages/Co2.jsx";

export default function App() {
  return (
    <div className="h-screen flex bg-bg text-text">
      {/* 좌측 사이드바 */}
      <Sidebar />

      {/* 우측 화면 */}
      <main className="flex-1 relative">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/co2" element={<Co2 />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
