import { NavLink, useLocation } from "react-router-dom";
import { Home, MessageSquare, Cloud, ShoppingCart, ArrowLeft } from "lucide-react";

const Item = ({ to, children, label }) => (
  <NavLink
    to={to}
    className={({ isActive }) =>
      `icon-btn text-sub ${isActive ? "text-text bg-white/10" : "hover:text-text"}`
    }
    title={label}
  >
    {children}
  </NavLink>
);

export default function Sidebar() {
  const { pathname } = useLocation();

  return (
    <aside className="h-full flex">
      {/* Figma 왼쪽 얇은 스트립 */}
      <div style={{ width: 24, backgroundColor: "#1C1C1C" }} />
      {/* 패널 */}
      <div className="w-20 bg-panel border-r border-white/10 flex flex-col items-center py-5 gap-5">
        <Item to="/" label="홈"><Home size={28} /></Item>
        <Item to="/chat" label="채팅"><MessageSquare size={26} /></Item>
        <Item to="/co2" label="이산화탄소"><Cloud size={28} /></Item>
        <div className="flex-1" />
        <Item to="/" label="뒤로"><ArrowLeft size={26} /></Item>

        {/* 현재 위치(선택 사항) */}
        <div className="mt-2 text-[10px] text-sub/80">
          {pathname.replace("/", "") || "home"}
        </div>
      </div>
    </aside>
  );
}
