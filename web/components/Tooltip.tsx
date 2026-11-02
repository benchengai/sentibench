"use client";

import { useCallback, useState, type ReactNode } from "react";

type Tip = { x: number; y: number; content: ReactNode } | null;

// One floating tooltip per chart. Marks call show() on pointer move and keyboard focus,
// so hover and focus give the same readout; every value is also in the table view.
export function useTooltip() {
  const [tip, setTip] = useState<Tip>(null);
  const show = useCallback((e: { clientX?: number; clientY?: number; currentTarget: Element }, content: ReactNode) => {
    let x = e.clientX;
    let y = e.clientY;
    if (x === undefined || y === undefined) {
      const r = e.currentTarget.getBoundingClientRect();
      x = r.left + r.width / 2;
      y = r.top;
    }
    setTip({ x, y, content });
  }, []);
  const hide = useCallback(() => setTip(null), []);
  return { tip, show, hide };
}

export function Tooltip({ tip }: { tip: Tip }) {
  if (!tip) return null;
  const left = Math.min(tip.x + 14, (typeof window !== "undefined" ? window.innerWidth : 1200) - 290);
  return (
    <div className="tooltip" role="status" style={{ left, top: tip.y + 14 }}>
      {tip.content}
    </div>
  );
}
