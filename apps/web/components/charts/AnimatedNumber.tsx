"use client";

import { useEffect, useRef, useState } from "react";

interface AnimatedNumberProps {
  value: number;
  format: (v: number) => string;
  duration?: number;
  className?: string;
}

/**
 * Robinhood-leaning soft tween — eases the displayed number to its target
 * over `duration`ms. No flash backgrounds; just calm motion.
 */
export function AnimatedNumber({
  value,
  format,
  duration = 600,
  className,
}: AnimatedNumberProps) {
  const [v, setV] = useState(value);
  const fromRef = useRef(value);

  useEffect(() => {
    const from = fromRef.current;
    const start = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setV(from + (value - from) * eased);
      if (t < 1) raf = requestAnimationFrame(tick);
      else fromRef.current = value;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, duration]);

  return <span className={className}>{format(v)}</span>;
}
