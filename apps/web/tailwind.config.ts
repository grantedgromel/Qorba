import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  darkMode: ["class", "[data-theme='dark']"],
  theme: {
    container: { center: true, padding: "1rem" },
    extend: {
      colors: {
        // Surfaces (numbered: 0 = page, 1 = card, 2 = hover, 3 = active)
        bg: {
          DEFAULT: "var(--bg-0)",
          0: "var(--bg-0)",
          1: "var(--bg-1)",
          2: "var(--bg-2)",
          3: "var(--bg-3)",
          inset: "var(--bg-inset)",
        },
        ink: {
          DEFAULT: "var(--ink-0)",
          0: "var(--ink-0)",
          1: "var(--ink-1)",
          2: "var(--ink-2)",
          3: "var(--ink-3)",
        },
        line: {
          DEFAULT: "var(--line)",
          soft: "var(--line-soft)",
        },
        accent: {
          DEFAULT: "var(--accent)",
          soft: "var(--accent-soft)",
          line: "var(--accent-line)",
          ink: "var(--accent-ink)",
        },
        pos: "var(--pos)",
        neg: "var(--neg)",
        warn: "var(--warn)",
        info: "var(--info)",
      },
      borderColor: {
        DEFAULT: "var(--line)",
        soft: "var(--line-soft)",
      },
      borderRadius: {
        sm: "var(--r-1)",
        md: "var(--r-2)",
        lg: "var(--r-3)",
        xl: "var(--r-4)",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
        serif: ["var(--font-serif)", "Georgia", "serif"],
      },
      fontSize: {
        // Match the design's smaller base step
        xxs: ["10px", "1.4"],
      },
      transitionTimingFunction: {
        qorba: "cubic-bezier(0.2, 0.7, 0.2, 1)",
      },
    },
  },
  plugins: [],
};

export default config;
