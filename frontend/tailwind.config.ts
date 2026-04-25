import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        card: "hsl(var(--card))",
        "card-foreground": "hsl(var(--card-foreground))",
        primary: "hsl(var(--primary))",
        "primary-foreground": "hsl(var(--primary-foreground))",
        border: "hsl(var(--border))",
        muted: "hsl(var(--muted))",
        ring: "hsl(var(--ring))",
      },
      boxShadow: {
        glow: "0 1px 2px rgba(0,0,0,0.3), 0 8px 32px rgba(0,0,0,0.25)",
        "card-hover": "0 8px 40px rgba(0,0,0,0.4), 0 0 0 1px rgba(148,163,184,0.06)",
      },
    },
  },
  plugins: [],
};

export default config;
