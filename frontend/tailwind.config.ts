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
        glow: "0 0 0 1px rgba(0,0,0,0.08), 0 8px 30px rgba(17,34,17,0.18)",
      },
    },
  },
  plugins: [],
};

export default config;
