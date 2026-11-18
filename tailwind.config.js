/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: { 950: '#041009', 900: '#07180f', 800: '#0b2417', 700: '#113222', 600: '#1a4730', 500: '#27603f' },
        lime: { 300: '#d4f77a', 400: '#bff04a', 500: '#a3e635' },
        pitch: '#0f3b26',
        offense: '#fbbf24',
        defense: '#60a5fa',
        predicted: '#e879f9',
        scenario: '#22d3ee',
      },
      fontFamily: {
        sans: ['"Segoe UI Variable"', '"Segoe UI"', 'Inter', 'system-ui', 'sans-serif'],
        display: ['Bahnschrift', '"Segoe UI Black"', 'Impact', 'sans-serif'],
        mono: ['"Cascadia Mono"', 'Consolas', 'monospace'],
      },
      boxShadow: { glow: '0 0 0 1px rgba(163,230,53,.25), 0 8px 30px -10px rgba(163,230,53,.25)' },
    },
  },
  plugins: [],
}
