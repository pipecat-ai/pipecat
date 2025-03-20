/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      keyframes: {
        'message-pop': {
          '0%': {
            opacity: '0',
            transform: 'scale(0.95) translateY(10px)'
          },
          '100%': {
            opacity: '1',
            transform: 'scale(1) translateY(0)'
          }
        },
        'memory-pop': {
          '0%': {
            opacity: '0',
            transform: 'translateY(10px)'
          },
          '100%': {
            opacity: '1',
            transform: 'translateY(0)'
          }
        },
        'loading-bar': {
          '0%': {
            transform: 'translateX(-100%)'
          },
          '100%': {
            transform: 'translateX(100%)'
          }
        },
        'shimmer': {
          '100%': {
            transform: 'translateX(100%)'
          }
        }
      },
      animation: {
        'message-pop': 'message-pop 0.3s ease-out',
        'memory-pop': 'memory-pop 0.3s ease-out forwards',
        'loading-bar': 'loading-bar 2s infinite',
        'shimmer': 'shimmer 1.5s infinite'
      }
    },
  },
  plugins: [],
} 