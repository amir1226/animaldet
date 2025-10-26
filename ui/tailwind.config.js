/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Primary - Vegetation greens
        canopy: {
          light: '#9BC995',
          DEFAULT: '#7A9B76',
          dark: '#4A6741',
        },
        // Secondary - Desert/Earth tones
        terrain: {
          light: '#D4C4A8',
          DEFAULT: '#C9B896',
          dark: '#8B5A2B',
        },
      },
    },
  },
  plugins: [],
}
