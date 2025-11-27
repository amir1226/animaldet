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
        // Presentation color palette
        forest: {
          DEFAULT: '#1a4d3d',
          dark: '#0f332a',
        },
        lavender: {
          light: '#e8eef9',
          DEFAULT: '#c5d5f0',
        },
        sage: {
          light: '#e8f0d4',
          DEFAULT: '#d4e8b4',
        },
        sand: {
          light: '#e8dcc8',
          DEFAULT: '#d4c8b4',
        },
      },
    },
  },
  plugins: [],
}
