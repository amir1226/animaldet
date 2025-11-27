// Sample images larger than 3MB for testing
export const LARGE_SAMPLE_IMAGES = [
  '102d2b93e0bad39c8c041242787eb9eb613848ec.JPG',
  '112351277475e7c55f29d5a3c8c5a349216b514a.JPG',
  '13524d73233a9cc6f30cd1ab853064150e074eaf.JPG',
  '136761239823dcfefbbfda0a570aee77c3608dbf.JPG',
  '14155f30121958a811385dd40c96f8e9294da086.JPG',
] as const

export type LargeSampleImage = typeof LARGE_SAMPLE_IMAGES[number]
