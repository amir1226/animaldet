// Sample images: one per class from test dataset
export const LARGE_SAMPLE_IMAGES = [
  { name: 'S_07_05_16_DSC00377.JPG', class: 'Topi' },
  { name: 'S_07_05_16_DSC00162.JPG', class: 'Buffalo' },
  { name: 'S_07_05_16_DSC00094.JPG', class: 'Kob' },
  { name: 'S_07_05_16_DSC00307.JPG', class: 'Warthog' },
  { name: 'S_07_05_16_DSC00604.JPG', class: 'Waterbuck' },
  { name: '01802f75da35434ab373569fffc1fd65a3417aef.JPG', class: 'Elephant' },
] as const

export type LargeSampleImage = typeof LARGE_SAMPLE_IMAGES[number]
