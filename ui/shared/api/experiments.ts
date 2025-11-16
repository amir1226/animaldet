export interface ExperimentDetection {
  image: string
  x: number
  y: number
  x_max: number
  y_max: number
  label: number
  score: number
}

export interface ExperimentData {
  url: string
  name: string
  detections: ExperimentDetection[]
  images: string[]
}

export class ExperimentsService {
  async loadExperimentFromURL(url: string, name: string): Promise<ExperimentData> {
    try {
      const response = await fetch(url)
      if (!response.ok) {
        throw new Error(`Failed to fetch CSV: ${response.statusText}`)
      }

      const text = await response.text()
      const detections = this.parseCSV(text)
      const images = this.extractUniqueImages(detections)

      return {
        url,
        name,
        detections,
        images
      }
    } catch (error) {
      console.error('Failed to load experiment:', error)
      throw error
    }
  }

  private parseCSV(csvText: string): ExperimentDetection[] {
    const lines = csvText.trim().split('\n')
    const detections: ExperimentDetection[] = []

    // Skip header row
    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim()
      if (!line) continue

      const parts = line.split(',')

      // Support two formats:
      // 1. Image,x,y,x_max,y_max,label,score (7 columns)
      // 2. Image,x1,y1,x2,y2,Label (6 columns, no score)
      if (parts.length === 6) {
        // Format: Image,x1,y1,x2,y2,Label
        detections.push({
          image: parts[0].trim(),
          x: parseFloat(parts[1]),
          y: parseFloat(parts[2]),
          x_max: parseFloat(parts[3]),
          y_max: parseFloat(parts[4]),
          label: parseInt(parts[5]),
          score: 1.0 // Default score for annotations without confidence
        })
      } else if (parts.length >= 7) {
        // Format: Image,x,y,x_max,y_max,label,score
        detections.push({
          image: parts[0].trim(),
          x: parseFloat(parts[1]),
          y: parseFloat(parts[2]),
          x_max: parseFloat(parts[3]),
          y_max: parseFloat(parts[4]),
          label: parseInt(parts[5]),
          score: parseFloat(parts[6])
        })
      }
    }

    console.log(`Parsed ${detections.length} detections`)
    if (detections.length > 0) {
      console.log('Sample detection:', detections[0])
    }

    return detections
  }

  private extractUniqueImages(detections: ExperimentDetection[]): string[] {
    const imageSet = new Set<string>()
    detections.forEach(det => imageSet.add(det.image))
    return Array.from(imageSet).sort()
  }

  getDetectionsForImage(detections: ExperimentDetection[], imageName: string): ExperimentDetection[] {
    const filtered = detections.filter(det => det.image === imageName)
    console.log(`Getting detections for image: "${imageName}"`)
    console.log(`Found ${filtered.length} detections`)
    if (filtered.length === 0 && detections.length > 0) {
      console.log('Available image names:', [...new Set(detections.map(d => d.image))].slice(0, 5))
    }
    return filtered
  }
}

export const experimentsService = new ExperimentsService()
