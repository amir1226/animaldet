import { httpClient } from '../http/client'

export interface BoundingBox {
  x: number
  y: number
  w: number
  h: number
  confidence: number
}

export interface Detection {
  bbox: BoundingBox
  class_id: number
  class_name: string
}

export interface Metadata {
  latency_ms: number
  input_shape: [number, number]
}

export interface InferenceResponse {
  data: {
    detections: Detection[]
    metadata: Metadata
  }
}

export interface InferenceOptions {
  confidenceThreshold?: number
  model?: 'nano' | 'small'
}

export class InferenceService {
  async runInference(imageBlob: Blob, options?: InferenceOptions): Promise<InferenceResponse> {
    const params = new URLSearchParams()
    if (options?.confidenceThreshold !== undefined) {
      params.append('confidence_threshold', options.confidenceThreshold.toString())
    }
    if (options?.model) {
      params.append('model', options.model)
    }

    const url = params.toString() ? `/inference?${params.toString()}` : '/inference'

    return await httpClient.post<InferenceResponse>(url, imageBlob, {
      headers: {
        'Content-Type': 'application/octet-stream',
      },
    })
  }

  async runInferenceFromDataURL(dataURL: string, options?: InferenceOptions): Promise<InferenceResponse> {
    const blob = await fetch(dataURL).then(r => r.blob())
    return this.runInference(blob, options)
  }
}

export const inferenceService = new InferenceService()
