import { httpClient } from '../http/client'
import { onnxInferenceService } from '../services/onnx-inference'

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

export type InferenceMode = 'api' | 'onnx'

export class InferenceService {
  private mode: InferenceMode = 'api'

  setMode(mode: InferenceMode) {
    this.mode = mode
  }

  getMode(): InferenceMode {
    return this.mode
  }

  async runInference(imageBlob: Blob): Promise<InferenceResponse> {
    if (this.mode === 'onnx') {
      return await onnxInferenceService.runInference(imageBlob)
    }

    return await httpClient.post<InferenceResponse>('/inference', imageBlob, {
      headers: {
        'Content-Type': 'application/octet-stream',
      },
    })
  }

  async runInferenceFromDataURL(dataURL: string): Promise<InferenceResponse> {
    const blob = await fetch(dataURL).then(r => r.blob())
    return this.runInference(blob)
  }
}

export const inferenceService = new InferenceService()
