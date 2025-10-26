import * as ort from 'onnxruntime-web'
import { Detection, InferenceResponse } from '@animaldet/shared/api/inference'

// Class names for the model (update based on your dataset)
const CLASS_NAMES = ['animal'] // Default, should be configurable

export interface OnnxInferenceConfig {
  modelPath: string
  inputSize: number
  confidenceThreshold: number
  classNames: string[]
}

export class OnnxInferenceService {
  private session: ort.InferenceSession | null = null
  private config: OnnxInferenceConfig
  private loading = false

  constructor(config: Partial<OnnxInferenceConfig> = {}) {
    this.config = {
      modelPath: '/model_web.onnx',
      inputSize: 512,
      confidenceThreshold: 0.3,
      classNames: CLASS_NAMES,
      ...config,
    }
  }

  async initialize(): Promise<void> {
    if (this.session || this.loading) return

    this.loading = true
    try {
      // Configure WASM paths for ONNX Runtime Web
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/'

      console.log('Loading ONNX model from', this.config.modelPath)
      this.session = await ort.InferenceSession.create(this.config.modelPath, {
        executionProviders: ['webgpu', 'webgl', 'wasm'], // Try GPU first
      })
      console.log('ONNX model loaded successfully')
    } catch (error) {
      console.error('Failed to load ONNX model:', error)
      throw error
    } finally {
      this.loading = false
    }
  }

  async runInference(imageBlob: Blob): Promise<InferenceResponse> {
    if (!this.session) {
      await this.initialize()
    }

    const startTime = performance.now()

    // Load image
    const imageBitmap = await createImageBitmap(imageBlob)
    const { width, height } = imageBitmap

    // Use stitching for large images
    const detections = (width > this.config.inputSize || height > this.config.inputSize)
      ? await this.runInferenceWithStitching(imageBitmap)
      : await this.runInferenceSingle(imageBitmap)

    const endTime = performance.now()

    return {
      data: {
        detections,
        metadata: {
          latency_ms: endTime - startTime,
          input_shape: [this.config.inputSize, this.config.inputSize],
        },
      },
    }
  }

  async runInferenceFromDataURL(dataURL: string): Promise<InferenceResponse> {
    const blob = await fetch(dataURL).then((r) => r.blob())
    return this.runInference(blob)
  }

  private async runInferenceSingle(imageBitmap: ImageBitmap): Promise<Detection[]> {
    // Preprocess entire image
    const inputTensor = await this.preprocessImage(imageBitmap)

    // Run inference
    const feeds = { images: inputTensor }
    const results = await this.session!.run(feeds)

    // Post-process results
    const predLogits = results.pred_logits
    const predBoxes = results.pred_boxes

    return this.postprocess(
      predLogits.data as Float32Array,
      predBoxes.data as Float32Array,
      predLogits.dims[1], // num_queries
      predLogits.dims[2], // num_classes
      imageBitmap.width,
      imageBitmap.height
    )
  }

  private async runInferenceWithStitching(imageBitmap: ImageBitmap): Promise<Detection[]> {
    const { width, height } = imageBitmap
    const { inputSize } = this.config

    // Calculate patches (non-overlapping for simplicity)
    const patches = this.createPatches(imageBitmap, inputSize)

    // Run inference on each patch
    const allDetections: Detection[] = []

    for (const patch of patches) {
      const inputTensor = await this.preprocessPatch(patch.imageData, inputSize)
      const feeds = { images: inputTensor }
      const results = await this.session!.run(feeds)

      const predLogits = results.pred_logits
      const predBoxes = results.pred_boxes

      // Get detections for this patch
      const patchDetections = this.postprocess(
        predLogits.data as Float32Array,
        predBoxes.data as Float32Array,
        predLogits.dims[1],
        predLogits.dims[2],
        inputSize,
        inputSize
      )

      // Rescale detections to original image coordinates
      for (const det of patchDetections) {
        det.bbox.x += patch.offsetX
        det.bbox.y += patch.offsetY
      }

      allDetections.push(...patchDetections)
    }

    // Apply NMS to remove duplicates
    return this.applyNMS(allDetections, 0.45)
  }

  private createPatches(
    imageBitmap: ImageBitmap,
    patchSize: number
  ): Array<{ imageData: ImageData; offsetX: number; offsetY: number }> {
    const { width, height } = imageBitmap
    const patches: Array<{ imageData: ImageData; offsetX: number; offsetY: number }> = []

    // Calculate number of patches needed
    const numPatchesX = Math.ceil(width / patchSize)
    const numPatchesY = Math.ceil(height / patchSize)

    // Create canvas to extract patches
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    const ctx = canvas.getContext('2d')!
    ctx.drawImage(imageBitmap, 0, 0)

    // Create patch canvas
    const patchCanvas = document.createElement('canvas')
    patchCanvas.width = patchSize
    patchCanvas.height = patchSize
    const patchCtx = patchCanvas.getContext('2d')!

    for (let py = 0; py < numPatchesY; py++) {
      for (let px = 0; px < numPatchesX; px++) {
        const offsetX = px * patchSize
        const offsetY = py * patchSize

        // Clear patch canvas
        patchCtx.fillStyle = 'black'
        patchCtx.fillRect(0, 0, patchSize, patchSize)

        // Extract patch from original image
        const srcWidth = Math.min(patchSize, width - offsetX)
        const srcHeight = Math.min(patchSize, height - offsetY)

        patchCtx.drawImage(
          canvas,
          offsetX, offsetY, srcWidth, srcHeight,
          0, 0, srcWidth, srcHeight
        )

        const imageData = patchCtx.getImageData(0, 0, patchSize, patchSize)
        patches.push({ imageData, offsetX, offsetY })
      }
    }

    return patches
  }

  private async preprocessPatch(imageData: ImageData, size: number): Promise<ort.Tensor> {
    const { data } = imageData

    // ImageNet normalization
    const mean = [0.485, 0.456, 0.406]
    const std = [0.229, 0.224, 0.225]

    // Convert to RGB and normalize with ImageNet stats
    const float32Data = new Float32Array(3 * size * size)
    for (let i = 0; i < size * size; i++) {
      float32Data[i] = (data[i * 4] / 255.0 - mean[0]) / std[0]
      float32Data[size * size + i] = (data[i * 4 + 1] / 255.0 - mean[1]) / std[1]
      float32Data[2 * size * size + i] = (data[i * 4 + 2] / 255.0 - mean[2]) / std[2]
    }

    return new ort.Tensor('float32', float32Data, [1, 3, size, size])
  }

  private async preprocessImage(imageBitmap: ImageBitmap): Promise<ort.Tensor> {
    const { inputSize } = this.config

    // Create canvas and resize image
    const canvas = document.createElement('canvas')
    canvas.width = inputSize
    canvas.height = inputSize
    const ctx = canvas.getContext('2d')!

    // Draw and resize image
    ctx.drawImage(imageBitmap, 0, 0, inputSize, inputSize)

    // Get image data
    const imageData = ctx.getImageData(0, 0, inputSize, inputSize)
    const { data } = imageData

    // ImageNet normalization (same as Python code)
    const mean = [0.485, 0.456, 0.406]
    const std = [0.229, 0.224, 0.225]

    // Convert to RGB and normalize with ImageNet stats
    const float32Data = new Float32Array(3 * inputSize * inputSize)
    for (let i = 0; i < inputSize * inputSize; i++) {
      // Normalize R channel
      float32Data[i] = (data[i * 4] / 255.0 - mean[0]) / std[0]
      // Normalize G channel
      float32Data[inputSize * inputSize + i] = (data[i * 4 + 1] / 255.0 - mean[1]) / std[1]
      // Normalize B channel
      float32Data[2 * inputSize * inputSize + i] = (data[i * 4 + 2] / 255.0 - mean[2]) / std[2]
    }

    // Create tensor with shape [1, 3, inputSize, inputSize]
    return new ort.Tensor('float32', float32Data, [1, 3, inputSize, inputSize])
  }

  private postprocess(
    logits: Float32Array,
    boxes: Float32Array,
    numQueries: number,
    numClasses: number,
    originalWidth: number,
    originalHeight: number
  ): Detection[] {
    const detections: Detection[] = []
    const { confidenceThreshold, classNames } = this.config

    for (let i = 0; i < numQueries; i++) {
      // Get class probabilities for this query
      const logitOffset = i * numClasses
      const probabilities = new Float32Array(numClasses)
      let maxProb = 0
      let maxClass = 0

      for (let c = 0; c < numClasses; c++) {
        // Sigmoid activation
        const prob = 1 / (1 + Math.exp(-logits[logitOffset + c]))
        probabilities[c] = prob

        if (prob > maxProb) {
          maxProb = prob
          maxClass = c
        }
      }

      // Filter by confidence threshold
      if (maxProb < confidenceThreshold) continue

      // Get bounding box (format: [cx, cy, w, h] normalized)
      const boxOffset = i * 4
      const cx = boxes[boxOffset]
      const cy = boxes[boxOffset + 1]
      const w = boxes[boxOffset + 2]
      const h = boxes[boxOffset + 3]

      // Convert to pixel coordinates (x, y, w, h)
      const x = (cx - w / 2) * originalWidth
      const y = (cy - h / 2) * originalHeight
      const width = w * originalWidth
      const height = h * originalHeight

      detections.push({
        bbox: {
          x,
          y,
          w: width,
          h: height,
          confidence: maxProb,
        },
        class_id: maxClass,
        class_name: classNames[maxClass] || `class_${maxClass}`,
      })
    }

    return detections
  }

  private applyNMS(detections: Detection[], iouThreshold: number): Detection[] {
    if (detections.length === 0) return []

    // Group detections by class
    const detectionsByClass = new Map<number, Detection[]>()
    for (const det of detections) {
      if (!detectionsByClass.has(det.class_id)) {
        detectionsByClass.set(det.class_id, [])
      }
      detectionsByClass.get(det.class_id)!.push(det)
    }

    // Apply NMS per class
    const result: Detection[] = []
    for (const [, classDets] of detectionsByClass) {
      result.push(...this.nmsForClass(classDets, iouThreshold))
    }

    return result
  }

  private nmsForClass(detections: Detection[], iouThreshold: number): Detection[] {
    if (detections.length === 0) return []

    // Sort by confidence (descending)
    const sorted = [...detections].sort((a, b) => b.bbox.confidence - a.bbox.confidence)

    const keep: Detection[] = []
    const suppressed = new Set<number>()

    for (let i = 0; i < sorted.length; i++) {
      if (suppressed.has(i)) continue

      const box1 = sorted[i]
      keep.push(box1)

      // Suppress overlapping boxes
      for (let j = i + 1; j < sorted.length; j++) {
        if (suppressed.has(j)) continue

        const box2 = sorted[j]
        const iou = this.calculateIoU(box1.bbox, box2.bbox)

        if (iou > iouThreshold) {
          suppressed.add(j)
        }
      }
    }

    return keep
  }

  private calculateIoU(box1: BoundingBox, box2: BoundingBox): number {
    const x1_min = box1.x
    const y1_min = box1.y
    const x1_max = box1.x + box1.w
    const y1_max = box1.y + box1.h

    const x2_min = box2.x
    const y2_min = box2.y
    const x2_max = box2.x + box2.w
    const y2_max = box2.y + box2.h

    // Calculate intersection
    const inter_x_min = Math.max(x1_min, x2_min)
    const inter_y_min = Math.max(y1_min, y2_min)
    const inter_x_max = Math.min(x1_max, x2_max)
    const inter_y_max = Math.min(y1_max, y2_max)

    const inter_width = Math.max(0, inter_x_max - inter_x_min)
    const inter_height = Math.max(0, inter_y_max - inter_y_min)
    const inter_area = inter_width * inter_height

    // Calculate union
    const box1_area = box1.w * box1.h
    const box2_area = box2.w * box2.h
    const union_area = box1_area + box2_area - inter_area

    return union_area > 0 ? inter_area / union_area : 0
  }

  isReady(): boolean {
    return this.session !== null
  }
}

export const onnxInferenceService = new OnnxInferenceService()
