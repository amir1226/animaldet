import { useState, useRef, useEffect, useCallback } from 'react'
import { inferenceService, Detection, Metadata } from '@animaldet/shared/api/inference'

interface ModelInfo {
  name: string
  model_path: string
  resolution: number
  num_classes: number
  description: string
}

interface ModelsResponse {
  models: Record<string, ModelInfo>
  default: string
  loaded: string[]
  current: string
}

function Inference() {
  const [image, setImage] = useState<string | null>(null)
  const [detections, setDetections] = useState<Detection[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [metadata, setMetadata] = useState<Metadata | null>(null)
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5)
  const [selectedModel, setSelectedModel] = useState<'nano' | 'small'>('small')
  const [availableModels, setAvailableModels] = useState<ModelsResponse | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setImage(e.target?.result as string)
        setDetections(null)
        setMetadata(null)
        // Clear canvas when image changes
        const canvas = canvasRef.current
        if (canvas) {
          const ctx = canvas.getContext('2d')
          if (ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height)
          }
        }
      }
      reader.readAsDataURL(file)
    }
  }

  const runInference = async () => {
    if (!image) return

    setLoading(true)
    try {
      const result = await inferenceService.runInferenceFromDataURL(image, {
        confidenceThreshold,
        model: selectedModel,
      })
      setDetections(result.data.detections)
      setMetadata(result.data.metadata)
      drawBoundingBoxes(result.data.detections)
    } catch (error) {
      console.error('Inference failed:', error)
      alert('Failed to run inference. Make sure the API is running on port 8000.')
    } finally {
      setLoading(false)
    }
  }

  // Load available models on mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch('/api/models')
        const data: ModelsResponse = await response.json()
        setAvailableModels(data)
        setSelectedModel(data.default as 'nano' | 'small')
      } catch (error) {
        console.error('Failed to fetch models:', error)
      }
    }
    fetchModels()
  }, [])

  const drawBoundingBoxes = useCallback((detections: Detection[]) => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img || !img.complete) return

    // Wait a bit for image to fully render
    setTimeout(() => {
      // Set canvas to match displayed image size
      const displayWidth = img.offsetWidth
      const displayHeight = img.offsetHeight

      canvas.width = displayWidth
      canvas.height = displayHeight
      canvas.style.width = `${displayWidth}px`
      canvas.style.height = `${displayHeight}px`

      // Calculate scale factors
      const scaleX = displayWidth / img.naturalWidth
      const scaleY = displayHeight / img.naturalHeight

      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.clearRect(0, 0, canvas.width, canvas.height)

    const colors = ['#D97706', '#000000', '#92400E', '#451A03']

    detections.forEach((det) => {
      const { x, y, w, h, confidence } = det.bbox
      const color = colors[det.class_id % colors.length]

      // Scale coordinates to match displayed image
      const scaledX = x * scaleX
      const scaledY = y * scaleY
      const scaledW = w * scaleX
      const scaledH = h * scaleY

      // Draw bounding box
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.strokeRect(scaledX, scaledY, scaledW, scaledH)

      // Draw label
      const label = `${det.class_name} ${(confidence * 100).toFixed(0)}%`
      ctx.font = 'bold 11px sans-serif'
      const textMetrics = ctx.measureText(label)
      const textWidth = textMetrics.width
      const textHeight = 16
      const padding = 4

      // Position label above box, or inside if too close to top
      const labelY = scaledY > textHeight + padding ? scaledY - 2 : scaledY + textHeight + padding

      // Draw label background
      ctx.fillStyle = color
      ctx.fillRect(scaledX, labelY - textHeight, textWidth + padding * 2, textHeight)

      // Draw label text
      ctx.fillStyle = '#fff'
      ctx.fillText(label, scaledX + padding, labelY - 4)
    })
    }, 100)
  }, [])

  useEffect(() => {
    const handleResize = () => {
      if (detections) {
        drawBoundingBoxes(detections)
      }
    }
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [detections, drawBoundingBoxes])

  return (
    <div className="max-w-6xl mx-auto">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-black mb-1">
          AnimalDet
        </h1>
        <p className="text-gray-600">Animal Detection with AI</p>
      </header>

      <div className="flex gap-3 mb-6 items-center">
        <label
          htmlFor="file-upload"
          className="px-6 py-2 bg-black text-white rounded cursor-pointer hover:bg-gray-800"
        >
          {image ? 'Change Image' : 'Upload Image'}
        </label>
        <input
          id="file-upload"
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          className="hidden"
        />

        {image && (
          <button
            onClick={runInference}
            disabled={loading}
            className="px-6 py-2 bg-black text-white rounded hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Detecting...' : 'Detect Animals'}
          </button>
        )}

        <div className="flex items-center gap-6 ml-auto">
          <div className="flex items-center gap-2">
            <label htmlFor="model-select" className="text-sm text-gray-700 whitespace-nowrap">
              Model:
            </label>
            <select
              id="model-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value as 'nano' | 'small')}
              className="px-3 py-1 border border-gray-300 rounded text-sm"
            >
              {availableModels && Object.entries(availableModels.models).map(([key, model]) => (
                <option key={key} value={key}>
                  {key.toUpperCase()} ({model.resolution}x{model.resolution})
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-3">
            <label htmlFor="confidence-slider" className="text-sm text-gray-700 whitespace-nowrap">
              Confidence: {(confidenceThreshold * 100).toFixed(0)}%
            </label>
            <input
              id="confidence-slider"
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
              className="w-32"
            />
          </div>
        </div>
      </div>

      {/* Model info card - always visible when models are loaded */}
      {availableModels && !image && (
        <div className="border border-gray-300 p-6 bg-gray-50 mb-6">
          <h2 className="text-lg font-semibold mb-4">Available Models</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(availableModels.models).map(([key, model]) => (
              <div
                key={key}
                className={`p-4 border-2 rounded ${
                  selectedModel === key ? 'border-black bg-white' : 'border-gray-300'
                }`}
              >
                <div className="font-bold text-lg mb-1">{key.toUpperCase()}</div>
                <div className="text-sm text-gray-600 mb-3">{model.description}</div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-gray-600">Resolution:</span>
                    <div className="font-medium">{model.resolution}x{model.resolution}</div>
                  </div>
                  <div>
                    <span className="text-gray-600">Classes:</span>
                    <div className="font-medium">{model.num_classes}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {image && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 relative border border-gray-300">
            <img
              ref={imgRef}
              src={image}
              alt="Uploaded"
              onLoad={() => detections && drawBoundingBoxes(detections)}
              className="w-full h-auto block"
            />
            <canvas ref={canvasRef} className="absolute top-0 left-0 pointer-events-none" />
          </div>

          <div className="flex flex-col gap-4">
            {/* Model Info */}
            {availableModels && (
              <div className="border border-gray-300 p-4 bg-gray-50">
                <div className="text-sm font-semibold text-gray-700 mb-2">Active Model</div>
                <div className="text-lg font-bold mb-1">{selectedModel.toUpperCase()}</div>
                <div className="text-xs text-gray-600 mb-2">
                  {availableModels.models[selectedModel]?.description}
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-gray-600">Resolution:</span>
                    <div className="font-medium">{availableModels.models[selectedModel]?.resolution}x{availableModels.models[selectedModel]?.resolution}</div>
                  </div>
                  <div>
                    <span className="text-gray-600">Classes:</span>
                    <div className="font-medium">{availableModels.models[selectedModel]?.num_classes}</div>
                  </div>
                </div>
              </div>
            )}

            {detections && (
              <>
                <div className="border border-gray-300 p-4">
                  <div className="text-sm text-gray-600 mb-1">Detections</div>
                  <div className="text-2xl font-bold">{detections.length}</div>
                </div>

                <div className="border border-gray-300 p-4">
                  <div className="text-sm text-gray-600 mb-1">Latency</div>
                  <div className="text-2xl font-bold">{metadata ? (metadata.latency_ms / 1000).toFixed(2) : '0'}s</div>
                </div>

                <div className="border border-gray-300 p-4 max-h-96 overflow-y-auto">
                  <h3 className="text-sm font-semibold mb-3 text-gray-600">Results</h3>
                  {detections.map((det, idx) => (
                    <div key={idx} className="flex justify-between items-center py-2 border-b border-gray-200 last:border-0">
                      <span className="text-sm">{det.class_name}</span>
                      <span className="text-sm font-mono text-amber-700">
                        {(det.bbox.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default Inference
