import { useState, useRef, useEffect, useCallback } from 'react'
import { inferenceService, Detection, Metadata, GroundTruthResponse } from '@animaldet/shared/api/inference'
import { Loading } from '@animaldet/shared/components'
import { LARGE_SAMPLE_IMAGES } from '@animaldet/shared/constants/sampleImages'

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

interface DetectionHistoryItem {
  id: string
  timestamp: Date
  image: string
  imageName?: string
  detections: Detection[]
  metadata: Metadata
  model: string
  confidenceThreshold: number
  groundTruth?: Detection[]
}

function Inference() {
  const [image, setImage] = useState<string | null>(null)
  const [currentImageName, setCurrentImageName] = useState<string | null>(null)
  const [detections, setDetections] = useState<Detection[] | null>(null)
  const [groundTruth, setGroundTruth] = useState<Detection[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [metadata, setMetadata] = useState<Metadata | null>(null)
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5)
  const [selectedModel, setSelectedModel] = useState<'nano' | 'small'>('small')
  const [availableModels, setAvailableModels] = useState<ModelsResponse | null>(null)
  const [detectionHistory, setDetectionHistory] = useState<DetectionHistoryItem[]>([])
  const [modalItem, setModalItem] = useState<DetectionHistoryItem | null>(null)
  const [sampleImages, setSampleImages] = useState<typeof LARGE_SAMPLE_IMAGES>([])
  const [showGroundTruth, setShowGroundTruth] = useState(true)
  const [showInferences, setShowInferences] = useState(true)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const gtCanvasRef = useRef<HTMLCanvasElement>(null)
  const gtImgRef = useRef<HTMLImageElement>(null)
  const modalCanvasRef = useRef<HTMLCanvasElement>(null)
  const modalImgRef = useRef<HTMLImageElement>(null)

  // Recommended confidence per model
  const recommendedConfidence: Record<string, number> = {
    small: 0.5,
    nano: 0.79,
  }

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

  const loadSampleImage = async (imageName: string) => {
    try {
      const response = await fetch(`/demo_images/${imageName}`)
      const blob = await response.blob()
      const reader = new FileReader()
      reader.onload = async (e) => {
        setImage(e.target?.result as string)
        setCurrentImageName(imageName)
        setDetections(null)
        setMetadata(null)
        setGroundTruth(null)

        // Fetch ground truth for this image
        const gtResponse = await inferenceService.getGroundTruth(imageName)
        if (gtResponse.available) {
          setGroundTruth(gtResponse.detections)
        }

        // Clear canvas when image changes
        const canvas = canvasRef.current
        if (canvas) {
          const ctx = canvas.getContext('2d')
          if (ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height)
          }
        }
      }
      reader.readAsDataURL(blob)
    } catch (error) {
      console.error('Failed to load sample image:', error)
      alert('Failed to load sample image')
    }
  }

  const downloadSampleImage = (imageName: string) => {
    const link = document.createElement('a')
    link.href = `/demo_images/${imageName}`
    link.download = imageName
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
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
      drawBoundingBoxes(result.data.detections, groundTruth)

      // Save to history
      const historyItem: DetectionHistoryItem = {
        id: Date.now().toString(),
        timestamp: new Date(),
        image,
        imageName: currentImageName || undefined,
        detections: result.data.detections,
        metadata: result.data.metadata,
        model: selectedModel,
        confidenceThreshold,
        groundTruth: groundTruth || undefined,
      }
      setDetectionHistory(prev => [historyItem, ...prev])
    } catch (error) {
      console.error('Inference failed:', error)
      alert('Failed to run inference. Make sure the API is running on port 8000.')
    } finally {
      setLoading(false)
    }
  }

  // Load available models and sample images on mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch('/api/models')
        const data: ModelsResponse = await response.json()
        setAvailableModels(data)
        const defaultModel = data.default as 'nano' | 'small'
        setSelectedModel(defaultModel)
        setConfidenceThreshold(recommendedConfidence[defaultModel] || 0.5)
      } catch (error) {
        console.error('Failed to fetch models:', error)
      }
    }

    fetchModels()
    setSampleImages(LARGE_SAMPLE_IMAGES)
  }, [])

  // Update confidence when model changes
  const handleModelChange = (model: 'nano' | 'small') => {
    setSelectedModel(model)
    setConfidenceThreshold(recommendedConfidence[model] || 0.5)
  }

  // Open modal to view detection
  const openModal = (item: DetectionHistoryItem) => {
    setModalItem(item)
  }

  // Close modal
  const closeModal = () => {
    setModalItem(null)
  }

  // Draw bounding boxes on modal
  const drawModalBoundingBoxes = useCallback((detections: Detection[], groundTruthDets?: Detection[]) => {
    const canvas = modalCanvasRef.current
    const img = modalImgRef.current
    if (!canvas || !img || !img.complete) return

    setTimeout(() => {
      const displayWidth = img.offsetWidth
      const displayHeight = img.offsetHeight

      canvas.width = displayWidth
      canvas.height = displayHeight
      canvas.style.width = `${displayWidth}px`
      canvas.style.height = `${displayHeight}px`

      const scaleX = displayWidth / img.naturalWidth
      const scaleY = displayHeight / img.naturalHeight

      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      const colors = ['#D97706', '#3B82F6', '#10B981', '#EF4444', '#8B5CF6', '#F59E0B']

      // Draw ground truth boxes first (dashed)
      if (groundTruthDets && showGroundTruth) {
        groundTruthDets.forEach((det) => {
          const { x, y, w, h, confidence } = det.bbox
          const color = colors[det.class_id % colors.length]

          const scaledX = x * scaleX
          const scaledY = y * scaleY
          const scaledW = w * scaleX
          const scaledH = h * scaleY

          ctx.strokeStyle = color
          ctx.lineWidth = 2
          ctx.setLineDash([5, 5])
          ctx.strokeRect(scaledX, scaledY, scaledW, scaledH)

          const label = `GT: ${det.class_name} ${(confidence * 100).toFixed(0)}%`
          ctx.font = 'bold 11px sans-serif'
          const textMetrics = ctx.measureText(label)
          const textWidth = textMetrics.width
          const textHeight = 16
          const padding = 4

          const labelY = scaledY > textHeight + padding ? scaledY - 2 : scaledY + textHeight + padding

          ctx.fillStyle = color
          ctx.fillRect(scaledX, labelY - textHeight, textWidth + padding * 2, textHeight)

          ctx.fillStyle = '#fff'
          ctx.fillText(label, scaledX + padding, labelY - 4)
        })
      }

      // Draw inference detections (solid)
      if (showInferences) {
        ctx.setLineDash([])
        detections.forEach((det) => {
          const { x, y, w, h, confidence } = det.bbox
          const color = colors[det.class_id % colors.length]

          const scaledX = x * scaleX
          const scaledY = y * scaleY
          const scaledW = w * scaleX
          const scaledH = h * scaleY

          ctx.strokeStyle = color
          ctx.lineWidth = 2
          ctx.strokeRect(scaledX, scaledY, scaledW, scaledH)

          const label = `${det.class_name} ${(confidence * 100).toFixed(0)}%`
          ctx.font = 'bold 11px sans-serif'
          const textMetrics = ctx.measureText(label)
          const textWidth = textMetrics.width
          const textHeight = 16
          const padding = 4

          const labelY = scaledY > textHeight + padding ? scaledY - 2 : scaledY + textHeight + padding

          ctx.fillStyle = color
          ctx.fillRect(scaledX, labelY - textHeight, textWidth + padding * 2, textHeight)

          ctx.fillStyle = '#fff'
          ctx.fillText(label, scaledX + padding, labelY - 4)
        })
      }
    }, 100)
  }, [showGroundTruth, showInferences])

  const drawGroundTruthOnly = useCallback((groundTruthDets: Detection[]) => {
    const canvas = gtCanvasRef.current
    const img = gtImgRef.current
    if (!canvas || !img || !img.complete) return

    setTimeout(() => {
      const displayWidth = img.offsetWidth
      const displayHeight = img.offsetHeight

      canvas.width = displayWidth
      canvas.height = displayHeight
      canvas.style.width = `${displayWidth}px`
      canvas.style.height = `${displayHeight}px`

      const scaleX = displayWidth / img.naturalWidth
      const scaleY = displayHeight / img.naturalHeight

      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      const colors = ['#D97706', '#3B82F6', '#10B981', '#EF4444', '#8B5CF6', '#F59E0B']

      groundTruthDets.forEach((det) => {
        const { x, y, w, h } = det.bbox
        const color = colors[det.class_id % colors.length]

        const scaledX = x * scaleX
        const scaledY = y * scaleY
        const scaledW = w * scaleX
        const scaledH = h * scaleY

        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.setLineDash([])
        ctx.strokeRect(scaledX, scaledY, scaledW, scaledH)

        const label = `${det.class_name}`
        ctx.font = 'bold 11px sans-serif'
        const textMetrics = ctx.measureText(label)
        const textWidth = textMetrics.width
        const textHeight = 16
        const padding = 4

        const labelY = scaledY > textHeight + padding ? scaledY - 2 : scaledY + textHeight + padding

        ctx.fillStyle = color
        ctx.fillRect(scaledX, labelY - textHeight, textWidth + padding * 2, textHeight)

        ctx.fillStyle = '#fff'
        ctx.fillText(label, scaledX + padding, labelY - 4)
      })
    }, 100)
  }, [])

  const drawInferenceOnly = useCallback((detections: Detection[]) => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img || !img.complete) return

    setTimeout(() => {
      const displayWidth = img.offsetWidth
      const displayHeight = img.offsetHeight

      canvas.width = displayWidth
      canvas.height = displayHeight
      canvas.style.width = `${displayWidth}px`
      canvas.style.height = `${displayHeight}px`

      const scaleX = displayWidth / img.naturalWidth
      const scaleY = displayHeight / img.naturalHeight

      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      const colors = ['#D97706', '#3B82F6', '#10B981', '#EF4444', '#8B5CF6', '#F59E0B']

      detections.forEach((det) => {
        const { x, y, w, h, confidence } = det.bbox
        const color = colors[det.class_id % colors.length]

        const scaledX = x * scaleX
        const scaledY = y * scaleY
        const scaledW = w * scaleX
        const scaledH = h * scaleY

        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.strokeRect(scaledX, scaledY, scaledW, scaledH)

        const label = `${det.class_name} ${(confidence * 100).toFixed(0)}%`
        ctx.font = 'bold 11px sans-serif'
        const textMetrics = ctx.measureText(label)
        const textWidth = textMetrics.width
        const textHeight = 16
        const padding = 4

        const labelY = scaledY > textHeight + padding ? scaledY - 2 : scaledY + textHeight + padding

        ctx.fillStyle = color
        ctx.fillRect(scaledX, labelY - textHeight, textWidth + padding * 2, textHeight)

        ctx.fillStyle = '#fff'
        ctx.fillText(label, scaledX + padding, labelY - 4)
      })
    }, 100)
  }, [])

  const drawBoundingBoxes = useCallback((detections: Detection[], groundTruthDets?: Detection[] | null) => {
    if (groundTruthDets && groundTruthDets.length > 0) {
      drawGroundTruthOnly(groundTruthDets)
    }
    drawInferenceOnly(detections)
  }, [drawGroundTruthOnly, drawInferenceOnly])

  useEffect(() => {
    const handleResize = () => {
      if (detections) {
        drawBoundingBoxes(detections, groundTruth)
      }
    }
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [detections, groundTruth, drawBoundingBoxes])

  // Redraw modal when showGroundTruth or showInferences changes
  useEffect(() => {
    if (modalItem) {
      drawModalBoundingBoxes(modalItem.detections, modalItem.groundTruth)
    }
  }, [showGroundTruth, showInferences, modalItem, drawModalBoundingBoxes])

  return (
    <div className="mx-auto p-6">
      <Loading show={loading} />
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
              onChange={(e) => handleModelChange(e.target.value as 'nano' | 'small')}
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
              min="0.1"
              max="1"
              step="0.01"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
              className="w-32"
            />
          </div>
        </div>
      </div>

      {/* Confidence warnings and recommendations */}
      <div className="mb-6 space-y-2">
        {confidenceThreshold === 0.1 && (
          <div className="border border-amber-400 bg-amber-50 p-3 rounded text-sm text-amber-800">
            ⚠️ With less than 10% confidence the model will perform pretty bad
          </div>
        )}
        {confidenceThreshold !== recommendedConfidence[selectedModel] && (
          <div className="border border-blue-400 bg-blue-50 p-3 rounded text-sm text-blue-800">
            💡 Recommended confidence for {selectedModel.toUpperCase()}: {(recommendedConfidence[selectedModel] * 100).toFixed(0)}%
          </div>
        )}
      </div>

      {/* Sample Images Section */}
      {sampleImages.length > 0 && !image && (
        <div className="border border-gray-300 p-6 bg-gray-50 mb-6">
          <h2 className="text-lg font-semibold mb-2">Sample Images for Testing</h2>
          <p className="text-sm text-gray-600 mb-4">
            Click to load an image or download it for later use
          </p>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {sampleImages.map((sample) => (
              <div
                key={sample.name}
                className="border border-gray-300 rounded overflow-hidden bg-white hover:shadow-lg transition-shadow group"
              >
                <div
                  className="cursor-pointer relative aspect-square bg-gray-200 overflow-hidden"
                  onClick={() => loadSampleImage(sample.name)}
                >
                  <img
                    src={`/demo_images/${sample.name}`}
                    alt={sample.name}
                    className="w-full h-full object-cover transition-opacity duration-300"
                    onLoad={(e) => {
                      const target = e.target as HTMLImageElement
                      target.style.opacity = '1'
                    }}
                    onError={(e) => {
                      console.error(`Failed to load image: ${sample.name}`)
                      const target = e.target as HTMLImageElement
                      target.style.display = 'none'
                      const parent = target.parentElement
                      if (parent) {
                        parent.classList.remove('bg-gray-200')
                        parent.classList.add('flex', 'items-center', 'justify-center', 'bg-red-50')
                        parent.innerHTML = '<div class="text-center p-4"><svg class="w-12 h-12 mx-auto text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg><p class="text-xs text-gray-500 mt-2">Failed to load</p></div>'
                      }
                    }}
                    style={{ opacity: 0 }}
                  />
                </div>
                <div className="p-2">
                  <p className="text-xs font-semibold text-gray-800 mb-1">{sample.class}</p>
                  <p className="text-xs text-gray-500 mb-2 truncate" title={sample.name}>{sample.name}</p>
                  <button
                    onClick={() => downloadSampleImage(sample.name)}
                    className="w-full text-xs px-2 py-1 bg-black text-white rounded hover:bg-gray-800"
                  >
                    Download
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Model info card - always visible when models are loaded */}
      {availableModels && !image && (
        <div className="border border-gray-300 p-6 bg-gray-50 mb-6">
          <h2 className="text-lg font-semibold mb-4">Available Models</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(availableModels.models).map(([key, model]) => (
              <div
                key={key}
                onClick={() => handleModelChange(key as 'nano' | 'small')}
                className={`p-4 border-2 rounded cursor-pointer transition-all ${
                  selectedModel === key
                    ? 'border-black bg-white shadow-md'
                    : 'border-gray-300 hover:border-gray-400 hover:bg-white'
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
        <>
          {/* Sample Images Selector - shown when image is loaded */}
          {sampleImages.length > 0 && (
            <div className="border border-gray-300 p-4 bg-gray-50 mb-6">
              <h3 className="text-sm font-semibold mb-3 text-gray-700">Quick Switch Sample Images</h3>
              <div className="flex gap-3 overflow-x-auto pb-2">
                {sampleImages.map((sample) => (
                  <div
                    key={sample.name}
                    onClick={() => loadSampleImage(sample.name)}
                    className={`flex-shrink-0 w-24 border-2 rounded overflow-hidden cursor-pointer transition-all ${
                      currentImageName === sample.name
                        ? 'border-black shadow-md'
                        : 'border-gray-300 hover:border-gray-400'
                    }`}
                  >
                    <img
                      src={`/demo_images/${sample.name}`}
                      alt={sample.class}
                      className="w-full h-20 object-cover"
                    />
                    <div className="bg-white px-1 py-0.5 text-xs text-center font-semibold text-gray-700">
                      {sample.class}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              {groundTruth && groundTruth.length > 0 ? (
                <div className="grid grid-cols-2 gap-4">
                {/* Ground Truth Image */}
                <div className="border border-gray-300">
                  <div className="bg-green-50 border-b border-gray-300 px-3 py-2">
                    <h3 className="text-sm font-semibold text-green-700">Ground Truth ({groundTruth.length})</h3>
                  </div>
                  <div className="relative">
                    <img
                      ref={gtImgRef}
                      src={image}
                      alt="Ground Truth"
                      onLoad={() => groundTruth && drawGroundTruthOnly(groundTruth)}
                      className="w-full h-auto block"
                    />
                    <canvas ref={gtCanvasRef} className="absolute top-0 left-0 pointer-events-none" />
                  </div>
                </div>

                {/* Inference Image */}
                <div className="border border-gray-300">
                  <div className="bg-amber-50 border-b border-gray-300 px-3 py-2">
                    <h3 className="text-sm font-semibold text-amber-700">Inference ({detections?.length || 0})</h3>
                  </div>
                  <div className="relative">
                    <img
                      ref={imgRef}
                      src={image}
                      alt="Inference"
                      onLoad={() => detections && drawInferenceOnly(detections)}
                      className="w-full h-auto block"
                    />
                    <canvas ref={canvasRef} className="absolute top-0 left-0 pointer-events-none" />
                  </div>
                </div>
              </div>
              ) : (
                <div className="relative border border-gray-300">
                  <img
                    ref={imgRef}
                    src={image}
                    alt="Inference"
                    onLoad={() => detections && drawInferenceOnly(detections)}
                    className="w-full h-auto block"
                  />
                  <canvas ref={canvasRef} className="absolute top-0 left-0 pointer-events-none" />
                </div>
              )}
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
                  {groundTruth && groundTruth.length > 0 && (
                    <div className="text-xs text-gray-500 mt-1">
                      Ground Truth: {groundTruth.length}
                    </div>
                  )}
                </div>

                <div className="border border-gray-300 p-4">
                  <div className="text-sm text-gray-600 mb-1">Latency</div>
                  <div className="text-2xl font-bold">{metadata ? (metadata.latency_ms / 1000).toFixed(2) : '0'}s</div>
                </div>

                {metadata?.gpu_memory_mb !== undefined && metadata.gpu_memory_mb !== null && (
                  <div className="border border-gray-300 p-4">
                    <div className="text-sm text-gray-600 mb-1">GPU Memory</div>
                    <div className="text-2xl font-bold">{metadata.gpu_memory_mb.toFixed(0)} MB</div>
                  </div>
                )}

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
        </>
      )}

      {/* Detection History Queue */}
      {detectionHistory.length > 0 && (
        <div className="mt-8">
          <h2 className="text-xl font-bold mb-4">Detection History</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
            {detectionHistory.map((item) => (
              <div
                key={item.id}
                onClick={() => openModal(item)}
                className="border border-gray-300 rounded overflow-hidden cursor-pointer hover:shadow-lg transition-shadow bg-white"
              >
                <div className="relative aspect-square">
                  <img
                    src={item.image}
                    alt="Detection"
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute top-2 right-2 bg-black bg-opacity-75 text-white text-xs px-2 py-1 rounded">
                    {item.detections.length} detected
                  </div>
                </div>
                <div className="p-3 space-y-1">
                  <div className="text-xs text-gray-500">
                    {item.timestamp.toLocaleTimeString()}
                  </div>
                  <div className="text-sm font-semibold">
                    {item.model.toUpperCase()}
                  </div>
                  <div className="text-xs text-gray-600 flex justify-between">
                    <span>Conf: {(item.confidenceThreshold * 100).toFixed(0)}%</span>
                    <span>{(item.metadata.latency_ms / 1000).toFixed(2)}s</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Modal for viewing detection details */}
      {modalItem && (
        <div
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4"
          onClick={closeModal}
        >
          <div
            className="bg-white rounded-lg max-w-6xl w-full max-h-[90vh] overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="sticky top-0 bg-white border-b border-gray-300 p-4 flex justify-between items-center">
              <h2 className="text-xl font-bold">Detection Details</h2>
              <button
                onClick={closeModal}
                className="text-gray-500 hover:text-black text-2xl font-bold w-8 h-8 flex items-center justify-center"
              >
                ×
              </button>
            </div>

            <div className="p-6">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2">
                  <div className="relative border border-gray-300">
                    <img
                      ref={modalImgRef}
                      src={modalItem.image}
                      alt="Detection"
                      onLoad={() => drawModalBoundingBoxes(modalItem.detections, modalItem.groundTruth)}
                      className="w-full h-auto block"
                    />
                    <canvas
                      ref={modalCanvasRef}
                      className="absolute top-0 left-0 pointer-events-none"
                    />
                    {(modalItem.detections || (modalItem.groundTruth && modalItem.groundTruth.length > 0)) && (
                      <div className="absolute top-2 right-2 bg-white border border-gray-300 rounded p-2 shadow-md space-y-2">
                        {modalItem.detections && (
                          <label className="flex items-center gap-2 text-sm cursor-pointer">
                            <input
                              type="checkbox"
                              checked={showInferences}
                              onChange={(e) => {
                                const newValue = e.target.checked
                                setShowInferences(newValue)
                              }}
                              className="cursor-pointer"
                            />
                            <span>Show Inferences</span>
                          </label>
                        )}
                        {modalItem.groundTruth && modalItem.groundTruth.length > 0 && (
                          <label className="flex items-center gap-2 text-sm cursor-pointer">
                            <input
                              type="checkbox"
                              checked={showGroundTruth}
                              onChange={(e) => {
                                const newValue = e.target.checked
                                setShowGroundTruth(newValue)
                              }}
                              className="cursor-pointer"
                            />
                            <span>Show Ground Truth</span>
                          </label>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                <div className="flex flex-col gap-4">
                  {/* Detection Info */}
                  <div className="border border-gray-300 p-4 bg-gray-50">
                    <div className="text-sm font-semibold text-gray-700 mb-2">Detection Info</div>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Time:</span>
                        <span className="font-medium">{modalItem.timestamp.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Model:</span>
                        <span className="font-medium">{modalItem.model.toUpperCase()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Confidence:</span>
                        <span className="font-medium">{(modalItem.confidenceThreshold * 100).toFixed(0)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Latency:</span>
                        <span className="font-medium">{(modalItem.metadata.latency_ms / 1000).toFixed(2)}s</span>
                      </div>
                      {modalItem.metadata.gpu_memory_mb !== undefined && modalItem.metadata.gpu_memory_mb !== null && (
                        <div className="flex justify-between">
                          <span className="text-gray-600">GPU Memory:</span>
                          <span className="font-medium">{modalItem.metadata.gpu_memory_mb.toFixed(0)} MB</span>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="border border-gray-300 p-4">
                    <div className="text-sm text-gray-600 mb-1">Detections</div>
                    <div className="text-2xl font-bold">{modalItem.detections.length}</div>
                    {modalItem.groundTruth && modalItem.groundTruth.length > 0 && (
                      <div className="text-xs text-gray-500 mt-1">
                        Ground Truth: {modalItem.groundTruth.length}
                      </div>
                    )}
                  </div>

                  <div className="border border-gray-300 p-4 max-h-96 overflow-y-auto">
                    <h3 className="text-sm font-semibold mb-3 text-gray-600">Results</h3>
                    {modalItem.detections.map((det, idx) => (
                      <div key={idx} className="flex justify-between items-center py-2 border-b border-gray-200 last:border-0">
                        <span className="text-sm">{det.class_name}</span>
                        <span className="text-sm font-mono text-amber-700">
                          {(det.bbox.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Inference
