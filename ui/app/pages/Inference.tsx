import { useState, useRef, useEffect, useCallback } from 'react'
import { inferenceService, Detection, Metadata, InferenceMode } from '@animaldet/shared/api/inference'

function Inference() {
  const [image, setImage] = useState<string | null>(null)
  const [detections, setDetections] = useState<Detection[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [metadata, setMetadata] = useState<Metadata | null>(null)
  const [inferenceMode, setInferenceMode] = useState<InferenceMode>('api')
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
      }
      reader.readAsDataURL(file)
    }
  }

  const runInference = async () => {
    if (!image) return

    setLoading(true)
    try {
      inferenceService.setMode(inferenceMode)
      const result = await inferenceService.runInferenceFromDataURL(image)
      setDetections(result.data.detections)
      setMetadata(result.data.metadata)
      drawBoundingBoxes(result.data.detections)
    } catch (error) {
      console.error('Inference failed:', error)
      const errorMsg = inferenceMode === 'api'
        ? 'Failed to run inference. Make sure the API is running on port 8000.'
        : 'Failed to run ONNX inference. Make sure the model is loaded.'
      alert(errorMsg)
    } finally {
      setLoading(false)
    }
  }

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

        <div className="ml-auto flex gap-2 items-center">
          <span className="text-sm text-gray-600">Mode:</span>
          <button
            onClick={() => setInferenceMode('api')}
            className={`px-4 py-1 text-sm rounded ${
              inferenceMode === 'api'
                ? 'bg-black text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            API
          </button>
          <button
            onClick={() => setInferenceMode('onnx')}
            className={`px-4 py-1 text-sm rounded ${
              inferenceMode === 'onnx'
                ? 'bg-black text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            ONNX
          </button>
        </div>
      </div>

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

          {detections && (
            <div className="flex flex-col gap-4">
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
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default Inference
