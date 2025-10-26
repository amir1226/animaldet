import { useState, useRef, useEffect, useCallback } from 'react'
import { inferenceAPI, Detection, Metadata } from '../shared/api/inference'

function App() {
  const [image, setImage] = useState<string | null>(null)
  const [detections, setDetections] = useState<Detection[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [metadata, setMetadata] = useState<Metadata | null>(null)
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
      const result = await inferenceAPI.runInferenceFromDataURL(image)
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

    const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']

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
    <div className="bg-white rounded-3xl p-12 shadow-2xl">
      <header className="text-center mb-12">
        <h1 className="text-5xl font-bold bg-gradient-to-br from-purple-500 to-purple-700 bg-clip-text text-transparent mb-2">
          🦁 AnimalDet
        </h1>
        <p className="text-gray-600 text-lg">Animal Detection with AI</p>
      </header>

      <div className="flex gap-4 justify-center mb-8">
        <label
          htmlFor="file-upload"
          className="px-8 py-4 text-lg font-semibold bg-gradient-to-br from-purple-500 to-purple-700 text-white rounded-xl cursor-pointer transition-all hover:-translate-y-0.5 hover:shadow-xl hover:shadow-purple-500/40"
        >
          {image ? '📷 Change Image' : '📁 Upload Image'}
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
            className="px-8 py-4 text-lg font-semibold bg-teal-400 text-white rounded-xl transition-all hover:-translate-y-0.5 hover:shadow-xl hover:shadow-teal-400/40 disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:translate-y-0 disabled:hover:shadow-none"
          >
            {loading ? '⏳ Detecting...' : '🔍 Detect Animals'}
          </button>
        )}
      </div>

      {image && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 relative rounded-xl overflow-hidden bg-gray-100 inline-block w-full">
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
            <div className="flex flex-col gap-6">
              <div className="grid grid-cols-1 gap-4">
                <div className="bg-gradient-to-br from-purple-500 to-purple-700 text-white p-6 rounded-xl flex flex-col items-center gap-2">
                  <span className="text-3xl font-bold">{detections.length}</span>
                  <span className="text-sm opacity-90">Detections</span>
                </div>
                <div className="bg-gradient-to-br from-purple-500 to-purple-700 text-white p-6 rounded-xl flex flex-col items-center gap-2">
                  <span className="text-3xl font-bold">{metadata ? (metadata.latency_ms / 1000).toFixed(2) : '0'}s</span>
                  <span className="text-sm opacity-90">Latency</span>
                </div>
                <div className="bg-gradient-to-br from-purple-500 to-purple-700 text-white p-6 rounded-xl flex flex-col items-center gap-2">
                  <span className="text-3xl font-bold">{metadata ? `${metadata.input_shape[0]}×${metadata.input_shape[1]}` : 'N/A'}</span>
                  <span className="text-sm opacity-90">Resolution</span>
                </div>
              </div>

              <div className="bg-gray-50 rounded-xl p-6 max-h-96 overflow-y-auto">
                <h3 className="mb-4 text-gray-800 font-semibold">Detections</h3>
                {detections.map((det, idx) => (
                  <div key={idx} className="flex justify-between items-center p-3 mb-2 bg-white rounded-lg border-l-4 border-teal-400">
                    <span className="font-semibold text-gray-800">{det.class_name}</span>
                    <span className="text-teal-400 font-bold">
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

export default App
