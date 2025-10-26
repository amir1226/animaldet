import { useState, useRef, useEffect, useCallback } from 'react'
import './App.css'

function App() {
  const [image, setImage] = useState(null)
  const [detections, setDetections] = useState(null)
  const [loading, setLoading] = useState(false)
  const [metadata, setMetadata] = useState(null)
  const canvasRef = useRef(null)
  const imgRef = useRef(null)

  const handleImageUpload = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setImage(e.target.result)
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
      const blob = await fetch(image).then(r => r.blob())
      const response = await fetch('/api/inference', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/octet-stream',
        },
        body: blob,
      })

      const result = await response.json()
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

  const drawBoundingBoxes = useCallback((detections) => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img || !img.complete) return

    // Wait a bit for image to fully render
    setTimeout(() => {
      // Set canvas to match displayed image size
      const rect = img.getBoundingClientRect()
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
      ctx.clearRect(0, 0, canvas.width, canvas.height)

    const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']

    detections.forEach((det, idx) => {
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
    <div className="app">
      <header className="header">
        <h1>🦁 AnimalDet</h1>
        <p>Animal Detection with AI</p>
      </header>

      <div className="upload-section">
        <label htmlFor="file-upload" className="upload-button">
          {image ? '📷 Change Image' : '📁 Upload Image'}
        </label>
        <input
          id="file-upload"
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          style={{ display: 'none' }}
        />

        {image && (
          <button
            onClick={runInference}
            disabled={loading}
            className="detect-button"
          >
            {loading ? '⏳ Detecting...' : '🔍 Detect Animals'}
          </button>
        )}
      </div>

      {image && (
        <div className="result-container">
          <div className="image-container">
            <img
              ref={imgRef}
              src={image}
              alt="Uploaded"
              onLoad={() => detections && drawBoundingBoxes(detections)}
            />
            <canvas ref={canvasRef} className="canvas-overlay" />
          </div>

          {detections && (
            <div className="detections-panel">
              <div className="stats">
                <div className="stat-card">
                  <span className="stat-value">{detections.length}</span>
                  <span className="stat-label">Detections</span>
                </div>
                <div className="stat-card">
                  <span className="stat-value">{(metadata.latency_ms / 1000).toFixed(2)}s</span>
                  <span className="stat-label">Latency</span>
                </div>
                <div className="stat-card">
                  <span className="stat-value">{metadata.input_shape[0]}×{metadata.input_shape[1]}</span>
                  <span className="stat-label">Resolution</span>
                </div>
              </div>

              <div className="detections-list">
                <h3>Detections</h3>
                {detections.map((det, idx) => (
                  <div key={idx} className="detection-item">
                    <span className="detection-class">{det.class_name}</span>
                    <span className="detection-confidence">
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
