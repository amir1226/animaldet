import { useEffect, useRef } from 'react'
import { ExperimentDetection } from '@animaldet/shared/api/experiments'

interface BoundingBoxCanvasProps {
  imageUrl: string
  detections: ExperimentDetection[]
  color: string
  opacity?: number
}

export default function BoundingBoxCanvas({
  imageUrl,
  detections,
  color,
  opacity = 1
}: BoundingBoxCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img) return

    const draw = () => {
      // Wait for image to be loaded
      if (!img.complete || img.naturalWidth === 0) return

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
      ctx.globalAlpha = opacity

      detections.forEach((det) => {
        const x = det.x * scaleX
        const y = det.y * scaleY
        const width = (det.x_max - det.x) * scaleX
        const height = (det.y_max - det.y) * scaleY

        // Draw bounding box
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.strokeRect(x, y, width, height)

        // Draw label
        const label = `${det.label} ${(det.score * 100).toFixed(0)}%`
        ctx.font = 'bold 11px sans-serif'
        const textMetrics = ctx.measureText(label)
        const textWidth = textMetrics.width
        const textHeight = 16
        const padding = 4

        const labelY = y > textHeight + padding ? y - 2 : y + textHeight + padding

        // Draw label background
        ctx.fillStyle = color
        ctx.fillRect(x, labelY - textHeight, textWidth + padding * 2, textHeight)

        // Draw label text
        ctx.fillStyle = '#fff'
        ctx.fillText(label, x + padding, labelY - 4)
      })

      ctx.globalAlpha = 1
    }

    // Draw when image loads or when dependencies change
    const handleImageLoad = () => draw()
    img.addEventListener('load', handleImageLoad)

    // Draw immediately if image is already loaded (cached)
    if (img.complete && img.naturalWidth > 0) {
      draw()
    }

    const handleResize = () => draw()
    window.addEventListener('resize', handleResize)

    return () => {
      img.removeEventListener('load', handleImageLoad)
      window.removeEventListener('resize', handleResize)
    }
  }, [imageUrl, detections, color, opacity])

  return (
    <div className="relative w-full h-full">
      <img
        ref={imgRef}
        src={imageUrl}
        alt="Experiment"
        className="w-full h-auto block"
      />
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 pointer-events-none"
      />
    </div>
  )
}
