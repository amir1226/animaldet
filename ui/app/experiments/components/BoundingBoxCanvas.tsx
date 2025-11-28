import { useEffect, useRef, useState, useMemo } from 'react'
import { ExperimentDetection } from '@animaldet/shared/api/experiments'
import { getClassName } from '@animaldet/shared/utils/classNames'
import { createPlaceholderImage } from '@animaldet/shared/utils/placeholderImage'

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
  const [imageLoadFailed, setImageLoadFailed] = useState(false)
  const [originalDimensions, setOriginalDimensions] = useState<{ width: number; height: number } | null>(null)

  // Reset state when imageUrl changes
  useEffect(() => {
    setImageLoadFailed(false)
    setOriginalDimensions(null)
  }, [imageUrl])

  // Calculate image dimensions from detection coordinates if needed
  const estimatedDimensions = useMemo(() => {
    if (detections.length === 0) {
      return { width: 3264, height: 2448 } // Default camera resolution
    }

    let maxX = 0
    let maxY = 0

    detections.forEach((det) => {
      maxX = Math.max(maxX, det.x_max)
      maxY = Math.max(maxY, det.y_max)
    })

    // Add some padding (10%) to ensure all detections fit
    return {
      width: Math.ceil(maxX * 1.1),
      height: Math.ceil(maxY * 1.1)
    }
  }, [detections])

  useEffect(() => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img) return

    const draw = () => {
      // Determine the original image dimensions
      let naturalWidth: number
      let naturalHeight: number

      if (imageLoadFailed && originalDimensions) {
        // Use stored dimensions from before failure
        naturalWidth = originalDimensions.width
        naturalHeight = originalDimensions.height
      } else if (img.naturalWidth > 0 && img.naturalHeight > 0) {
        // Image loaded successfully
        naturalWidth = img.naturalWidth
        naturalHeight = img.naturalHeight
      } else if (imageLoadFailed) {
        // Image failed and no stored dimensions - use estimated dimensions
        naturalWidth = estimatedDimensions.width
        naturalHeight = estimatedDimensions.height
      } else {
        // Image not ready yet
        return
      }

      const displayWidth = img.offsetWidth || 800
      const displayHeight = img.offsetHeight || 600

      canvas.width = displayWidth
      canvas.height = displayHeight
      canvas.style.width = `${displayWidth}px`
      canvas.style.height = `${displayHeight}px`

      const scaleX = displayWidth / naturalWidth
      const scaleY = displayHeight / naturalHeight

      const ctx = canvas.getContext('2d')
      if (!ctx) return

      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.globalAlpha = opacity

      // Shared color palette for consistent class colors across all views
      const classColors = ['#D97706', '#3B82F6', '#10B981', '#EF4444', '#8B5CF6', '#F59E0B']

      detections.forEach((det) => {
        const x = det.x * scaleX
        const y = det.y * scaleY
        const width = (det.x_max - det.x) * scaleX
        const height = (det.y_max - det.y) * scaleY

        // Use class-based color instead of experiment color
        const detectionColor = classColors[det.label % classColors.length]

        // Draw bounding box
        ctx.strokeStyle = detectionColor
        ctx.lineWidth = 2
        ctx.strokeRect(x, y, width, height)

        // Draw label
        const label = getClassName(det.label)
        ctx.font = 'bold 11px sans-serif'
        const textMetrics = ctx.measureText(label)
        const textWidth = textMetrics.width
        const textHeight = 16
        const padding = 4

        const labelY = y > textHeight + padding ? y - 2 : y + textHeight + padding

        // Draw label background
        ctx.fillStyle = detectionColor
        ctx.fillRect(x, labelY - textHeight, textWidth + padding * 2, textHeight)

        // Draw label text
        ctx.fillStyle = '#fff'
        ctx.fillText(label, x + padding, labelY - 4)
      })

      ctx.globalAlpha = 1
    }

    // Draw when image loads successfully
    const handleImageLoad = () => {
      // Store dimensions before any potential failure
      if (imgRef.current && imgRef.current.naturalWidth > 0) {
        setOriginalDimensions({
          width: imgRef.current.naturalWidth,
          height: imgRef.current.naturalHeight
        })
      }
      setImageLoadFailed(false)
      requestAnimationFrame(() => {
        draw()
      })
    }

    // Draw when image fails to load
    const handleImageError = () => {
      setImageLoadFailed(true)
      // Wait for placeholder to load
      setTimeout(() => {
        draw()
      }, 100)
    }

    img.addEventListener('load', handleImageLoad)
    img.addEventListener('error', handleImageError)

    // Draw immediately if image is already loaded
    if (img.complete) {
      if (img.naturalWidth > 0) {
        handleImageLoad()
      } else {
        handleImageError()
      }
    }

    const handleResize = () => draw()
    window.addEventListener('resize', handleResize)

    return () => {
      img.removeEventListener('load', handleImageLoad)
      img.removeEventListener('error', handleImageError)
      window.removeEventListener('resize', handleResize)
    }
  }, [imageUrl, detections, color, opacity, estimatedDimensions])

  // Generate a fixed-size placeholder to prevent layout explosion
  // We use estimatedDimensions for detection scaling, not for the placeholder itself
  const placeholderSrc = useMemo(() => {
    const aspectRatio = estimatedDimensions.width / estimatedDimensions.height

    // Fixed max size to prevent layout issues
    const maxSize = 800
    let width = maxSize
    let height = maxSize / aspectRatio

    if (height > maxSize) {
      height = maxSize
      width = maxSize * aspectRatio
    }

    return createPlaceholderImage(Math.round(width), Math.round(height))
  }, [estimatedDimensions])

  return (
    <div className="relative w-full h-full">
      <img
        key={imageUrl}
        ref={imgRef}
        src={imageUrl}
        alt="Experiment"
        className="w-full h-auto block"
        onError={(e) => {
          const target = e.target as HTMLImageElement
          target.src = placeholderSrc
        }}
      />
      <canvas
        key={`${imageUrl}-canvas`}
        ref={canvasRef}
        className="absolute top-0 left-0 pointer-events-none"
      />
    </div>
  )
}
