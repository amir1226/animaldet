/**
 * Creates a placeholder image data URL when the actual image fails to load
 * @param width - Width of the placeholder image
 * @param height - Height of the placeholder image
 * @param backgroundColor - Background color (default: #e5e7eb - gray-200)
 * @returns Data URL for the placeholder image
 */
export function createPlaceholderImage(
  width: number = 800,
  height: number = 600,
  backgroundColor: string = '#e5e7eb'
): string {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height

  const ctx = canvas.getContext('2d')
  if (!ctx) return ''

  // Fill background
  ctx.fillStyle = backgroundColor
  ctx.fillRect(0, 0, width, height)

  // Add border
  ctx.strokeStyle = '#9ca3af' // gray-400
  ctx.lineWidth = 2
  ctx.strokeRect(0, 0, width, height)

  // Add text
  ctx.fillStyle = '#6b7280' // gray-500
  ctx.font = 'bold 24px sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText('Image Not Found', width / 2, height / 2 - 20)

  ctx.font = '16px sans-serif'
  ctx.fillStyle = '#9ca3af' // gray-400
  ctx.fillText(`${width} × ${height}`, width / 2, height / 2 + 20)

  return canvas.toDataURL('image/png')
}

/**
 * Gets image dimensions from an image element or returns defaults
 * @param img - Image element to get dimensions from
 * @returns Object with width and height
 */
export function getImageDimensions(img: HTMLImageElement | null): { width: number; height: number } {
  if (!img) return { width: 800, height: 600 }

  // Try to get natural dimensions first (original image size)
  if (img.naturalWidth > 0 && img.naturalHeight > 0) {
    return { width: img.naturalWidth, height: img.naturalHeight }
  }

  // Fall back to display dimensions
  if (img.offsetWidth > 0 && img.offsetHeight > 0) {
    return { width: img.offsetWidth, height: img.offsetHeight }
  }

  // Default fallback
  return { width: 800, height: 600 }
}
