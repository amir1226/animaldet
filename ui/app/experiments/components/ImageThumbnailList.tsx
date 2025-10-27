import { useState, useRef } from 'react'

interface ImageThumbnailListProps {
  images: string[]
  selectedImage: string | null
  onSelectImage: (image: string) => void
  getImageUrl: (imageName: string) => string
}

const ITEM_HEIGHT = 100 // Height of each thumbnail item
const OVERSCAN = 3 // Number of extra items to render above and below viewport

export default function ImageThumbnailList({
  images,
  selectedImage,
  onSelectImage,
  getImageUrl
}: ImageThumbnailListProps) {
  const [scrollTop, setScrollTop] = useState(0)
  const [containerHeight, setContainerHeight] = useState(0)
  const containerRef = useRef<HTMLDivElement>(null)

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop)
    if (!containerHeight && containerRef.current) {
      setContainerHeight(containerRef.current.clientHeight)
    }
  }

  // Calculate visible range
  const effectiveHeight = containerHeight || 600
  const startIndex = Math.max(0, Math.floor(scrollTop / ITEM_HEIGHT) - OVERSCAN)
  const endIndex = Math.min(
    images.length,
    Math.ceil((scrollTop + effectiveHeight) / ITEM_HEIGHT) + OVERSCAN
  )

  const visibleImages = images.slice(startIndex, endIndex)
  const totalHeight = images.length * ITEM_HEIGHT
  const offsetY = startIndex * ITEM_HEIGHT

  return (
    <div className="flex flex-col h-full">
      <div className="text-sm font-semibold text-gray-700 mb-2 px-1">
        Images ({images.length})
      </div>
      <div
        ref={containerRef}
        className="flex-1 border border-gray-300 rounded overflow-y-auto bg-gray-50"
        onScroll={handleScroll}
      >
        <div style={{ height: `${totalHeight}px`, position: 'relative' }}>
          <div style={{ transform: `translateY(${offsetY}px)` }}>
            {visibleImages.map((image, idx) => {
              const actualIndex = startIndex + idx
              const isSelected = image === selectedImage

              return (
                <div
                  key={`${image}-${actualIndex}`}
                  className={`flex items-center gap-3 p-2 cursor-pointer transition-colors border-b border-gray-200 ${
                    isSelected
                      ? 'bg-blue-100 hover:bg-blue-200'
                      : 'hover:bg-gray-100'
                  }`}
                  style={{ height: `${ITEM_HEIGHT}px` }}
                  onClick={() => onSelectImage(image)}
                >
                  <div className="flex-shrink-0 w-20 h-20 bg-gray-200 rounded overflow-hidden">
                    <img
                      src={getImageUrl(image)}
                      alt={image}
                      className="w-full h-full object-cover"
                      loading="lazy"
                      onError={(e) => {
                        const target = e.target as HTMLImageElement
                        target.style.display = 'none'
                        const parent = target.parentElement
                        if (parent) {
                          parent.classList.add('flex', 'items-center', 'justify-center')
                          parent.innerHTML = '<span class="text-xs text-gray-500">Error</span>'
                        }
                      }}
                    />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className={`text-sm truncate ${isSelected ? 'font-semibold text-blue-700' : 'text-gray-700'}`}>
                      {image}
                    </div>
                  </div>
                  {isSelected && (
                    <div className="flex-shrink-0">
                      <svg className="w-5 h-5 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
