import { useState, useEffect, useMemo } from 'react'
import { Disclosure, DisclosureButton, DisclosurePanel } from '@headlessui/react'
import { experimentsService, ExperimentData } from '@animaldet/shared/api/experiments'
import BoundingBoxCanvas from '../components/BoundingBoxCanvas'
import ImageThumbnailList from '../components/ImageThumbnailList'
import { createPlaceholderImage } from '@animaldet/shared/utils/placeholderImage'

const CLASS_NAMES: Record<number, string> = {
  1: 'Topi',
  2: 'Buffalo',
  3: 'Kob',
  4: 'Warthog',
  5: 'Waterbuck',
  6: 'Elephant',
}

interface PrebuiltExperiment {
  name: string
  csvUrl: string
  imageBaseUrl: string
  description: string
}

const PREBUILT_EXPERIMENTS: PrebuiltExperiment[] = [
  {
    name: 'RF-DETR Phase 2',
    csvUrl: '/experiments/rfdetr_detections_phase2.csv',
    imageBaseUrl: '/demo_images',
    description: 'RF-DETR model detections on phase 2 test set'
  },
  {
    name: 'Ground Truth',
    csvUrl: '/experiments/test_big_size_A_B_E_K_WH_WB.csv',
    imageBaseUrl: '/demo_images',
    description: 'Ground truth annotations'
  }
]

export default function ExperimentComparison() {
  const [experiment1, setExperiment1] = useState<ExperimentData | null>(null)
  const [experiment2, setExperiment2] = useState<ExperimentData | null>(null)
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [loading1, setLoading1] = useState(false)
  const [loading2, setLoading2] = useState(false)
  const [imageBaseUrl, setImageBaseUrl] = useState('')
  const [showExp1, setShowExp1] = useState(true)
  const [showExp2, setShowExp2] = useState(true)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [selectedClass, setSelectedClass] = useState<number | null>(null)

  // Filter images by selected class
  const filteredImages = useMemo(() => {
    const allImages = new Set([
      ...(experiment1?.images || []),
      ...(experiment2?.images || [])
    ])

    if (!selectedClass) {
      return allImages
    }

    // Find images that contain detections of the selected class
    const imagesWithClass = new Set<string>()

    experiment1?.detections.forEach(det => {
      if (det.label === selectedClass) {
        imagesWithClass.add(det.image)
      }
    })

    experiment2?.detections.forEach(det => {
      if (det.label === selectedClass) {
        imagesWithClass.add(det.image)
      }
    })

    return imagesWithClass
  }, [experiment1, experiment2, selectedClass])

  const availableImages = filteredImages

  const getImageUrl = (imageName: string) => {
    if (!imageBaseUrl) return imageName
    return `${imageBaseUrl.replace(/\/$/, '')}/${imageName}`
  }

  const exp1Detections = selectedImage && experiment1
    ? experimentsService.getDetectionsForImage(experiment1.detections, selectedImage)
    : []

  const exp2Detections = selectedImage && experiment2
    ? experimentsService.getDetectionsForImage(experiment2.detections, selectedImage)
    : []

  const loadPrebuiltExperiment = async (preset: PrebuiltExperiment, experimentNumber: 1 | 2) => {
    const setLoading = experimentNumber === 1 ? setLoading1 : setLoading2
    const setExperiment = experimentNumber === 1 ? setExperiment1 : setExperiment2

    setLoading(true)
    try {
      const data = await experimentsService.loadExperimentFromURL(preset.csvUrl, preset.name)
      setExperiment(data)
      setImageBaseUrl(preset.imageBaseUrl)

      // Auto-select first image if available
      if (data.images.length > 0 && !selectedImage) {
        setSelectedImage(data.images[0])
      }
    } catch (error) {
      alert(`Failed to load experiment: ${error}`)
    } finally {
      setLoading(false)
    }
  }

  // Load default experiments on mount
  useEffect(() => {
    const loadDefaults = async () => {
      setLoading1(true)
      setLoading2(true)

      try {
        // Load RF-DETR Phase 2 as Experiment 1
        const exp1Data = await experimentsService.loadExperimentFromURL(
          PREBUILT_EXPERIMENTS[0].csvUrl,
          PREBUILT_EXPERIMENTS[0].name
        )
        setExperiment1(exp1Data)
        setImageBaseUrl(PREBUILT_EXPERIMENTS[0].imageBaseUrl)

        // Load HerdNet Ground Truth as Experiment 2
        const exp2Data = await experimentsService.loadExperimentFromURL(
          PREBUILT_EXPERIMENTS[1].csvUrl,
          PREBUILT_EXPERIMENTS[1].name
        )
        setExperiment2(exp2Data)

        // Auto-select first image
        if (exp1Data.images.length > 0) {
          setSelectedImage(exp1Data.images[0])
        }
      } catch (error) {
        alert(`Failed to load default experiments: ${error}`)
      } finally {
        setLoading1(false)
        setLoading2(false)
      }
    }

    loadDefaults()
  }, [])

  // Auto-select first image when filtered images change (e.g., when class filter changes)
  useEffect(() => {
    const images = Array.from(availableImages)
    if (images.length > 0) {
      // If current selected image is not in filtered list, select the first one
      if (!selectedImage || !availableImages.has(selectedImage)) {
        setSelectedImage(images[0])
      }
    } else {
      // No images available, clear selection
      setSelectedImage(null)
    }
  }, [availableImages])

  const hasExperiments = experiment1 || experiment2

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Main content area */}
      <div className="flex-1 overflow-y-auto">
        <div className={`h-full transition-all duration-300 p-6`}>
          <header className="pb-0 mb-8">
            <h1 className="text-3xl font-bold text-black mb-1">
              Results Comparison
            </h1>
            <p className="text-gray-600">Compare detection results across experiments</p>
          </header>

      {/* Class Filter */}
      {(experiment1 || experiment2) && (
        <div className="mb-6">
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Filter by Animal Class
          </label>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setSelectedClass(null)}
              className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
                selectedClass === null
                  ? 'bg-black text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              All Classes
            </button>
            {Object.entries(CLASS_NAMES).map(([classId, className]) => (
              <button
                key={classId}
                onClick={() => setSelectedClass(Number(classId))}
                className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
                  selectedClass === Number(classId)
                    ? 'bg-black text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {className}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Experiment Info */}
      {(experiment1 || experiment2) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {experiment1 && (
            <div className="border border-gray-300 p-4 bg-blue-50">
              <h3 className="font-semibold text-sm mb-2">{experiment1.name}</h3>
              <div className="text-xs text-gray-600 space-y-1">
                <div>Images: {experiment1.images.length}</div>
                <div>Total Detections: {experiment1.detections.length}</div>
                {selectedImage && (
                  <div className="text-blue-600 font-semibold">
                    Current Image Detections: {exp1Detections.length}
                  </div>
                )}
              </div>
            </div>
          )}
          {experiment2 && (
            <div className="border border-gray-300 p-4 bg-green-50">
              <h3 className="font-semibold text-sm mb-2">{experiment2.name}</h3>
              <div className="text-xs text-gray-600 space-y-1">
                <div>Images: {experiment2.images.length}</div>
                <div>Total Detections: {experiment2.detections.length}</div>
                {selectedImage && (
                  <div className="text-green-600 font-semibold">
                    Current Image Detections: {exp2Detections.length}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Image Selector - Fallback for mobile/narrow screens */}
      {availableImages.size > 0 && (
        <div className="mb-6 lg:hidden">
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Select Image ({availableImages.size} available)
          </label>
          <select
            value={selectedImage || ''}
            onChange={(e) => setSelectedImage(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded"
          >
            <option value="">-- Select an image --</option>
            {Array.from(availableImages).sort().map((image) => (
              <option key={image} value={image}>
                {image}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Visualization Controls */}
      {selectedImage && (experiment1 || experiment2) && (
        <div className="mb-4">
          <Disclosure defaultOpen>
            <DisclosureButton className="w-full text-left py-2 px-4 bg-gray-100 hover:bg-gray-200 rounded font-semibold text-gray-800 text-sm">
              Show/Hide Detections
            </DisclosureButton>
            <DisclosurePanel className="pt-4 flex gap-4 items-center pl-4">
              {experiment1 && (
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showExp1}
                    onChange={(e) => setShowExp1(e.target.checked)}
                    className="w-4 h-4"
                  />
                  <span className="text-sm flex items-center gap-2">
                    <span className="w-4 h-4 rounded" style={{ backgroundColor: '#3B82F6' }}></span>
                    {experiment1.name}
                  </span>
                </label>
              )}
              {experiment2 && (
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showExp2}
                    onChange={(e) => setShowExp2(e.target.checked)}
                    className="w-4 h-4"
                  />
                  <span className="text-sm flex items-center gap-2">
                    <span className="w-4 h-4 rounded" style={{ backgroundColor: '#10B981' }}></span>
                    {experiment2.name}
                  </span>
                </label>
              )}
            </DisclosurePanel>
          </Disclosure>
        </div>
      )}

      {/* Visualization */}
      {selectedImage && (
        <div className="w-full">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-0 w-full">
          {/* Experiment 1 */}
          {experiment1 && (
            <div className="border-2 border-blue-300 relative w-full">
              <div className="bg-blue-600 text-white px-3 py-2 text-sm font-semibold">
                {experiment1.name} ({exp1Detections.length} detections)
              </div>
              <div className="relative w-full">
                {showExp1 && (
                  <div className="absolute top-0 left-0 w-full h-full z-10">
                    <BoundingBoxCanvas
                      imageUrl={getImageUrl(selectedImage)}
                      detections={exp1Detections}
                      color="#3B82F6"
                      opacity={1}
                    />
                  </div>
                )}
                <img
                  src={getImageUrl(selectedImage)}
                  alt={selectedImage}
                  className="w-full h-auto block object-contain"
                  onError={(e) => {
                    const target = e.target as HTMLImageElement
                    target.src = createPlaceholderImage()
                  }}
                />
              </div>
            </div>
          )}

          {/* Experiment 2 */}
          {experiment2 && (
            <div className="border-2 border-green-300 relative w-full">
              <div className="bg-green-600 text-white px-3 py-2 text-sm font-semibold">
                {experiment2.name} ({exp2Detections.length} detections)
              </div>
              <div className="relative w-full">
                {showExp2 && (
                  <div className="absolute top-0 left-0 w-full h-full z-10">
                    <BoundingBoxCanvas
                      imageUrl={getImageUrl(selectedImage)}
                      detections={exp2Detections}
                      color="#10B981"
                      opacity={1}
                    />
                  </div>
                )}
                <img
                  src={getImageUrl(selectedImage)}
                  alt={selectedImage}
                  className="w-full h-auto block object-contain"
                  onError={(e) => {
                    const target = e.target as HTMLImageElement
                    target.src = createPlaceholderImage()
                  }}
                />
              </div>
            </div>
          )}
        </div>
        </div>
      )}

      {/* No data message */}
      {!experiment1 && !experiment2 && (
        <div className="text-center py-12 text-gray-500">
          Load at least one experiment to begin comparison
        </div>
      )}
        </div>
      </div>


      {/* Right Sidebar - Image List */}
      {hasExperiments && availableImages.size > 0 && (
        <div className="w-80 flex-shrink-0 bg-white border-l border-gray-300 shadow-lg overflow-y-auto">
          <div className="h-full p-4">
            <ImageThumbnailList
              images={Array.from(availableImages).sort()}
              selectedImage={selectedImage}
              onSelectImage={setSelectedImage}
              getImageUrl={getImageUrl}
            />
          </div>
        </div>
      )}
    </div>
  )
}
