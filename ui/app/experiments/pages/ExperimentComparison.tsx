import { useState } from 'react'
import { Disclosure, DisclosureButton, DisclosurePanel } from '@headlessui/react'
import { experimentsService, ExperimentData } from '@animaldet/shared/api/experiments'
import ExperimentLoader from '../components/ExperimentLoader'
import BoundingBoxCanvas from '../components/BoundingBoxCanvas'
import ImageThumbnailList from '../components/ImageThumbnailList'

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

  const handleLoadExperiment1 = async (url: string, name: string) => {
    setLoading1(true)
    try {
      const data = await experimentsService.loadExperimentFromURL(url, name)
      setExperiment1(data)

      // Auto-select first image if available
      if (data.images.length > 0 && !selectedImage) {
        setSelectedImage(data.images[0])
      }
    } catch (error) {
      alert(`Failed to load experiment: ${error}`)
    } finally {
      setLoading1(false)
    }
  }

  const handleLoadExperiment2 = async (url: string, name: string) => {
    setLoading2(true)
    try {
      const data = await experimentsService.loadExperimentFromURL(url, name)
      setExperiment2(data)

      // Auto-select first image if available
      if (data.images.length > 0 && !selectedImage) {
        setSelectedImage(data.images[0])
      }
    } catch (error) {
      alert(`Failed to load experiment: ${error}`)
    } finally {
      setLoading2(false)
    }
  }

  const availableImages = new Set([
    ...(experiment1?.images || []),
    ...(experiment2?.images || [])
  ])

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

  const hasExperiments = experiment1 || experiment2

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Main content area */}
      <div className="flex-1 overflow-y-auto">
        <div className={`h-full transition-all duration-300`}>
          <header className="p-6 pb-0 mb-8">
            <h1 className="text-3xl font-bold text-black mb-1">
              Experiment Comparison
            </h1>
            <p className="text-gray-600">Compare detection results across experiments</p>
          </header>

      {/* Quick Start Experiments */}
      <div className="mb-8 px-6">
        <Disclosure defaultOpen>
          <DisclosureButton className="w-full text-left py-2 px-4 bg-gray-100 hover:bg-gray-200 rounded font-semibold text-gray-800">
            Quick Start Experiments
          </DisclosureButton>
          <DisclosurePanel className="pt-4 space-y-6">
            {/* Prebuilt Experiments */}
            <div>
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Prebuilt Experiments</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {PREBUILT_EXPERIMENTS.map((preset) => (
                  <div
                    key={preset.name}
                    className="border border-gray-300 rounded p-4 bg-gray-50 hover:bg-gray-100 transition-colors"
                  >
                    <h3 className="font-semibold text-gray-900 mb-2">{preset.name}</h3>
                    <p className="text-sm text-gray-600 mb-3">{preset.description}</p>
                    <div className="flex gap-2">
                      <button
                        onClick={() => loadPrebuiltExperiment(preset, 1)}
                        disabled={loading1}
                        className="flex-1 px-3 py-2 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Load as Exp 1
                      </button>
                      <button
                        onClick={() => loadPrebuiltExperiment(preset, 2)}
                        disabled={loading2}
                        className="flex-1 px-3 py-2 bg-green-600 text-white rounded text-sm hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Load as Exp 2
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Image Base URL */}
            <div>
              <label className="block text-sm text-gray-600 mb-2">
                Image Base URL (optional - if images are hosted remotely)
              </label>
              <input
                type="text"
                value={imageBaseUrl}
                onChange={(e) => setImageBaseUrl(e.target.value)}
                placeholder="https://example.com/images"
                className="w-full px-4 py-2 border border-gray-300 rounded"
              />
            </div>

            {/* Experiment Loaders */}
            <div>
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Load Custom Experiments from URL</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <ExperimentLoader
                  onLoad={handleLoadExperiment1}
                  loading={loading1}
                  experimentNumber={1}
                />
                <ExperimentLoader
                  onLoad={handleLoadExperiment2}
                  loading={loading2}
                  experimentNumber={2}
                />
              </div>
            </div>
          </DisclosurePanel>
        </Disclosure>
      </div>

      {/* Experiment Info */}
      {(experiment1 || experiment2) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6 px-6">
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
        <div className="mb-6 lg:hidden px-6">
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
        <div className="mb-4 px-6">
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
                  onError={() => alert('Failed to load image. Check the image base URL.')}
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
                  onError={() => alert('Failed to load image. Check the image base URL.')}
                />
              </div>
            </div>
          )}
        </div>
        </div>
      )}

      {/* No data message */}
      {!experiment1 && !experiment2 && (
        <div className="text-center py-12 text-gray-500 px-6">
          Load at least one experiment to begin comparison
        </div>
      )}
        </div>
      </div>

      {/* Sidebar Toggle Button */}
      {hasExperiments && availableImages.size > 0 && (
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className={`fixed top-4 z-50 p-2 bg-gray-800 text-white rounded hover:bg-gray-700 transition-all duration-300 ${sidebarOpen ? 'right-[21rem]' : 'right-4'}`}
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            {sidebarOpen ? (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            ) : (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            )}
          </svg>
        </button>
      )}

      {/* Right Sidebar - Image List */}
      {hasExperiments && availableImages.size > 0 && (
        <div className={`hidden lg:block fixed right-0 top-0 bottom-0 w-80 bg-white border-l border-gray-300 shadow-lg overflow-hidden transition-transform duration-300 ${sidebarOpen ? 'translate-x-0' : 'translate-x-full'}`}>
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
