import { useState } from 'react'
import { experimentsService, ExperimentData } from '@animaldet/shared/api/experiments'
import ExperimentLoader from '../components/ExperimentLoader'
import BoundingBoxCanvas from '../components/BoundingBoxCanvas'

interface PrebuiltExperiment {
  name: string
  csvUrl: string
  imageBaseUrl: string
  description: string
}

const PREBUILT_EXPERIMENTS: PrebuiltExperiment[] = [
  {
    name: 'RF-DETR Phase 2',
    csvUrl: 'http://0.0.0.0:8000/outputs/inference/rfdetr_detections_phase2.csv',
    imageBaseUrl: 'http://0.0.0.0:8000/data/herdnet/raw/test',
    description: 'RF-DETR model detections on phase 2 test set'
  },
  {
    name: 'HerdNet Ground Truth',
    csvUrl: 'http://0.0.0.0:8000/data/herdnet/raw/groundtruth/csv/test_big_size_A_B_E_K_WH_WB.csv',
    imageBaseUrl: 'http://0.0.0.0:8000/data/herdnet/raw/test',
    description: 'Ground truth annotations for HerdNet test set'
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

  return (
    <div className="max-w-7xl mx-auto">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-black mb-1">
          Experiment Comparison
        </h1>
        <p className="text-gray-600">Compare detection results across experiments</p>
      </header>

      {/* Prebuilt Experiments */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold text-gray-800 mb-4">Quick Start Experiments</h2>
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
      <div className="mb-6">
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
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
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

      {/* Image Selector */}
      {availableImages.size > 0 && (
        <div className="mb-6">
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
        <div className="mb-4 flex gap-4 items-center">
          <span className="text-sm font-semibold text-gray-700">Show:</span>
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
        </div>
      )}

      {/* Visualization */}
      {selectedImage && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Experiment 1 */}
          {experiment1 && (
            <div className="border-2 border-blue-300 relative">
              <div className="bg-blue-600 text-white px-3 py-2 text-sm font-semibold">
                {experiment1.name} ({exp1Detections.length} detections)
              </div>
              <div className="relative">
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
                  className="w-full h-auto block"
                  onError={() => alert('Failed to load image. Check the image base URL.')}
                />
              </div>
            </div>
          )}

          {/* Experiment 2 */}
          {experiment2 && (
            <div className="border-2 border-green-300 relative">
              <div className="bg-green-600 text-white px-3 py-2 text-sm font-semibold">
                {experiment2.name} ({exp2Detections.length} detections)
              </div>
              <div className="relative">
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
                  className="w-full h-auto block"
                  onError={() => alert('Failed to load image. Check the image base URL.')}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* No data message */}
      {!experiment1 && !experiment2 && (
        <div className="text-center py-12 text-gray-500">
          Load at least one experiment to begin comparison
        </div>
      )}
    </div>
  )
}
