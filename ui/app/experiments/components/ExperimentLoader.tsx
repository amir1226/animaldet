import { useState } from 'react'

interface ExperimentLoaderProps {
  onLoad: (url: string, name: string) => void
  loading: boolean
  experimentNumber: number
}

export default function ExperimentLoader({
  onLoad,
  loading,
  experimentNumber
}: ExperimentLoaderProps) {
  const [url, setUrl] = useState('')
  const [name, setName] = useState(`Experiment ${experimentNumber}`)

  const handleLoad = () => {
    if (url.trim()) {
      onLoad(url.trim(), name.trim())
    }
  }

  return (
    <div className="border border-gray-300 p-4 space-y-3">
      <h3 className="font-semibold text-sm text-gray-700">
        Experiment {experimentNumber}
      </h3>

      <div>
        <label className="block text-xs text-gray-600 mb-1">Name</label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder={`Experiment ${experimentNumber}`}
          className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
        />
      </div>

      <div>
        <label className="block text-xs text-gray-600 mb-1">CSV URL</label>
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://example.com/detections.csv"
          className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
        />
      </div>

      <button
        onClick={handleLoad}
        disabled={loading || !url.trim()}
        className="w-full px-4 py-2 bg-black text-white rounded text-sm hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? 'Loading...' : 'Load Experiment'}
      </button>
    </div>
  )
}
