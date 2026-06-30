import { useState } from 'react'

export default function URLInput({ onSubmit, loading }) {
  const [url, setUrl] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (url.trim()) {
      onSubmit(url.trim())
    }
  }

  const isValidUrl = url.includes('youtube.com/watch') || url.includes('youtu.be/')

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-2xl mx-auto">
      <div className="flex gap-3">
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Paste a YouTube URL..."
          className="flex-1 px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg
                     text-white placeholder-gray-500 focus:outline-none focus:border-indigo-500
                     focus:ring-1 focus:ring-indigo-500 transition-colors"
          disabled={loading}
        />
        <button
          type="submit"
          disabled={!isValidUrl || loading}
          className="px-6 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700
                     disabled:text-gray-500 text-white font-medium rounded-lg
                     transition-colors whitespace-nowrap"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Processing
            </span>
          ) : 'Generate'}
        </button>
      </div>
      {url && !isValidUrl && (
        <p className="mt-2 text-sm text-gray-500">Enter a valid YouTube URL</p>
      )}
    </form>
  )
}
