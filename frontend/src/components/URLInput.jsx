import { useState } from 'react'

const MODES = [
  { id: 'youtube', label: 'YouTube URL' },
  { id: 'local', label: 'Local file' },
]

export default function URLInput({ onSubmit, loading }) {
  const [mode, setMode] = useState('youtube')
  const [value, setValue] = useState('')

  const isYouTube = value.includes('youtube.com/watch') || value.includes('youtu.be/')
  const isLocalPath = /\.(mp4|mov|mkv|webm|m4v|avi)$/i.test(value.trim())
  const isValid = mode === 'youtube' ? isYouTube : isLocalPath

  const handleSubmit = (e) => {
    e.preventDefault()
    if (value.trim() && isValid) {
      onSubmit(value.trim(), mode)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-2xl mx-auto space-y-3">
      {/* Source toggle */}
      <div className="flex gap-2 justify-center">
        {MODES.map((m) => (
          <button
            key={m.id}
            type="button"
            onClick={() => setMode(m.id)}
            disabled={loading}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              mode === m.id
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:text-white'
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="flex gap-3">
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={mode === 'youtube' ? 'Paste a YouTube URL...' : '/path/to/your-export.mp4'}
          className="flex-1 px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg
                     text-white placeholder-gray-500 focus:outline-none focus:border-indigo-500
                     focus:ring-1 focus:ring-indigo-500 transition-colors"
          disabled={loading}
        />
        <button
          type="submit"
          disabled={!isValid || loading}
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

      {value && !isValid && (
        <p className="mt-2 text-sm text-gray-500">
          {mode === 'youtube'
            ? 'Enter a valid YouTube URL'
            : 'Enter a path to a video file (.mp4, .mov, .mkv, .webm, .m4v, .avi)'}
        </p>
      )}
    </form>
  )
}
