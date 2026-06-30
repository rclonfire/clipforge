import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useJobStatus } from '../hooks/useJobStatus'
import { useApi } from '../hooks/useApi'
import JobStatusBadge from '../components/JobStatusBadge'
import ThumbnailGallery from '../components/ThumbnailGallery'
import ClipCard from '../components/ClipCard'
import ExportPanel from '../components/ExportPanel'
import { formatDuration } from '../utils/formatters'

const STAGES = [
  { key: 'downloading', label: 'Downloading' },
  { key: 'transcribing', label: 'Transcribing' },
  { key: 'extracting_frames', label: 'Extracting frames' },
  { key: 'generating_thumbnails', label: 'Generating thumbnails' },
  { key: 'analyzing', label: 'Analyzing audio' },
  { key: 'detecting_clips', label: 'Detecting clips' },
  { key: 'complete', label: 'Complete' },
]

function ProgressBar({ status }) {
  const currentIdx = STAGES.findIndex((s) => s.key === status)
  return (
    <div className="flex items-center gap-1 w-full max-w-md">
      {STAGES.map((stage, i) => (
        <div key={stage.key} className="flex-1">
          <div
            className={`h-1.5 rounded-full transition-colors ${
              i <= currentIdx
                ? status === 'failed' ? 'bg-red-500' : 'bg-indigo-500'
                : 'bg-gray-800'
            }`}
          />
          <span className={`text-xs mt-1 block text-center ${
            i <= currentIdx ? 'text-gray-400' : 'text-gray-700'
          }`}>
            {stage.label}
          </span>
        </div>
      ))}
    </div>
  )
}

export default function JobDetail() {
  const { id } = useParams()
  const { job, progress } = useJobStatus(id)
  const [tab, setTab] = useState('thumbnails')
  const [thumbnails, setThumbnails] = useState([])
  const [clips, setClips] = useState([])
  const [keptIds, setKeptIds] = useState(new Set())
  const api = useApi()

  const handleKeep = async (clipId, keep) => {
    try {
      await api.updateClipKept(id, clipId, keep)
      setKeptIds(prev => {
        const next = new Set(prev)
        keep ? next.add(clipId) : next.delete(clipId)
        return next
      })
    } catch (err) {
      console.error('Failed to update kept state:', err)
    }
  }

  // Fetch results when job completes
  useEffect(() => {
    if (job?.status === 'complete') {
      api.getThumbnails(id).then(setThumbnails).catch(console.error)
      api.getClips(id).then(data => {
        setClips(data)
        // Initialize kept state from server
        const kept = new Set(data.filter(c => c.kept).map(c => c.id))
        setKeptIds(kept)
      }).catch(console.error)
    }
  }, [job?.status, id])

  if (!job) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="animate-spin h-8 w-8 border-2 border-indigo-500 border-t-transparent rounded-full" />
      </div>
    )
  }

  const isProcessing = !['complete', 'failed'].includes(job.status)

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="space-y-4">
        <Link to="/" className="text-sm text-gray-500 hover:text-gray-400 transition-colors">
          &larr; Back
        </Link>
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1 min-w-0">
            <h1 className="text-2xl font-bold text-white truncate">
              {job.video_title || 'Processing...'}
            </h1>
            <div className="flex items-center gap-3 text-sm text-gray-500">
              <span className="truncate max-w-xs">{job.youtube_url}</span>
              {job.video_duration_seconds && (
                <span>{formatDuration(job.video_duration_seconds)}</span>
              )}
            </div>
          </div>
          <JobStatusBadge status={job.status} />
        </div>
      </div>

      {/* Progress */}
      {isProcessing && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 space-y-4">
          <ProgressBar status={job.status} />
          <p className="text-sm text-gray-400 animate-pulse">
            {progress.message || job.progress_message || 'Processing...'}
          </p>
        </div>
      )}

      {/* Error */}
      {job.status === 'failed' && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 space-y-3">
          <p className="text-sm text-red-400">{job.error_message || 'An error occurred'}</p>
          <Link
            to="/"
            className="inline-block px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white text-sm font-medium rounded-lg transition-colors"
          >
            Try Again
          </Link>
        </div>
      )}

      {/* Results */}
      {job.status === 'complete' && (
        <>
          {/* Tabs */}
          <div className="flex gap-1 bg-gray-900 rounded-lg p-1">
            <button
              onClick={() => setTab('thumbnails')}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
                tab === 'thumbnails'
                  ? 'bg-gray-800 text-white'
                  : 'text-gray-500 hover:text-gray-400'
              }`}
            >
              Thumbnails ({thumbnails.length})
            </button>
            <button
              onClick={() => setTab('clips')}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
                tab === 'clips'
                  ? 'bg-gray-800 text-white'
                  : 'text-gray-500 hover:text-gray-400'
              }`}
            >
              Clips ({clips.length}){keptIds.size > 0 ? ` • ${keptIds.size} kept` : ''}
            </button>
          </div>

          {/* Thumbnail Gallery */}
          {tab === 'thumbnails' && (
            <ThumbnailGallery thumbnails={thumbnails} />
          )}

          {/* Clip List */}
          {tab === 'clips' && (
            <div className="space-y-6">
              <div className="space-y-3">
                {clips.length === 0 ? (
                  <p className="text-gray-500 text-center py-8">No clips detected</p>
                ) : (
                  clips.map((c) => <ClipCard key={c.id} clip={c} isKept={keptIds.has(c.id)} onKeep={handleKeep} />)
                )}
              </div>
              {keptIds.size > 0 && (
                <ExportPanel jobId={id} keptCount={keptIds.size} />
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}
