import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import URLInput from '../components/URLInput'
import JobStatusBadge from '../components/JobStatusBadge'
import { useApi } from '../hooks/useApi'
import { formatDuration } from '../utils/formatters'

export default function Home() {
  const [loading, setLoading] = useState(false)
  const [jobs, setJobs] = useState([])
  const navigate = useNavigate()
  const api = useApi()

  useEffect(() => {
    api.listJobs().then(setJobs).catch(console.error)
  }, [])

  const handleSubmit = async (value, mode = 'youtube') => {
    setLoading(true)
    try {
      const job = mode === 'local'
        ? await api.createLocalJob(value)
        : await api.createJob(value)
      navigate(`/job/${job.id}`)
    } catch (err) {
      console.error('Failed to create job:', err)
      alert(
        mode === 'local'
          ? (err.response?.data?.detail || 'Failed to start local job. Check the file path and try again.')
          : 'Failed to submit video. Check the URL and try again.'
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-12">
      {/* Hero */}
      <div className="text-center space-y-4 py-8">
        <h1 className="text-4xl font-bold text-white">
          AI-Powered Thumbnails & Clips
        </h1>
        <p className="text-gray-400 max-w-lg mx-auto">
          Paste a YouTube URL or point ClipForge at a local file to generate
          high-CTR thumbnails and discover your best short-form clip moments,
          scored for virality.
        </p>
      </div>

      {/* URL Input */}
      <URLInput onSubmit={handleSubmit} loading={loading} />

      {/* Recent Jobs */}
      {jobs.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-gray-300">Recent Jobs</h2>
          <div className="space-y-2">
            {jobs.map((job) => (
              <button
                key={job.id}
                onClick={() => navigate(`/job/${job.id}`)}
                className="w-full flex items-center justify-between px-4 py-3
                           bg-gray-900 border border-gray-800 rounded-lg
                           hover:border-gray-700 transition-colors text-left"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <JobStatusBadge status={job.status} />
                  <span className="text-sm text-white truncate">
                    {job.video_title || job.youtube_url}
                  </span>
                </div>
                <div className="flex items-center gap-3 flex-shrink-0">
                  {job.video_duration_seconds && (
                    <span className="text-xs text-gray-500">
                      {formatDuration(job.video_duration_seconds)}
                    </span>
                  )}
                  <span className="text-xs text-gray-600">
                    {job.id}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
