import { useState, useEffect, useRef } from 'react'
import { useApi } from '../hooks/useApi'

const PRESETS = [
  { id: 'tiktok', label: 'TikTok', desc: '1080x1920, max 60s', icon: null },
  { id: 'shorts', label: 'YouTube Shorts', desc: '1080x1920, max 60s', icon: null },
  { id: 'original', label: 'Original (16:9)', desc: '1920x1080, no crop', icon: null },
]

export default function ExportPanel({ jobId, keptCount }) {
  const [platform, setPlatform] = useState('tiktok')
  const [verticalCrop, setVerticalCrop] = useState(true)
  const [captions, setCaptions] = useState(true)
  const [exporting, setExporting] = useState(false)
  const [batchId, setBatchId] = useState(null)
  const [exportStatus, setExportStatus] = useState(null)
  const [error, setError] = useState(null)
  const [prepping, setPrepping] = useState(false)
  const [prepResult, setPrepResult] = useState(null)
  const [prepError, setPrepError] = useState(null)
  const pollRef = useRef(null)
  const api = useApi()

  // When "original" preset selected, disable vertical crop
  useEffect(() => {
    if (platform === 'original') {
      setVerticalCrop(false)
    } else {
      setVerticalCrop(true)
    }
  }, [platform])

  // Poll export status
  useEffect(() => {
    if (!batchId) return
    pollRef.current = setInterval(async () => {
      try {
        const status = await api.getExportStatus(jobId, batchId)
        setExportStatus(status)
        if (status.status === 'complete' || status.status === 'failed') {
          clearInterval(pollRef.current)
          setExporting(false)
          if (status.status === 'failed') {
            setError(status.progress_message || 'Export failed')
          }
        }
      } catch (err) {
        console.error('Export status poll failed:', err)
      }
    }, 1500) // Poll every 1.5s
    return () => clearInterval(pollRef.current)
  }, [batchId, jobId])

  const handleExport = async () => {
    setError(null)
    setExporting(true)
    setExportStatus(null)
    try {
      const result = await api.createExport(jobId, {
        platform,
        vertical_crop: verticalCrop,
        captions,
      })
      setBatchId(result.batch_id)
      setExportStatus({
        status: 'pending',
        progress_message: 'Export queued...',
        total_clips: result.total_clips,
        completed_clips: 0,
      })
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to start export')
      setExporting(false)
    }
  }

  const handlePrep = async () => {
    setPrepError(null)
    setPrepping(true)
    try {
      const result = await api.createPostPrep(jobId, {
        platforms: ['youtube', 'tiktok'],
        posts_per_day: 3,
      })
      setPrepResult(result)
    } catch (err) {
      setPrepError(err.response?.data?.detail || 'Failed to prep clips for posting')
    } finally {
      setPrepping(false)
    }
  }

  const downloadUrl = batchId ? api.getExportDownloadUrl(jobId, batchId) : null
  const prepDownloadUrl =
    prepResult ? api.getPostPrepDownloadUrl(jobId, prepResult.prep_id) : null
  const isComplete = exportStatus?.status === 'complete'
  const progressPct = exportStatus
    ? Math.round((exportStatus.completed_clips / Math.max(exportStatus.total_clips, 1)) * 100)
    : 0

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 space-y-5">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">Export Clips</h3>
        <span className="text-sm text-gray-500">{keptCount} clip{keptCount !== 1 ? 's' : ''} selected</span>
      </div>

      {/* Platform Presets */}
      <div className="space-y-2">
        <label className="text-sm font-medium text-gray-400">Platform</label>
        <div className="grid grid-cols-3 gap-2">
          {PRESETS.map((preset) => (
            <button
              key={preset.id}
              onClick={() => setPlatform(preset.id)}
              disabled={exporting}
              className={`p-3 rounded-lg border text-left transition-colors ${
                platform === preset.id
                  ? 'border-indigo-500 bg-indigo-500/10 text-white'
                  : 'border-gray-700 bg-gray-800/50 text-gray-400 hover:border-gray-600'
              } ${exporting ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <div className="text-sm font-medium">{preset.label}</div>
              <div className="text-xs text-gray-500 mt-0.5">{preset.desc}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Options */}
      <div className="flex items-center gap-6">
        <label className={`flex items-center gap-2 text-sm ${platform === 'original' ? 'opacity-40' : 'text-gray-300'}`}>
          <input
            type="checkbox"
            checked={verticalCrop}
            onChange={(e) => setVerticalCrop(e.target.checked)}
            disabled={exporting || platform === 'original'}
            className="rounded border-gray-600 bg-gray-800 text-indigo-500 focus:ring-indigo-500"
          />
          9:16 Vertical Crop
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-300">
          <input
            type="checkbox"
            checked={captions}
            onChange={(e) => setCaptions(e.target.checked)}
            disabled={exporting}
            className="rounded border-gray-600 bg-gray-800 text-indigo-500 focus:ring-indigo-500"
          />
          Caption Burn-In
        </label>
      </div>

      {/* Export Button */}
      {!isComplete && (
        <button
          onClick={handleExport}
          disabled={exporting}
          className={`w-full py-3 rounded-lg text-sm font-semibold transition-colors ${
            exporting
              ? 'bg-indigo-500/30 text-indigo-300 cursor-wait'
              : 'bg-indigo-500 text-white hover:bg-indigo-600'
          }`}
        >
          {exporting ? 'Exporting...' : `Export ${keptCount} Clip${keptCount !== 1 ? 's' : ''}`}
        </button>
      )}

      {/* Progress */}
      {exportStatus && !isComplete && exportStatus.status !== 'failed' && (
        <div className="space-y-2">
          <div className="flex justify-between text-xs text-gray-500">
            <span>{exportStatus.progress_message}</span>
            <span>{exportStatus.completed_clips}/{exportStatus.total_clips}</span>
          </div>
          <div className="w-full bg-gray-800 rounded-full h-2">
            <div
              className="bg-indigo-500 h-2 rounded-full transition-all duration-500"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        </div>
      )}

      {/* Download */}
      {isComplete && downloadUrl && (
        <a
          href={downloadUrl}
          download
          className="block w-full py-3 rounded-lg text-sm font-semibold text-center bg-green-500 text-white hover:bg-green-600 transition-colors"
        >
          Download ZIP ({exportStatus.total_clips} clip{exportStatus.total_clips !== 1 ? 's' : ''} + thumbnails)
        </a>
      )}

      {/* Prep for posting (YouTube Shorts + TikTok) */}
      {isComplete && (
        <div className="border-t border-gray-800 pt-5 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-semibold text-white">Prep for posting</h4>
              <p className="text-xs text-gray-500 mt-0.5">
                Titles, captions, hashtags & a 2–3/day schedule for YouTube Shorts + TikTok
              </p>
            </div>
          </div>

          {!prepResult && (
            <button
              onClick={handlePrep}
              disabled={prepping}
              className={`w-full py-3 rounded-lg text-sm font-semibold transition-colors ${
                prepping
                  ? 'bg-purple-500/30 text-purple-300 cursor-wait'
                  : 'bg-purple-500 text-white hover:bg-purple-600'
              }`}
            >
              {prepping ? 'Generating post copy...' : 'Prep for YouTube + TikTok'}
            </button>
          )}

          {prepResult && (
            <div className="space-y-3">
              <a
                href={prepDownloadUrl}
                download
                className="block w-full py-3 rounded-lg text-sm font-semibold text-center bg-green-500 text-white hover:bg-green-600 transition-colors"
              >
                Download post-ready bundle ({prepResult.clip_count} clip
                {prepResult.clip_count !== 1 ? 's' : ''})
              </a>

              {prepResult.clips?.[0] && (
                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-3 space-y-2">
                  <p className="text-xs font-medium text-gray-400">Example — clip 1</p>
                  <div className="text-xs text-gray-300">
                    <span className="text-gray-500">YouTube:</span>{' '}
                    {prepResult.clips[0].youtube.title}
                  </div>
                  <div className="text-xs text-gray-300">
                    <span className="text-gray-500">TikTok:</span>{' '}
                    {prepResult.clips[0].tiktok.caption}{' '}
                    <span className="text-purple-400">
                      {prepResult.clips[0].tiktok.hashtags.join(' ')}
                    </span>
                  </div>
                </div>
              )}

              <p className="text-xs text-gray-500">
                {prepResult.schedule?.length || 0} posts scheduled starting{' '}
                {prepResult.start_date}. Upload each clip to the platform's own scheduler
                using <code className="text-gray-400">schedule.md</code> in the bundle.
              </p>
            </div>
          )}

          {prepError && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
              <p className="text-sm text-red-400">{prepError}</p>
            </div>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}
    </div>
  )
}
