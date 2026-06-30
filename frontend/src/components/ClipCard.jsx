import { useState, useRef } from 'react'
import { formatTimestamp, scoreColor, scoreBg } from '../utils/formatters'
import ScoreBreakdown from './ScoreBreakdown'

const clipTypeBadges = {
  punchline: 'bg-pink-500/20 text-pink-400',
  reaction: 'bg-orange-500/20 text-orange-400',
  relatable_moment: 'bg-blue-500/20 text-blue-400',
  story_hook: 'bg-purple-500/20 text-purple-400',
  visual_gag: 'bg-yellow-500/20 text-yellow-400',
  hot_take: 'bg-red-500/20 text-red-400',
  transformation: 'bg-green-500/20 text-green-400',
}

const editTypeBadges = {
  hook: { bg: 'bg-red-500/20', text: 'text-red-400', icon: '\u26A1' },
  pacing: { bg: 'bg-blue-500/20', text: 'text-blue-400', icon: '\u23F1\uFE0F' },
  visual: { bg: 'bg-purple-500/20', text: 'text-purple-400', icon: '\uD83C\uDFA5' },
  audio: { bg: 'bg-green-500/20', text: 'text-green-400', icon: '\uD83C\uDFB5' },
  text_overlay: { bg: 'bg-yellow-500/20', text: 'text-yellow-400', icon: '\uD83D\uDCDD' },
  ending: { bg: 'bg-orange-500/20', text: 'text-orange-400', icon: '\uD83C\uDFAC' },
}

const priorityDot = {
  high: 'bg-red-400',
  medium: 'bg-yellow-400',
  low: 'bg-gray-500',
}

function ClipPreview({ previewUrl }) {
  const videoRef = useRef(null)
  const [isPlaying, setIsPlaying] = useState(false)

  const togglePlay = () => {
    if (!videoRef.current) return
    if (videoRef.current.paused) {
      videoRef.current.muted = false
      videoRef.current.play()
      setIsPlaying(true)
    } else {
      videoRef.current.pause()
      setIsPlaying(false)
    }
  }

  if (!previewUrl) return null
  return (
    <div className="relative mb-3 cursor-pointer" onClick={togglePlay}>
      <video
        ref={videoRef}
        src={previewUrl}
        className="w-full rounded-lg aspect-[9/16] max-h-[400px] object-contain bg-black"
        loop
        preload="metadata"
        onEnded={() => setIsPlaying(false)}
      />
      {/* Play/pause button overlay */}
      {!isPlaying && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="bg-black/60 rounded-full p-4 hover:bg-black/80 transition-colors">
            <svg className="w-8 h-8 text-white ml-1" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
          </div>
        </div>
      )}
    </div>
  )
}

export default function ClipCard({ clip, isKept, onKeep }) {
  const [showEdits, setShowEdits] = useState(false)
  const [keepLoading, setKeepLoading] = useState(false)
  const badgeClass = clipTypeBadges[clip.clip_type] || 'bg-gray-500/20 text-gray-400'
  const suggestions = clip.edit_suggestions || []

  const handleKeepClick = async () => {
    setKeepLoading(true)
    try {
      await onKeep(clip.id, !isKept)
    } finally {
      setKeepLoading(false)
    }
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 hover:border-gray-700 transition-colors">
      <ClipPreview previewUrl={clip.preview_url} />
      <div className="flex items-start gap-4">
        {/* Score */}
        <div className={`flex-shrink-0 w-16 h-16 rounded-xl border flex flex-col items-center justify-center ${scoreBg(clip.virality_score)}`}>
          <span className={`text-2xl font-bold ${scoreColor(clip.virality_score)}`}>
            {clip.virality_score}
          </span>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0 space-y-2">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="font-medium text-white truncate">
              {clip.clip_title || 'Untitled Clip'}
            </h3>
            <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${badgeClass}`}>
              {(clip.clip_type || 'clip').replace('_', ' ')}
            </span>
            <span className="text-xs text-gray-500">
              {formatTimestamp(clip.start_time_seconds)} - {formatTimestamp(clip.end_time_seconds)}
              {' '}({Math.round(clip.duration_seconds)}s)
            </span>
          </div>

          {clip.hook_text && (
            <p className="text-sm text-indigo-400 italic">
              Hook: &ldquo;{clip.hook_text}&rdquo;
            </p>
          )}

          {clip.transcript_snippet && (
            <p className="text-sm text-gray-400 line-clamp-2">
              {clip.transcript_snippet}
            </p>
          )}

          {clip.reasoning && (
            <p className="text-sm text-gray-500">
              {clip.reasoning}
            </p>
          )}

          {clip.suggested_caption && (
            <p className="text-xs text-gray-600 truncate">
              Caption: {clip.suggested_caption}
            </p>
          )}

          <ScoreBreakdown breakdown={clip.score_breakdown} />

          {/* Edit Suggestions */}
          {suggestions.length > 0 && (
            <div className="pt-2">
              <button
                onClick={() => setShowEdits(!showEdits)}
                className="flex items-center gap-1.5 text-xs font-medium text-indigo-400 hover:text-indigo-300 transition-colors"
              >
                <svg
                  className={`w-3.5 h-3.5 transition-transform ${showEdits ? 'rotate-90' : ''}`}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                </svg>
                Edit Suggestions ({suggestions.length})
              </button>

              {showEdits && (
                <div className="mt-2 space-y-2">
                  {suggestions.map((s, i) => {
                    const badge = editTypeBadges[s.type] || editTypeBadges.visual
                    const dotClass = priorityDot[s.priority] || priorityDot.medium
                    return (
                      <div
                        key={i}
                        className="bg-gray-800/60 border border-gray-700/50 rounded-lg p-3 space-y-1"
                      >
                        <div className="flex items-center gap-2">
                          <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wide ${badge.bg} ${badge.text}`}>
                            <span>{badge.icon}</span>
                            {s.type.replace('_', ' ')}
                          </span>
                          <span className={`w-2 h-2 rounded-full ${dotClass}`} title={`${s.priority} priority`} />
                          <span className="text-[10px] text-gray-600 uppercase tracking-wide">{s.priority}</span>
                        </div>
                        <p className="text-sm text-gray-300 leading-relaxed">
                          {s.suggestion}
                        </p>
                        {s.reference && (
                          <p className="text-xs text-gray-500 italic">
                            Based on: {s.reference}
                          </p>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      {onKeep && (
        <div className="mt-3 pt-3 border-t border-gray-800 flex justify-end">
          <button
            onClick={handleKeepClick}
            disabled={keepLoading}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
              keepLoading ? 'opacity-50 cursor-wait' :
              isKept
                ? 'bg-green-500/20 text-green-400 border border-green-500/40 hover:bg-green-500/30'
                : 'bg-gray-800 text-gray-400 border border-gray-700 hover:bg-gray-700 hover:text-white'
            }`}
          >
            {keepLoading ? '...' : isKept ? 'Kept' : 'Keep'}
          </button>
        </div>
      )}
    </div>
  )
}
