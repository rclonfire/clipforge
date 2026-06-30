import { useState, useRef, useEffect } from 'react'
import CompareView from './CompareView'

const CTR_CONFIG = {
  high: { label: 'HIGH CTR', color: '#34d399', glow: 'rgba(52,211,153,0.3)', gradient: 'from-emerald-500 to-emerald-400' },
  medium: { label: 'MED CTR', color: '#fbbf24', glow: 'rgba(251,191,36,0.3)', gradient: 'from-amber-500 to-yellow-400' },
  low: { label: 'LOW CTR', color: '#6b7280', glow: 'rgba(107,114,128,0.2)', gradient: 'from-gray-500 to-gray-400' },
}

export default function ThumbnailGallery({ thumbnails }) {
  const [selectedIdx, setSelectedIdx] = useState(0)
  const [compareMode, setCompareMode] = useState(false)
  const [compareIdxA, setCompareIdxA] = useState(0)
  const [compareIdxB, setCompareIdxB] = useState(1)
  const [expandedReasoning, setExpandedReasoning] = useState(false)
  const [heroLoaded, setHeroLoaded] = useState(false)
  const filmstripRef = useRef(null)

  const selected = thumbnails[selectedIdx]
  const tierConfig = CTR_CONFIG[selected?.estimated_ctr_tier] || CTR_CONFIG.medium

  // Reset hero load state on selection change
  useEffect(() => {
    setHeroLoaded(false)
  }, [selectedIdx])

  // Keyboard navigation
  useEffect(() => {
    const handler = (e) => {
      if (compareMode) return
      if (e.key === 'ArrowLeft') {
        setSelectedIdx((i) => Math.max(0, i - 1))
      } else if (e.key === 'ArrowRight') {
        setSelectedIdx((i) => Math.min(thumbnails.length - 1, i + 1))
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [compareMode, thumbnails.length])

  if (!thumbnails.length) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-3">
        <div className="w-16 h-16 rounded-2xl bg-gray-900 border border-gray-800 flex items-center justify-center">
          <svg className="w-7 h-7 text-gray-700" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0022.5 18.75V5.25A2.25 2.25 0 0020.25 3H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21z" />
          </svg>
        </div>
        <p className="text-gray-600 text-sm font-medium tracking-wide">No thumbnails generated yet</p>
      </div>
    )
  }

  if (compareMode) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-widest">Compare Mode</h3>
          <button
            onClick={() => setCompareMode(false)}
            className="gallery-btn-secondary"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
            Exit Compare
          </button>
        </div>
        <CompareView
          thumbnails={thumbnails}
          idxA={compareIdxA}
          idxB={compareIdxB}
          onChangeA={setCompareIdxA}
          onChangeB={setCompareIdxB}
        />
      </div>
    )
  }

  return (
    <div className="thumbnail-gallery space-y-5">
      {/* Toolbar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-widest">
            {thumbnails.length} Concept{thumbnails.length !== 1 ? 's' : ''}
          </h3>
          <div className="flex items-center gap-1.5 text-xs text-gray-600">
            <kbd className="px-1.5 py-0.5 bg-gray-900 border border-gray-800 rounded text-[10px]">&larr;</kbd>
            <kbd className="px-1.5 py-0.5 bg-gray-900 border border-gray-800 rounded text-[10px]">&rarr;</kbd>
            <span>navigate</span>
          </div>
        </div>
        {thumbnails.length >= 2 && (
          <button
            onClick={() => {
              setCompareIdxA(0)
              setCompareIdxB(Math.min(1, thumbnails.length - 1))
              setCompareMode(true)
            }}
            className="gallery-btn-secondary"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7" />
            </svg>
            Compare
          </button>
        )}
      </div>

      {/* Hero Preview */}
      <div className="hero-container relative rounded-2xl overflow-hidden border border-gray-800/60 bg-gray-900">
        {/* Ambient glow behind image */}
        <div
          className="absolute inset-0 opacity-30 blur-3xl scale-110 transition-opacity duration-700"
          style={{
            backgroundImage: `url(${selected.file_url})`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
          }}
        />

        {/* Main image */}
        <div className="relative aspect-video">
          <img
            key={selected.file_url}
            src={selected.file_url}
            alt={selected.text_overlay || `Thumbnail ${selectedIdx + 1}`}
            className={`w-full h-full object-cover transition-all duration-500 ${heroLoaded ? 'opacity-100 scale-100' : 'opacity-0 scale-[1.02]'}`}
            onLoad={() => setHeroLoaded(true)}
          />

          {/* Gradient overlays */}
          <div className="absolute inset-0 bg-gradient-to-t from-gray-950/90 via-transparent to-transparent" />
          <div className="absolute inset-0 bg-gradient-to-r from-gray-950/30 to-transparent" />

          {/* Top-right: badges */}
          <div className="absolute top-4 right-4 flex items-center gap-2">
            {/* AI Generated badge */}
            {selected.generation_type === 'gemini' && (
              <div
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-full backdrop-blur-md border"
                style={{
                  backgroundColor: 'rgba(168, 85, 247, 0.15)',
                  borderColor: 'rgba(168, 85, 247, 0.4)',
                  boxShadow: '0 0 20px rgba(168, 85, 247, 0.2)',
                }}
              >
                <span className="text-[10px]">&#10024;</span>
                <span className="text-xs font-bold tracking-wider text-purple-300">AI Generated</span>
              </div>
            )}
            {/* CTR Badge */}
            <div
              className="ctr-badge flex items-center gap-2 px-3 py-1.5 rounded-full backdrop-blur-md border"
              style={{
                backgroundColor: `${tierConfig.color}15`,
                borderColor: `${tierConfig.color}40`,
                boxShadow: `0 0 20px ${tierConfig.glow}`,
              }}
            >
              <span
                className="w-2 h-2 rounded-full animate-pulse"
                style={{ backgroundColor: tierConfig.color }}
              />
              <span
                className="text-xs font-bold tracking-wider"
                style={{ color: tierConfig.color }}
              >
                {tierConfig.label}
              </span>
            </div>
          </div>

          {/* Bottom overlay: metadata */}
          <div className="absolute bottom-0 left-0 right-0 p-6">
            <div className="flex items-end justify-between gap-4">
              <div className="space-y-2 min-w-0 flex-1">
                {selected.text_overlay && (
                  <p className="text-lg font-bold text-white/90 tracking-wide">
                    "{selected.text_overlay}"
                  </p>
                )}
                <div className="flex items-center gap-3 flex-wrap">
                  {selected.text_position && (
                    <span className="meta-chip">
                      {selected.text_position.replace('-', ' ')}
                    </span>
                  )}
                  {selected.style_notes && (
                    <span className="meta-chip">
                      {selected.style_notes}
                    </span>
                  )}
                  <span className="meta-chip">
                    Frame #{selected.frame_index}
                  </span>
                </div>
              </div>

              {/* Download button */}
              <a
                href={selected.file_url}
                download
                className="gallery-btn-primary flex-shrink-0"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
                </svg>
                Download
              </a>
            </div>
          </div>

          {/* Navigation arrows */}
          {selectedIdx > 0 && (
            <button
              onClick={() => setSelectedIdx(selectedIdx - 1)}
              className="hero-nav-btn left-3"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
              </svg>
            </button>
          )}
          {selectedIdx < thumbnails.length - 1 && (
            <button
              onClick={() => setSelectedIdx(selectedIdx + 1)}
              className="hero-nav-btn right-3"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* AI Reasoning Panel */}
      {selected.reasoning && (
        <div className="reasoning-panel">
          <button
            onClick={() => setExpandedReasoning(!expandedReasoning)}
            className="w-full flex items-center justify-between px-4 py-3 text-left"
          >
            <div className="flex items-center gap-2.5">
              <div className="w-5 h-5 rounded-md bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center">
                <svg className="w-3 h-3 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
                </svg>
              </div>
              <span className="text-sm font-medium text-gray-300">AI Reasoning</span>
            </div>
            <svg
              className={`w-4 h-4 text-gray-500 transition-transform duration-300 ${expandedReasoning ? 'rotate-180' : ''}`}
              fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
            </svg>
          </button>
          <div
            className={`reasoning-content ${expandedReasoning ? 'expanded' : ''}`}
          >
            <div className="px-4 pb-4">
              <p className="text-sm text-gray-400 leading-relaxed">
                {selected.reasoning}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Filmstrip */}
      <div className="relative" ref={filmstripRef}>
        <div className="filmstrip flex gap-3 overflow-x-auto pb-2 scrollbar-hide">
          {thumbnails.map((thumb, i) => {
            const isSelected = i === selectedIdx
            const tier = CTR_CONFIG[thumb.estimated_ctr_tier] || CTR_CONFIG.medium
            return (
              <button
                key={thumb.id || i}
                onClick={() => setSelectedIdx(i)}
                className={`filmstrip-item flex-shrink-0 relative rounded-xl overflow-hidden border-2 transition-all duration-300 group ${
                  isSelected
                    ? 'border-indigo-500 ring-2 ring-indigo-500/30 scale-[1.02]'
                    : 'border-gray-800/60 hover:border-gray-700 hover:scale-[1.01]'
                }`}
                style={{
                  width: thumbnails.length <= 3 ? 'calc(33.333% - 8px)' : '200px',
                }}
              >
                <div className="aspect-video">
                  <img
                    src={thumb.file_url}
                    alt={thumb.text_overlay || `Thumbnail ${i + 1}`}
                    className={`w-full h-full object-cover transition-all duration-300 ${
                      isSelected ? 'brightness-100' : 'brightness-75 group-hover:brightness-90'
                    }`}
                  />
                </div>

                {/* Filmstrip overlay */}
                <div className={`absolute inset-0 transition-opacity duration-300 ${
                  isSelected ? 'opacity-0' : 'opacity-100 group-hover:opacity-50'
                }`}>
                  <div className="absolute inset-0 bg-gradient-to-t from-gray-950/80 to-transparent" />
                </div>

                {/* CTR indicator line */}
                <div
                  className="absolute bottom-0 left-0 right-0 h-0.5 transition-all duration-300"
                  style={{
                    backgroundColor: tier.color,
                    opacity: isSelected ? 1 : 0.5,
                    boxShadow: isSelected ? `0 0 8px ${tier.glow}` : 'none',
                  }}
                />

                {/* Index badge */}
                <div className={`absolute top-2 left-2 w-6 h-6 rounded-md flex items-center justify-center text-[10px] font-bold transition-all duration-300 ${
                  isSelected
                    ? 'bg-indigo-500 text-white'
                    : 'bg-gray-900/80 text-gray-400 backdrop-blur-sm'
                }`}>
                  {i + 1}
                </div>

                {/* AI badge on filmstrip */}
                {thumb.generation_type === 'gemini' && (
                  <div className="absolute top-2 right-2 w-5 h-5 rounded-md flex items-center justify-center bg-purple-500/30 backdrop-blur-sm border border-purple-500/40">
                    <span className="text-[8px]">&#10024;</span>
                  </div>
                )}
              </button>
            )
          })}
        </div>
      </div>
    </div>
  )
}
