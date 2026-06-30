import { useState } from 'react'

const CTR_CONFIG = {
  high: { label: 'HIGH', color: '#34d399', bg: 'rgba(52,211,153,0.12)' },
  medium: { label: 'MED', color: '#fbbf24', bg: 'rgba(251,191,36,0.12)' },
  low: { label: 'LOW', color: '#6b7280', bg: 'rgba(107,114,128,0.12)' },
}

const CTR_RANK = { high: 3, medium: 2, low: 1 }

function ThumbnailSelector({ thumbnails, selectedIdx, onChange, side }) {
  return (
    <div className="flex items-center gap-2">
      <span className={`text-[10px] font-bold uppercase tracking-widest ${
        side === 'a' ? 'text-indigo-400' : 'text-violet-400'
      }`}>
        {side === 'a' ? 'Left' : 'Right'}
      </span>
      <div className="flex gap-1.5">
        {thumbnails.map((_, i) => (
          <button
            key={i}
            onClick={() => onChange(i)}
            className={`w-7 h-7 rounded-lg text-xs font-bold transition-all duration-200 ${
              i === selectedIdx
                ? side === 'a'
                  ? 'bg-indigo-500 text-white shadow-lg shadow-indigo-500/30'
                  : 'bg-violet-500 text-white shadow-lg shadow-violet-500/30'
                : 'bg-gray-800 text-gray-500 hover:bg-gray-750 hover:text-gray-300'
            }`}
          >
            {i + 1}
          </button>
        ))}
      </div>
    </div>
  )
}

function CompareCard({ thumbnail, side, isWinner }) {
  const tier = CTR_CONFIG[thumbnail.estimated_ctr_tier] || CTR_CONFIG.medium
  const accentColor = side === 'a' ? '#818cf8' : '#a78bfa'

  return (
    <div className="flex-1 min-w-0 space-y-3">
      <div
        className={`relative rounded-xl overflow-hidden border-2 transition-all duration-500 ${
          isWinner ? 'border-emerald-500/50 shadow-lg shadow-emerald-500/10' : 'border-gray-800/60'
        }`}
      >
        {/* Winner badge */}
        {isWinner && (
          <div className="absolute top-3 right-3 z-10 flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/20 backdrop-blur-md border border-emerald-500/30">
            <svg className="w-3 h-3 text-emerald-400" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z" clipRule="evenodd" />
            </svg>
            <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider">Stronger</span>
          </div>
        )}

        <div className="aspect-video">
          <img
            src={thumbnail.file_url}
            alt={thumbnail.text_overlay || 'Thumbnail'}
            className="w-full h-full object-cover"
          />
        </div>

        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-gray-950/80 to-transparent p-3">
          <div
            className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider"
            style={{ backgroundColor: tier.bg, color: tier.color }}
          >
            <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: tier.color }} />
            {tier.label} CTR
          </div>
        </div>
      </div>

      {/* Metadata */}
      <div className="space-y-2 px-1">
        {thumbnail.text_overlay && (
          <p className="text-sm font-semibold text-gray-200 truncate">
            "{thumbnail.text_overlay}"
          </p>
        )}
        <div className="flex items-center gap-2 flex-wrap">
          {thumbnail.text_position && (
            <span className="meta-chip text-[10px]">{thumbnail.text_position.replace('-', ' ')}</span>
          )}
          {thumbnail.style_notes && (
            <span className="meta-chip text-[10px]">{thumbnail.style_notes}</span>
          )}
        </div>
        {thumbnail.reasoning && (
          <p className="text-xs text-gray-500 leading-relaxed line-clamp-3">
            {thumbnail.reasoning}
          </p>
        )}
      </div>
    </div>
  )
}

function ComparisonMetrics({ thumbA, thumbB }) {
  const tierA = CTR_RANK[thumbA.estimated_ctr_tier] || 2
  const tierB = CTR_RANK[thumbB.estimated_ctr_tier] || 2
  const configA = CTR_CONFIG[thumbA.estimated_ctr_tier] || CTR_CONFIG.medium
  const configB = CTR_CONFIG[thumbB.estimated_ctr_tier] || CTR_CONFIG.medium

  const metrics = [
    {
      label: 'CTR Potential',
      valueA: tierA,
      valueB: tierB,
      maxVal: 3,
      colorA: configA.color,
      colorB: configB.color,
    },
    {
      label: 'Text Impact',
      valueA: thumbA.text_overlay ? Math.min(thumbA.text_overlay.split(' ').length, 5) : 0,
      valueB: thumbB.text_overlay ? Math.min(thumbB.text_overlay.split(' ').length, 5) : 0,
      maxVal: 5,
      colorA: '#818cf8',
      colorB: '#a78bfa',
    },
  ]

  return (
    <div className="compare-metrics flex flex-col items-center justify-center gap-4 px-2">
      <div className="w-px h-8 bg-gradient-to-b from-transparent via-gray-700 to-transparent" />
      <span className="text-[10px] font-bold text-gray-600 uppercase tracking-widest -rotate-90 whitespace-nowrap">
        vs
      </span>
      <div className="w-px h-8 bg-gradient-to-b from-transparent via-gray-700 to-transparent" />

      <div className="space-y-3 w-full">
        {metrics.map((m) => {
          const pctA = (m.valueA / m.maxVal) * 100
          const pctB = (m.valueB / m.maxVal) * 100
          return (
            <div key={m.label} className="space-y-1">
              <span className="text-[9px] font-semibold text-gray-600 uppercase tracking-wider block text-center">
                {m.label}
              </span>
              {/* Bar A (right-aligned) */}
              <div className="flex items-center gap-1">
                <div className="flex-1 h-1.5 rounded-full bg-gray-800 overflow-hidden flex justify-end">
                  <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{ width: `${pctA}%`, backgroundColor: m.colorA }}
                  />
                </div>
                <div className="w-0.5 h-3 bg-gray-700 rounded-full" />
                <div className="flex-1 h-1.5 rounded-full bg-gray-800 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{ width: `${pctB}%`, backgroundColor: m.colorB }}
                  />
                </div>
              </div>
            </div>
          )
        })}
      </div>

      <div className="w-px h-8 bg-gradient-to-b from-transparent via-gray-700 to-transparent" />
    </div>
  )
}

export default function CompareView({ thumbnails, idxA, idxB, onChangeA, onChangeB }) {
  const thumbA = thumbnails[idxA]
  const thumbB = thumbnails[idxB]

  if (!thumbA || !thumbB) return null

  const rankA = CTR_RANK[thumbA.estimated_ctr_tier] || 2
  const rankB = CTR_RANK[thumbB.estimated_ctr_tier] || 2
  const winnerA = rankA > rankB
  const winnerB = rankB > rankA

  return (
    <div className="compare-view space-y-5">
      {/* Selectors */}
      <div className="flex items-center justify-between">
        <ThumbnailSelector thumbnails={thumbnails} selectedIdx={idxA} onChange={onChangeA} side="a" />
        <ThumbnailSelector thumbnails={thumbnails} selectedIdx={idxB} onChange={onChangeB} side="b" />
      </div>

      {/* Side-by-side */}
      <div className="flex gap-3 items-start">
        <CompareCard thumbnail={thumbA} side="a" isWinner={winnerA} />
        <ComparisonMetrics thumbA={thumbA} thumbB={thumbB} />
        <CompareCard thumbnail={thumbB} side="b" isWinner={winnerB} />
      </div>

      {/* Download row */}
      <div className="flex gap-3">
        <a
          href={thumbA.file_url}
          download
          className="flex-1 gallery-btn-secondary justify-center"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
          </svg>
          Download Left
        </a>
        <a
          href={thumbB.file_url}
          download
          className="flex-1 gallery-btn-secondary justify-center"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
          </svg>
          Download Right
        </a>
      </div>
    </div>
  )
}
