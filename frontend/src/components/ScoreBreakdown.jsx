const labels = {
  hook_strength: 'Hook',
  standalone_clarity: 'Clarity',
  emotional_arc: 'Arc',
  trend_alignment: 'Trend',
  rewatch_potential: 'Rewatch',
}

export default function ScoreBreakdown({ breakdown }) {
  if (!breakdown) return null

  return (
    <div className="flex gap-3 flex-wrap pt-1">
      {Object.entries(labels).map(([key, label]) => {
        const value = breakdown[key] || 0
        return (
          <div key={key} className="flex items-center gap-1.5">
            <span className="text-xs text-gray-500">{label}</span>
            <div className="w-16 h-1.5 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all"
                style={{
                  width: `${value}%`,
                  backgroundColor: value >= 70 ? '#22c55e' : value >= 40 ? '#eab308' : '#ef4444',
                }}
              />
            </div>
            <span className="text-xs text-gray-600">{value}</span>
          </div>
        )
      })}
    </div>
  )
}
