const statusConfig = {
  pending: { label: 'Pending', color: 'bg-gray-500/20 text-gray-400' },
  downloading: { label: 'Downloading', color: 'bg-blue-500/20 text-blue-400' },
  transcribing: { label: 'Transcribing', color: 'bg-purple-500/20 text-purple-400' },
  analyzing: { label: 'Analyzing', color: 'bg-yellow-500/20 text-yellow-400' },
  generating: { label: 'Generating', color: 'bg-indigo-500/20 text-indigo-400' },
  complete: { label: 'Complete', color: 'bg-green-500/20 text-green-400' },
  failed: { label: 'Failed', color: 'bg-red-500/20 text-red-400' },
}

export default function JobStatusBadge({ status }) {
  const config = statusConfig[status] || statusConfig.pending
  const isProcessing = !['complete', 'failed', 'pending'].includes(status)

  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${config.color}`}>
      {isProcessing && (
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-current opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-current" />
        </span>
      )}
      {config.label}
    </span>
  )
}
