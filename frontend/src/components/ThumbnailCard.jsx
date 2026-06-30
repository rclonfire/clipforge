import { ctrTierBadge } from '../utils/formatters'

export default function ThumbnailCard({ thumbnail }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden hover:border-gray-700 transition-colors">
      <div className="aspect-video bg-gray-800 relative group">
        <img
          src={thumbnail.file_url}
          alt={thumbnail.text_overlay || 'Thumbnail'}
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-colors flex items-center justify-center">
          <a
            href={thumbnail.file_url}
            download
            className="opacity-0 group-hover:opacity-100 transition-opacity px-4 py-2 bg-white text-gray-900 rounded-lg font-medium text-sm"
          >
            Download
          </a>
        </div>
      </div>
      <div className="p-4 space-y-3">
        <div className="flex items-center gap-2">
          {thumbnail.estimated_ctr_tier && (
            <span className={`px-2 py-0.5 rounded-full text-xs font-medium border ${ctrTierBadge(thumbnail.estimated_ctr_tier)}`}>
              {thumbnail.estimated_ctr_tier.toUpperCase()} CTR
            </span>
          )}
          {thumbnail.text_overlay && (
            <span className="text-xs text-gray-500 truncate">
              "{thumbnail.text_overlay}"
            </span>
          )}
        </div>
        {thumbnail.reasoning && (
          <p className="text-sm text-gray-400 leading-relaxed">
            {thumbnail.reasoning}
          </p>
        )}
      </div>
    </div>
  )
}
