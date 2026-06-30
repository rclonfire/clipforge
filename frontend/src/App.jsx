import { Routes, Route, Link } from 'react-router-dom'
import Home from './pages/Home'
import JobDetail from './pages/JobDetail'

export default function App() {
  return (
    <div className="min-h-screen bg-gray-950">
      <nav className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <Link to="/" className="text-xl font-extrabold text-white tracking-tight" style={{ fontFamily: 'Syne, sans-serif' }}>
            Clip<span className="text-indigo-400">Forge</span>
          </Link>
          <span className="text-xs text-gray-500">AI Thumbnail & Clip Generator</span>
        </div>
      </nav>
      <main className="max-w-6xl mx-auto px-4 py-8">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/job/:id" element={<JobDetail />} />
        </Routes>
      </main>
    </div>
  )
}
