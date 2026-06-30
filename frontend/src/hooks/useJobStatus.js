import { useState, useEffect, useRef } from 'react'
import { useApi } from './useApi'

export function useJobStatus(jobId) {
  const [job, setJob] = useState(null)
  const [progress, setProgress] = useState({ stage: '', message: '' })
  const intervalRef = useRef(null)
  const api = useApi()

  useEffect(() => {
    if (!jobId) return

    // Initial fetch
    api.getJob(jobId).then(setJob).catch(console.error)

    // Poll for updates
    intervalRef.current = setInterval(async () => {
      try {
        const [jobData, progressData] = await Promise.all([
          api.getJob(jobId),
          api.getJobProgress(jobId),
        ])
        setJob(jobData)
        setProgress(progressData)

        // Stop polling when job is complete or failed
        if (jobData.status === 'complete' || jobData.status === 'failed') {
          clearInterval(intervalRef.current)
        }
      } catch (err) {
        console.error('Poll error:', err)
      }
    }, 2000)

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [jobId])

  return { job, progress }
}
