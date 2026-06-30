import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
})

export function useApi() {
  return {
    createJob: (youtubeUrl) =>
      api.post('/jobs', { youtube_url: youtubeUrl }).then((r) => r.data),

    listJobs: () =>
      api.get('/jobs').then((r) => r.data),

    getJob: (jobId) =>
      api.get(`/jobs/${jobId}`).then((r) => r.data),

    getJobProgress: (jobId) =>
      api.get(`/jobs/${jobId}/progress`).then((r) => r.data),

    getThumbnails: (jobId) =>
      api.get(`/jobs/${jobId}/thumbnails`).then((r) => r.data),

    getClips: (jobId) =>
      api.get(`/jobs/${jobId}/clips`).then((r) => r.data),

    updateClipKept: (jobId, clipId, kept) =>
      api.patch(`/jobs/${jobId}/clips/${clipId}`, { kept }).then((r) => r.data),

    createExport: (jobId, options) =>
      api.post(`/jobs/${jobId}/exports`, options).then((r) => r.data),

    getExportStatus: (jobId, batchId) =>
      api.get(`/jobs/${jobId}/exports/${batchId}/status`).then((r) => r.data),

    getExportDownloadUrl: (jobId, batchId) =>
      `/api/jobs/${jobId}/exports/${batchId}/download`,

    getClipThumbnailUrl: (jobId, clipId) =>
      `/api/jobs/${jobId}/clips/${clipId}/thumbnail`,
  }
}
