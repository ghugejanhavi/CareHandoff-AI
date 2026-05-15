import type {
  AnalysisResult,
  ReviewPackage,
  ValidationResult,
  CareSetting,
  FeedbackAction,
} from './types'

const BASE = '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error((body as { detail?: string }).detail ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string }>('/health'),

  listPatients: () => request<{ patients: string[] }>('/patients'),

  analyze: (hadm_id: string, care_setting: CareSetting) =>
    request<AnalysisResult>('/analyze', {
      method: 'POST',
      body: JSON.stringify({ hadm_id, care_setting }),
    }),

  getReview: () => request<ReviewPackage>('/review'),

  submitFeedback: (item_id: string, action: FeedbackAction, notes = '') =>
    request<{ status: string; message: string; reanalysis: string }>(
      '/review/feedback',
      { method: 'POST', body: JSON.stringify({ item_id, action, notes }) },
    ),

  confirmAll: () =>
    request<{ confirmed: number; message: string }>('/review/confirm-all', {
      method: 'POST',
    }),

  generateNote: () =>
    request<{ revised_note: string; gaps_addressed: number }>('/note/generate', {
      method: 'POST',
    }),

  getValidation: () => request<ValidationResult>('/note/validation'),

  saveNote: () =>
    request<{ success: boolean; message: string }>('/note/save', {
      method: 'POST',
    }),
}
