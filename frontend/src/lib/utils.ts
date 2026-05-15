export function severityBadgeClass(s: string): string {
  switch (s.toLowerCase()) {
    case 'critical': return 'text-red-700 bg-red-50 border-red-200'
    case 'high':     return 'text-orange-700 bg-orange-50 border-orange-200'
    case 'medium':   return 'text-amber-700 bg-amber-50 border-amber-200'
    default:         return 'text-blue-700 bg-blue-50 border-blue-200'
  }
}

export function severityDotClass(s: string): string {
  switch (s.toLowerCase()) {
    case 'critical': return 'bg-red-500'
    case 'high':     return 'bg-orange-500'
    case 'medium':   return 'bg-amber-500'
    default:         return 'bg-blue-500'
  }
}

export function statusBadgeClass(s: string): string {
  switch (s) {
    case 'confirmed':                  return 'text-green-700 bg-green-50'
    case 'dismissed':                  return 'text-gray-500 bg-gray-100'
    case 'escalated':                  return 'text-purple-700 bg-purple-50'
    case 're_flagged_after_dismissal': return 'text-orange-700 bg-orange-50'
    default:                           return 'text-blue-700 bg-blue-50'
  }
}

export function statusLabel(s: string): string {
  switch (s) {
    case 'confirmed':                  return 'Confirmed'
    case 'dismissed':                  return 'Dismissed'
    case 'escalated':                  return 'Escalated'
    case 're_flagged_after_dismissal': return 'Re-flagged'
    default:                           return 'Pending Review'
  }
}

export function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1)
}
