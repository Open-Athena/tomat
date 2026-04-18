import { useEffect, useState } from 'react'

export type ThemeMode = 'light' | 'dark' | 'auto'

export function useThemeToggle() {
  const [mode, setMode] = useState<ThemeMode>(
    () => (localStorage.getItem('theme') as ThemeMode | null) ?? 'auto',
  )
  useEffect(() => {
    const html = document.documentElement
    if (mode === 'auto') html.removeAttribute('data-theme')
    else html.setAttribute('data-theme', mode)
    localStorage.setItem('theme', mode)
  }, [mode])
  return [mode, setMode] as const
}

/** Three-state toggle: light → auto → dark → light. */
export function ThemeToggle({ className = 'theme-toggle' }: { className?: string }) {
  const [mode, setMode] = useThemeToggle()
  // Use a functional setter so rapid double-clicks compose correctly.
  const cycle = () => setMode(m => (m === 'dark' ? 'light' : m === 'light' ? 'auto' : 'dark'))
  return (
    <button
      className={className}
      onClick={cycle}
      aria-label="theme"
      title={`theme: ${mode}`}
    >
      {mode === 'dark' ? '🌙' : mode === 'light' ? '☀️' : '🖥️'}
    </button>
  )
}

/** Hoverlabel styling that matches the current theme — Plotly's default is
 * pale-bg + dark-text and disappears in dark mode. */
export function themedHoverlabel(isDark: boolean) {
  return isDark
    ? { bgcolor: '#1d2125', bordercolor: '#2f343a', font: { color: '#e4e4e4' as const } }
    : { bgcolor: '#ffffff', bordercolor: '#d5d5d5', font: { color: '#222' as const } }
}
