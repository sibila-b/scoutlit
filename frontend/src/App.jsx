import { useState } from 'react'

export default function App() {
  const [topic, setTopic] = useState('')

  function handleSubmit(e) {
    e.preventDefault()
    // feature implementation pending
    console.log('Search topic:', topic)
  }

  return (
    <main>
      <h1>ScoutLit</h1>
      <p>AI-powered academic literature assistant</p>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="Enter a research topic…"
          aria-label="Research topic"
        />
        <button type="submit">Search</button>
      </form>
    </main>
  )
}
