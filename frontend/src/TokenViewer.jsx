import { useState } from 'react'

const s = {
  input: {
    width: '100%', padding: '10px 14px', background: 'var(--surface2)',
    border: '1px solid var(--border)', borderRadius: 'var(--radius)',
    color: 'var(--text)', fontSize: 14, fontFamily: 'var(--mono)', outline: 'none',
    resize: 'vertical', minHeight: 80,
  },
  btn: {
    padding: '9px 20px', background: 'var(--blue)', color: '#fff',
    border: 'none', borderRadius: 'var(--radius)', cursor: 'pointer',
    fontSize: 13, fontWeight: 500, marginTop: 8,
  },
  token: (type) => ({
    display: 'inline-flex', flexDirection: 'column', alignItems: 'center',
    padding: '6px 8px', margin: '3px', borderRadius: 6, fontSize: 13,
    fontFamily: 'var(--mono)', cursor: 'default',
    background: type === 'special' ? 'var(--purple-dim)' : type === 'subword' ? 'var(--amber-dim)' : 'var(--surface2)',
    color: type === 'special' ? 'var(--purple)' : type === 'subword' ? 'var(--amber)' : 'var(--text)',
    border: `1px solid ${type === 'special' ? 'rgba(167,139,250,0.25)' : type === 'subword' ? 'rgba(251,191,36,0.25)' : 'var(--border)'}`,
    minWidth: 36, textAlign: 'center', transition: 'transform 0.1s',
    position: 'relative',
  }),
  tokenId: { fontSize: 9, color: 'var(--muted)', marginTop: 3, fontFamily: 'var(--mono)' },
  stat: { background: 'var(--surface2)', borderRadius: 'var(--radius)', padding: '10px 14px', flex: 1, minWidth: 90 },
  statVal: { fontSize: 22, fontWeight: 700, color: 'var(--text)' },
  statLabel: { fontSize: 11, color: 'var(--muted)', marginTop: 2 },
}

const EXAMPLES = [
  'What is cosine similarity?',
  'ChromaDB stores vector embeddings persistently.',
  'The all-MiniLM-L6-v2 model produces 384-dimensional vectors.',
  'RAG stands for Retrieval Augmented Generation.',
  'Llama 3.2 runs locally via Ollama with temperature 0.1.',
]

export default function TokenViewer({ API }) {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [hoveredIdx, setHoveredIdx] = useState(null)
  const [showIds, setShowIds] = useState(true)

  const tokenize = async (inputText) => {
    const t = (inputText ?? text).trim()
    if (!t) return
    setLoading(true)
    try {
      const r = await fetch(`${API}/debug/tokenize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: t }),
      })
      const d = await r.json()
      if (!r.ok) { alert(d.detail); return }
      setResult(d)
    } catch (e) {
      alert('Backend error: ' + e.message)
    } finally {
      setLoading(false)
    }
  }

  const hov = result?.tokens?.[hoveredIdx]

  return (
    <div style={{ maxWidth: 860 }}>
      <div style={{ fontSize: 13, color: 'var(--muted)', marginBottom: 16 }}>
        Runs the real <strong style={{ color: 'var(--text)' }}>BertTokenizer (WordPiece)</strong> — the exact tokenizer inside all-MiniLM-L6-v2. Shows every token string, vocabulary ID, subword flags, and character offsets.
      </div>

      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 6 }}>Try an example:</div>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {EXAMPLES.map((ex, i) => (
            <button
              key={i}
              onClick={() => { setText(ex); tokenize(ex) }}
              style={{ padding: '4px 10px', background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 20, color: 'var(--muted)', fontSize: 12, cursor: 'pointer' }}
            >
              {ex.slice(0, 36)}{ex.length > 36 ? '…' : ''}
            </button>
          ))}
        </div>
      </div>

      <textarea
        style={s.input}
        value={text}
        onChange={e => setText(e.target.value)}
        placeholder="Type or paste any text to tokenize with the real BertTokenizer…"
        onKeyDown={e => e.key === 'Enter' && e.ctrlKey && tokenize()}
      />
      <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
        <button style={s.btn} onClick={() => tokenize()} disabled={loading || !text.trim()}>
          {loading ? 'Tokenizing…' : 'Tokenize'}
        </button>
        <span style={{ fontSize: 12, color: 'var(--muted)' }}>Ctrl+Enter</span>
        {result && (
          <label style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--muted)', cursor: 'pointer' }}>
            <input type="checkbox" checked={showIds} onChange={e => setShowIds(e.target.checked)} />
            Show IDs
          </label>
        )}
      </div>

      {result && (
        <div style={{ marginTop: 20 }}>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 16 }}>
            <div style={s.stat}>
              <div style={s.statVal}>{result.token_count}</div>
              <div style={s.statLabel}>Tokens</div>
            </div>
            <div style={s.stat}>
              <div style={s.statVal}>{result.word_count}</div>
              <div style={s.statLabel}>Words</div>
            </div>
            <div style={s.stat}>
              <div style={s.statVal}>{result.tokens_per_word}</div>
              <div style={s.statLabel}>Tokens/word</div>
            </div>
            <div style={s.stat}>
              <div style={{ ...s.statVal, color: result.within_model_limit ? 'var(--green)' : 'var(--red)' }}>
                {result.token_count} / {result.model_limit}
              </div>
              <div style={s.statLabel}>Model limit</div>
            </div>
            <div style={s.stat}>
              <div style={s.statVal}>{result.vocab_size?.toLocaleString()}</div>
              <div style={s.statLabel}>Vocab size</div>
            </div>
          </div>

          {!result.within_model_limit && (
            <div style={{ padding: '10px 14px', background: 'rgba(248,113,113,0.1)', border: '1px solid rgba(248,113,113,0.2)', borderRadius: 'var(--radius)', color: 'var(--red)', fontSize: 12, marginBottom: 14 }}>
              This text exceeds the 512-token limit. MiniLM will truncate it during embedding, losing the tail. This is why chunk size matters.
            </div>
          )}

          <div style={{ padding: '10px 14px', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', marginBottom: 14, fontSize: 12, lineHeight: 1.8, flexWrap: 'wrap', display: 'flex', gap: 0 }}>
            {result.tokens.map((tok, i) => {
              const type = tok.is_special ? 'special' : tok.is_subword ? 'subword' : 'normal'
              return (
                <span
                  key={i}
                  style={{
                    ...s.token(type),
                    transform: hoveredIdx === i ? 'scale(1.15)' : 'scale(1)',
                    boxShadow: hoveredIdx === i ? '0 0 0 2px var(--blue)' : 'none',
                  }}
                  onMouseEnter={() => setHoveredIdx(i)}
                  onMouseLeave={() => setHoveredIdx(null)}
                >
                  <span>{tok.token}</span>
                  {showIds && <span style={s.tokenId}>{tok.id}</span>}
                </span>
              )
            })}
          </div>

          {hov && (
            <div style={{ padding: '12px 14px', background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', fontSize: 12, fontFamily: 'var(--mono)', marginBottom: 14 }}>
              <span style={{ color: 'var(--blue)' }}>token</span>: "<strong>{hov.token}</strong>" &nbsp;·&nbsp;
              <span style={{ color: 'var(--blue)' }}>id</span>: {hov.id} &nbsp;·&nbsp;
              <span style={{ color: 'var(--blue)' }}>chars</span>: {hov.char_start}–{hov.char_end}
              {hov.is_subword && <span style={{ color: 'var(--amber)' }}> · subword continuation</span>}
              {hov.is_special && <span style={{ color: 'var(--purple)' }}> · special token (not embedded as real text)</span>}
            </div>
          )}

          <div style={{ display: 'flex', gap: 16, fontSize: 11 }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ ...s.token('special'), display: 'inline-block', padding: '2px 8px' }}>[CLS]</span>
              Special — boundary tokens, always id 101/102
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ ...s.token('subword'), display: 'inline-block', padding: '2px 8px' }}>##ing</span>
              Subword — continuation of previous token
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ ...s.token('normal'), display: 'inline-block', padding: '2px 8px' }}>word</span>
              Full token
            </span>
          </div>

          <div style={{ marginTop: 20, padding: '14px', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius-lg)' }}>
            <div style={{ fontSize: 12, fontWeight: 500, color: 'var(--amber)', marginBottom: 8 }}>Why this matters for RAG</div>
            <div style={{ fontSize: 12, color: 'var(--muted)', lineHeight: 1.8 }}>
              MiniLM can only process <strong style={{ color: 'var(--text)' }}>512 tokens</strong> per chunk. If a chunk exceeds this, the end gets silently truncated before embedding — the embedding won't represent the full chunk. This is why the ingest step counts tokens per chunk and why chunk size is set in characters rather than words (characters are a cheaper proxy).
              <br /><br />
              English averages ~1.3 tokens per word, so 500 chars ≈ 80 words ≈ 104 tokens — comfortably within 512. But code, technical terms, or non-English text can be much denser.
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
