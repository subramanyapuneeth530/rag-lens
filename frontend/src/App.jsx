import { useState, useCallback } from 'react'
import Pipeline from './Pipeline.jsx'
import VectorSpace from './VectorSpace.jsx'
import TokenViewer from './TokenViewer.jsx'

const API = '/api'

const s = {
  shell: { display: 'flex', height: '100vh', overflow: 'hidden' },
  sidebar: {
    width: 220, flexShrink: 0, background: 'var(--surface)',
    borderRight: '1px solid var(--border)', display: 'flex',
    flexDirection: 'column', padding: '20px 0',
  },
  logo: {
    padding: '0 20px 20px', fontSize: 18, fontWeight: 700,
    color: 'var(--blue)', letterSpacing: '-0.5px',
    borderBottom: '1px solid var(--border)', marginBottom: 8,
  },
  logoSub: { fontSize: 11, color: 'var(--muted)', fontWeight: 400, marginTop: 2 },
  navItem: (active) => ({
    display: 'flex', alignItems: 'center', gap: 10, padding: '9px 20px',
    cursor: 'pointer', fontSize: 13, fontWeight: active ? 500 : 400,
    color: active ? 'var(--blue)' : 'var(--muted)',
    background: active ? 'rgba(74,158,255,0.08)' : 'transparent',
    borderLeft: active ? '2px solid var(--blue)' : '2px solid transparent',
    transition: 'all 0.15s',
  }),
  main: { flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' },
  topbar: {
    padding: '12px 24px', borderBottom: '1px solid var(--border)',
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    background: 'var(--surface)', flexShrink: 0,
  },
  content: { flex: 1, overflow: 'auto', padding: 24 },
  uploadZone: {
    border: '1.5px dashed var(--border)', borderRadius: 'var(--radius-lg)',
    padding: '32px 24px', textAlign: 'center', cursor: 'pointer',
    transition: 'border-color 0.2s',
  },
  fileChip: {
    display: 'inline-flex', alignItems: 'center', gap: 8,
    background: 'var(--green-dim)', color: 'var(--green)',
    border: '1px solid rgba(74,222,128,0.2)', borderRadius: 20,
    padding: '4px 12px', fontSize: 12, fontWeight: 500,
  },
  sourcesList: { display: 'flex', flexDirection: 'column', gap: 4, marginTop: 8 },
  sourceRow: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '8px 12px', background: 'var(--surface2)',
    borderRadius: 'var(--radius)', border: '1px solid var(--border)',
  },
  btn: (variant = 'primary') => ({
    padding: '7px 16px', borderRadius: 'var(--radius)', border: 'none',
    cursor: 'pointer', fontSize: 13, fontWeight: 500,
    background: variant === 'primary' ? 'var(--blue)' : variant === 'danger' ? 'rgba(248,113,113,0.15)' : 'var(--surface2)',
    color: variant === 'primary' ? '#fff' : variant === 'danger' ? 'var(--red)' : 'var(--muted)',
    transition: 'opacity 0.15s',
  }),
  badge: (color = 'blue') => ({
    display: 'inline-block', padding: '2px 8px', borderRadius: 20,
    fontSize: 11, fontWeight: 500,
    background: `var(--${color}-dim)`, color: `var(--${color})`,
  }),
}

const views = [
  { id: 'upload', label: 'Documents', icon: '◻' },
  { id: 'pipeline', label: 'Pipeline explorer', icon: '◈' },
  { id: 'vectorspace', label: '3D vector space', icon: '⬡' },
  { id: 'tokens', label: 'Token viewer', icon: '◇' },
]

export default function App() {
  const [view, setView] = useState('upload')
  const [sources, setSources] = useState([])
  const [uploading, setUploading] = useState(false)
  const [uploadMsg, setUploadMsg] = useState('')
  const [pipelineData, setPipelineData] = useState(null)
  const [health, setHealth] = useState(null)

  const checkHealth = useCallback(async () => {
    try {
      const r = await fetch(`${API}/health`)
      const d = await r.json()
      setHealth(d.status === 'ok')
    } catch {
      setHealth(false)
    }
  }, [])

  const loadSources = useCallback(async () => {
    try {
      const r = await fetch(`${API}/sources`)
      const d = await r.json()
      setSources(d.sources || [])
    } catch {}
  }, [])

  useState(() => { checkHealth(); loadSources() }, [])

  const handleUpload = async (file) => {
    if (!file || !file.name.endsWith('.txt')) {
      setUploadMsg('Only .txt files are supported.')
      return
    }
    setUploading(true)
    setUploadMsg('')
    const fd = new FormData()
    fd.append('file', file)
    try {
      const r = await fetch(`${API}/ingest`, { method: 'POST', body: fd })
      const d = await r.json()
      if (!r.ok) { setUploadMsg(d.detail || 'Upload failed.'); return }
      setUploadMsg(`Indexed ${d.chunks} chunks · ${d.total_tokens} tokens · avg ${d.avg_tokens_per_chunk} tok/chunk`)
      loadSources()
    } catch (e) {
      setUploadMsg('Cannot reach backend. Is uvicorn running on port 8000?')
    } finally {
      setUploading(false)
    }
  }

  const handleDelete = async (source) => {
    await fetch(`${API}/source/${encodeURIComponent(source)}`, { method: 'DELETE' })
    loadSources()
  }

  return (
    <div style={s.shell}>
      <aside style={s.sidebar}>
        <div style={s.logo}>
          rag-lens
          <div style={s.logoSub}>full pipeline transparency</div>
        </div>
        {views.map(v => (
          <div key={v.id} style={s.navItem(view === v.id)} onClick={() => setView(v.id)}>
            <span style={{ fontSize: 16 }}>{v.icon}</span>
            {v.label}
          </div>
        ))}
        <div style={{ marginTop: 'auto', padding: '16px 20px', borderTop: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 6 }}>backend</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: health === true ? 'var(--green)' : health === false ? 'var(--red)' : 'var(--muted)' }} />
            <span style={{ color: 'var(--muted)' }}>
              {health === true ? 'connected' : health === false ? 'offline' : 'checking…'}
            </span>
          </div>
          <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 8 }}>
            {sources.length} document{sources.length !== 1 ? 's' : ''} indexed
          </div>
        </div>
      </aside>

      <main style={s.main}>
        <div style={s.topbar}>
          <div style={{ fontWeight: 500 }}>{views.find(v => v.id === view)?.label}</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <span style={s.badge('blue')}>MiniLM · 384d</span>
            <span style={s.badge('purple')}>Llama 3.2</span>
            <span style={s.badge('teal')}>ChromaDB</span>
          </div>
        </div>

        <div style={s.content}>
          {view === 'upload' && (
            <UploadView
              sources={sources}
              uploading={uploading}
              uploadMsg={uploadMsg}
              onUpload={handleUpload}
              onDelete={handleDelete}
              s={s}
            />
          )}
          {view === 'pipeline' && (
            <Pipeline
              API={API}
              sources={sources}
              pipelineData={pipelineData}
              setPipelineData={setPipelineData}
            />
          )}
          {view === 'vectorspace' && (
            <VectorSpace API={API} sources={sources} />
          )}
          {view === 'tokens' && (
            <TokenViewer API={API} />
          )}
        </div>
      </main>
    </div>
  )
}

function UploadView({ sources, uploading, uploadMsg, onUpload, onDelete, s }) {
  const onDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer?.files?.[0]
    if (file) onUpload(file)
  }
  return (
    <div style={{ maxWidth: 600 }}>
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 16, fontWeight: 500, marginBottom: 4 }}>Upload a document</div>
        <div style={{ color: 'var(--muted)', fontSize: 13 }}>
          Plain text files only. This becomes the knowledge base the RAG pipeline searches.
        </div>
      </div>

      <div
        style={s.uploadZone}
        onDrop={onDrop}
        onDragOver={(e) => e.preventDefault()}
        onClick={() => document.getElementById('file-input').click()}
      >
        <div style={{ fontSize: 28, marginBottom: 8, color: 'var(--muted)' }}>+</div>
        <div style={{ color: 'var(--muted)', fontSize: 13 }}>Drop a .txt file here, or click to browse</div>
        <input
          id="file-input" type="file" accept=".txt" style={{ display: 'none' }}
          onChange={(e) => onUpload(e.target.files?.[0])}
        />
      </div>

      {uploading && (
        <div style={{ marginTop: 12, color: 'var(--blue)', fontSize: 13 }}>
          Chunking → embedding with MiniLM → storing in ChromaDB…
        </div>
      )}
      {uploadMsg && (
        <div style={{ marginTop: 12, padding: '10px 14px', background: 'var(--surface2)', borderRadius: 'var(--radius)', fontSize: 13, color: uploadMsg.includes('Cannot') ? 'var(--red)' : 'var(--green)' }}>
          {uploadMsg}
        </div>
      )}

      {sources.length > 0 && (
        <div style={{ marginTop: 28 }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 10 }}>Indexed documents</div>
          <div style={s.sourcesList}>
            {sources.map(src => (
              <div key={src.source} style={s.sourceRow}>
                <div>
                  <div style={{ fontSize: 13 }}>{src.source}</div>
                  <div style={{ fontSize: 11, color: 'var(--muted)' }}>{src.chunk_count} chunks</div>
                </div>
                <button style={s.btn('danger')} onClick={() => onDelete(src.source)}>Remove</button>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ marginTop: 28, padding: '16px', background: 'var(--surface)', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border)' }}>
        <div style={{ fontSize: 12, fontWeight: 500, color: 'var(--amber)', marginBottom: 8 }}>What happens when you upload</div>
        {[
          ['1. Read', 'UTF-8 text is read and unicode-normalized'],
          ['2. Chunk', 'Split on paragraph → sentence → word boundaries (RecursiveCharacterSplitter logic, no LangChain)'],
          ['3. Tokenize', 'Real BertTokenizer counts tokens per chunk to detect model limit violations'],
          ['4. Embed', 'all-MiniLM-L6-v2 encodes each chunk → 384-dim L2-normalized vector'],
          ['5. Store', 'ChromaDB saves text + embedding + metadata (source, char offsets, token count)'],
        ].map(([step, desc]) => (
          <div key={step} style={{ display: 'flex', gap: 12, marginBottom: 8, fontSize: 12 }}>
            <span style={{ color: 'var(--amber)', fontWeight: 500, minWidth: 60, fontFamily: 'var(--mono)' }}>{step}</span>
            <span style={{ color: 'var(--muted)' }}>{desc}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
