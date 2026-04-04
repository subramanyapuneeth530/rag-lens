import { useState, useRef, useEffect } from 'react'

const s = {
  row: { display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 },
  card: {
    background: 'var(--surface)', border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)', padding: '16px 18px',
  },
  stat: { background: 'var(--surface2)', borderRadius: 'var(--radius)', padding: '10px 14px', flex: 1, minWidth: 100 },
  statVal: { fontSize: 22, fontWeight: 700, color: 'var(--text)' },
  statLabel: { fontSize: 11, color: 'var(--muted)', marginTop: 2 },
  stageTab: (active) => ({
    padding: '7px 14px', fontSize: 12, fontWeight: active ? 500 : 400,
    borderRadius: 20, cursor: 'pointer', border: 'none',
    background: active ? 'var(--blue)' : 'var(--surface2)',
    color: active ? '#fff' : 'var(--muted)',
    transition: 'all 0.15s',
  }),
  pre: {
    background: 'var(--surface2)', borderRadius: 'var(--radius)',
    padding: '12px 14px', fontFamily: 'var(--mono)', fontSize: 12,
    lineHeight: 1.7, overflowX: 'auto', whiteSpace: 'pre-wrap',
    wordBreak: 'break-all', color: 'var(--text)', maxHeight: 300, overflowY: 'auto',
  },
  input: {
    width: '100%', padding: '10px 14px', background: 'var(--surface2)',
    border: '1px solid var(--border)', borderRadius: 'var(--radius)',
    color: 'var(--text)', fontSize: 14, fontFamily: 'var(--font)',
    outline: 'none',
  },
  btn: {
    padding: '9px 20px', background: 'var(--blue)', color: '#fff',
    border: 'none', borderRadius: 'var(--radius)', cursor: 'pointer',
    fontSize: 13, fontWeight: 500,
  },
  conceptBox: (color = 'blue') => ({
    borderLeft: `3px solid var(--${color})`, padding: '10px 14px',
    background: `var(--${color}-dim)`, borderRadius: '0 var(--radius) var(--radius) 0',
    fontSize: 12, color: 'var(--muted)', lineHeight: 1.7, marginBottom: 14,
  }),
  vecBar: (val, max) => ({
    height: 10, borderRadius: 3, marginBottom: 2,
    width: `${Math.abs(val) / (max || 1) * 100}%`,
    background: val >= 0 ? 'var(--blue)' : 'var(--red)',
    transition: 'width 0.4s ease',
  }),
  simBar: (sim) => ({
    height: 22, borderRadius: 4, display: 'inline-block',
    width: `${Math.max(sim * 100, 8)}%`,
    background: sim > 0.7 ? 'var(--green)' : sim > 0.4 ? 'var(--blue)' : 'var(--muted)',
    transition: 'width 0.5s ease',
  }),
  token: (type) => ({
    display: 'inline-block', padding: '3px 7px', margin: '2px',
    borderRadius: 4, fontSize: 12, fontFamily: 'var(--mono)',
    background: type === 'special' ? 'var(--purple-dim)' : type === 'subword' ? 'var(--amber-dim)' : 'var(--surface2)',
    color: type === 'special' ? 'var(--purple)' : type === 'subword' ? 'var(--amber)' : 'var(--text)',
    border: `1px solid ${type === 'special' ? 'rgba(167,139,250,0.2)' : type === 'subword' ? 'rgba(251,191,36,0.2)' : 'var(--border)'}`,
    cursor: 'default',
    position: 'relative',
  }),
}

const STAGES = [
  '1. Ingest', '2. Chunk', '3. Tokenize',
  '4. Embed', '5. Vector space', '6. Retrieve', '7. Generate',
]

export default function Pipeline({ API, sources, pipelineData, setPipelineData }) {
  const [question, setQuestion] = useState('What is cosine similarity?')
  const [topK, setTopK] = useState(3)
  const [useLLM, setUseLLM] = useState(true)
  const [loading, setLoading] = useState(false)
  const [stage, setStage] = useState(0)
  const [streamedAnswer, setStreamedAnswer] = useState('')
  const [streamChunks, setStreamChunks] = useState([])
  const [streamPrompt, setStreamPrompt] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [hoveredToken, setHoveredToken] = useState(null)
  const evtRef = useRef(null)

  const runPipeline = async () => {
    if (!question.trim() || !sources.length) return
    setLoading(true)
    setPipelineData(null)
    setStreamedAnswer('')
    setStreamChunks([])
    setStreamPrompt('')
    setStage(0)

    try {
      const r = await fetch(`${API}/debug/pipeline`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, top_k: topK, use_llm: false }),
      })
      const d = await r.json()
      if (!r.ok) { alert(d.detail); return }
      setPipelineData(d)
      setStage(1)

      if (useLLM) {
        setStreaming(true)
        setStage(6)
        const evtSource = new EventSource(
          `${API}/query/stream?` + new URLSearchParams(),
        )
        if (evtRef.current) evtRef.current.close()

        const resp = await fetch(`${API}/query/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, top_k: topK, use_llm: true }),
        })

        const reader = resp.body.getReader()
        const decoder = new TextDecoder()
        let buf = ''
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buf += decoder.decode(value, { stream: true })
          const lines = buf.split('\n\n')
          buf = lines.pop()
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            try {
              const evt = JSON.parse(line.slice(6))
              if (evt.type === 'retrieval') {
                setStreamChunks(evt.chunks)
                setStreamPrompt(evt.prompt)
              } else if (evt.type === 'token') {
                setStreamedAnswer(prev => prev + evt.token)
              } else if (evt.type === 'done') {
                setStreaming(false)
              }
            } catch {}
          }
        }
        setStreaming(false)
      }
    } catch (e) {
      alert('Backend error: ' + e.message)
    } finally {
      setLoading(false)
    }
  }

  const d = pipelineData

  return (
    <div style={{ maxWidth: 900 }}>
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 13, color: 'var(--muted)', marginBottom: 12 }}>
          Runs the full pipeline and exposes every internal value — real token IDs, all 384 embedding dims, exact PCA coords, cosine scores, prompt string, timing.
        </div>
        <div style={{ display: 'flex', gap: 10, marginBottom: 10 }}>
          <input
            style={{ ...s.input, flex: 1 }}
            value={question}
            onChange={e => setQuestion(e.target.value)}
            placeholder="Ask a question about your document…"
            onKeyDown={e => e.key === 'Enter' && runPipeline()}
          />
          <select
            value={topK}
            onChange={e => setTopK(Number(e.target.value))}
            style={{ padding: '9px 12px', background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', color: 'var(--text)', fontSize: 13 }}
          >
            {[1,2,3,4,5].map(k => <option key={k} value={k}>top {k}</option>)}
          </select>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, color: 'var(--muted)', cursor: 'pointer' }}>
            <input type="checkbox" checked={useLLM} onChange={e => setUseLLM(e.target.checked)} />
            LLM
          </label>
          <button style={s.btn} onClick={runPipeline} disabled={loading || !sources.length}>
            {loading ? 'Running…' : 'Run pipeline'}
          </button>
        </div>
        {!sources.length && <div style={{ fontSize: 12, color: 'var(--red)' }}>Upload a document first.</div>}
      </div>

      {d && (
        <>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 16 }}>
            {STAGES.map((lbl, i) => (
              <button key={i} style={s.stageTab(stage === i)} onClick={() => setStage(i)}>{lbl}</button>
            ))}
          </div>

          {/* Stage 1: Ingest */}
          {stage === 0 && (
            <div style={s.card}>
              <div style={s.conceptBox('blue')}>
                <strong style={{ color: 'var(--blue)' }}>What happened:</strong> Your .txt was read, unicode-normalized, and loaded as a raw string. LangChain is not involved — just Python's built-in <code>open()</code>.
              </div>
              <div style={s.row}>
                <div style={s.stat}><div style={s.statVal}>{d.stage_1_ingest.total_chunks_in_store}</div><div style={s.statLabel}>Total chunks in ChromaDB</div></div>
                <div style={s.stat}><div style={s.statVal}>{d.stage_1_ingest.sources.length}</div><div style={s.statLabel}>Indexed sources</div></div>
              </div>
              <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 8 }}>Sources: {d.stage_1_ingest.sources.join(', ')}</div>
            </div>
          )}

          {/* Stage 2: Chunk */}
          {stage === 1 && (
            <div style={s.card}>
              <div style={s.conceptBox('amber')}>
                <strong style={{ color: 'var(--amber)' }}>RecursiveCharacterTextSplitter</strong> — tries \n\n first, then \n, then ". ", then " ". Overlap means the tail of one chunk appears at the head of the next, so no sentence falls completely between two chunks.
              </div>
              <div style={s.row}>
                <div style={s.stat}><div style={s.statVal}>{d.stage_2_chunking.retrieved_chunk_count}</div><div style={s.statLabel}>Chunks retrieved</div></div>
                <div style={s.stat}><div style={s.statVal}>{d.stage_2_chunking.top_k}</div><div style={s.statLabel}>top_k</div></div>
              </div>
              {d.stage_6_retrieval.retrieved_chunks.map((c, i) => (
                <div key={i} style={{ background: 'var(--surface2)', borderRadius: 'var(--radius)', padding: '10px 12px', marginBottom: 8, fontSize: 12 }}>
                  <div style={{ color: 'var(--muted)', marginBottom: 4 }}>
                    Chunk {c.chunk_index} · {c.token_count} tokens · chars {c.char_start}–{c.char_end}
                  </div>
                  <div style={{ fontFamily: 'var(--mono)', lineHeight: 1.6 }}>{c.text}</div>
                </div>
              ))}
            </div>
          )}

          {/* Stage 3: Tokenize */}
          {stage === 2 && (
            <div style={s.card}>
              <div style={s.conceptBox('purple')}>
                <strong style={{ color: 'var(--purple)' }}>BertTokenizer (WordPiece)</strong> — same tokenizer inside MiniLM. <span style={{ color: 'var(--purple)' }}>[CLS]/[SEP]</span> are special boundary tokens. <span style={{ color: 'var(--amber)' }}>##prefix</span> means subword continuation. Hover any token to see its vocabulary ID.
              </div>
              <div style={s.row}>
                <div style={s.stat}><div style={s.statVal}>{d.stage_3_tokenization.query.token_count}</div><div style={s.statLabel}>Query tokens</div></div>
                <div style={s.stat}><div style={s.statVal}>{d.stage_3_tokenization.query.word_count}</div><div style={s.statLabel}>Words</div></div>
                <div style={s.stat}><div style={s.statVal}>{d.stage_3_tokenization.query.tokens_per_word}</div><div style={s.statLabel}>Tokens/word</div></div>
                <div style={s.stat}>
                  <div style={{ ...s.statVal, color: d.stage_3_tokenization.query.within_model_limit ? 'var(--green)' : 'var(--red)' }}>
                    {d.stage_3_tokenization.query.within_model_limit ? '✓' : '✗'}
                  </div>
                  <div style={s.statLabel}>Within 512 limit</div>
                </div>
              </div>
              <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 8 }}>
                Vocab size: {d.stage_3_tokenization.query.vocab_size?.toLocaleString()} · {d.stage_3_tokenization.query.tokenizer_name}
              </div>
              <div style={{ lineHeight: 2 }}>
                {d.stage_3_tokenization.query.tokens.map((t, i) => {
                  const type = t.is_special ? 'special' : t.is_subword ? 'subword' : 'normal'
                  return (
                    <span
                      key={i}
                      style={s.token(type)}
                      onMouseEnter={() => setHoveredToken({ ...t, x: i })}
                      onMouseLeave={() => setHoveredToken(null)}
                      title={`id: ${t.id} · chars ${t.char_start}–${t.char_end}`}
                    >
                      {t.token}
                    </span>
                  )
                })}
              </div>
              {hoveredToken && (
                <div style={{ marginTop: 10, padding: '8px 12px', background: 'var(--surface2)', borderRadius: 'var(--radius)', fontSize: 12, fontFamily: 'var(--mono)' }}>
                  token: <strong>"{hoveredToken.token}"</strong> · id: <strong>{hoveredToken.id}</strong> · chars {hoveredToken.char_start}–{hoveredToken.char_end}
                  {hoveredToken.is_subword && ' · subword (continuation)'}
                  {hoveredToken.is_special && ' · special token'}
                </div>
              )}
              <div style={{ marginTop: 12, fontSize: 11, display: 'flex', gap: 12 }}>
                <span><span style={{ ...s.token('special'), display: 'inline-block' }}>[CLS]</span> special</span>
                <span><span style={{ ...s.token('subword'), display: 'inline-block' }}>##word</span> subword</span>
                <span><span style={{ ...s.token('normal'), display: 'inline-block' }}>word</span> full token</span>
              </div>
            </div>
          )}

          {/* Stage 4: Embed */}
          {stage === 3 && (
            <div style={s.card}>
              <div style={s.conceptBox('teal')}>
                <strong style={{ color: 'var(--teal)' }}>all-MiniLM-L6-v2</strong> reads your tokens through 6 transformer layers and outputs one 384-dimensional vector. L2-normalized so cosine similarity = dot product. Showing first 20 of 384 dims.
              </div>
              <div style={s.row}>
                <div style={s.stat}><div style={s.statVal}>{d.stage_4_embedding.dimensions}</div><div style={s.statLabel}>Dimensions</div></div>
                <div style={s.stat}><div style={s.statVal}>{d.stage_4_embedding.query_embed_ms}ms</div><div style={s.statLabel}>Embed time</div></div>
                <div style={s.stat}><div style={s.statVal}>{d.stage_4_embedding.query_embedding_norm}</div><div style={s.statLabel}>L2 norm</div></div>
              </div>
              <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 10 }}>Query vector — first 20 dimensions</div>
              {(() => {
                const preview = d.stage_4_embedding.query_embedding_preview
                const max = Math.max(...preview.map(Math.abs))
                return preview.map((v, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3 }}>
                    <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--muted)', width: 22, textAlign: 'right' }}>{i}</span>
                    <div style={{ flex: 1, height: 10, background: 'var(--surface2)', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={s.vecBar(v, max)} />
                    </div>
                    <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--muted)', width: 56, textAlign: 'right' }}>{v.toFixed(4)}</span>
                  </div>
                ))
              })()}
            </div>
          )}

          {/* Stage 5: Vector space */}
          {stage === 4 && (
            <div style={s.card}>
              <div style={s.conceptBox('purple')}>
                <strong style={{ color: 'var(--purple)' }}>PCA projection</strong> — sklearn compresses 384 dims → 3 for visualization. PC1/PC2/PC3 are the directions of maximum variance. Go to the 3D Vector Space view to explore this interactively.
              </div>
              <div style={s.row}>
                <div style={s.stat}><div style={s.statVal}>{d.stage_5_vector_space.pca_metadata.n_embeddings}</div><div style={s.statLabel}>Total chunks</div></div>
                <div style={s.stat}><div style={s.statVal}>{d.stage_5_vector_space.pca_metadata.original_dims}</div><div style={s.statLabel}>Original dims</div></div>
                <div style={s.stat}><div style={s.statVal}>{(d.stage_5_vector_space.pca_metadata.total_variance_explained * 100).toFixed(1)}%</div><div style={s.statLabel}>Variance explained</div></div>
              </div>
              <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 10 }}>
                Per-component: PC1={((d.stage_5_vector_space.pca_metadata.explained_variance?.[0] || 0)*100).toFixed(1)}% · PC2={((d.stage_5_vector_space.pca_metadata.explained_variance?.[1] || 0)*100).toFixed(1)}% · PC3={((d.stage_5_vector_space.pca_metadata.explained_variance?.[2] || 0)*100).toFixed(1)}%
              </div>
              {d.stage_5_vector_space.query_point && (
                <div style={{ fontFamily: 'var(--mono)', fontSize: 12, padding: '10px 12px', background: 'var(--surface2)', borderRadius: 'var(--radius)' }}>
                  Query position in PCA space:<br />
                  x={d.stage_5_vector_space.query_point.x} · y={d.stage_5_vector_space.query_point.y} · z={d.stage_5_vector_space.query_point.z}
                </div>
              )}
              <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 12 }}>
                Chunk 3D positions (first 5):
              </div>
              {d.stage_5_vector_space.chunk_points.slice(0, 5).map((p, i) => (
                <div key={i} style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--muted)', marginTop: 4 }}>
                  [{p.chunk_index}] ({p.x}, {p.y}, {p.z}) — "{p.text_preview.slice(0, 50)}…"
                </div>
              ))}
            </div>
          )}

          {/* Stage 6: Retrieve */}
          {stage === 5 && (
            <div style={s.card}>
              <div style={s.conceptBox('green')}>
                <strong style={{ color: 'var(--green)' }}>Cosine similarity</strong> = 1 − ChromaDB distance. The query vector is compared to every stored chunk vector. top_k={topK} chunks are returned. Score closer to 1.0 = more semantically similar.
              </div>
              {d.stage_6_retrieval.retrieved_chunks.map((c, i) => (
                <div key={i} style={{ marginBottom: 14 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                    <div style={{ width: 20, height: 20, borderRadius: '50%', background: 'var(--blue)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 700, flexShrink: 0 }}>{i+1}</div>
                    <div style={{ flex: 1 }}>
                      <div style={s.simBar(c.similarity)} />
                    </div>
                    <span style={{ fontFamily: 'var(--mono)', fontSize: 13, fontWeight: 600, color: c.similarity > 0.7 ? 'var(--green)' : c.similarity > 0.4 ? 'var(--blue)' : 'var(--muted)' }}>
                      {c.similarity.toFixed(4)}
                    </span>
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 4 }}>
                    {c.source} · chunk {c.chunk_index} · {c.token_count} tokens · norm {c.embedding_norm}
                  </div>
                  <div style={{ fontFamily: 'var(--mono)', fontSize: 12, background: 'var(--surface2)', padding: '8px 10px', borderRadius: 'var(--radius)', lineHeight: 1.6 }}>
                    {c.text}
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 4 }}>
                    Embedding preview (first 20 dims): [{c.embedding_preview?.slice(0,5).map(v => v.toFixed(3)).join(', ')}…]
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Stage 7: Generate */}
          {stage === 6 && (
            <div style={s.card}>
              <div style={s.conceptBox('amber')}>
                <strong style={{ color: 'var(--amber)' }}>Prompt construction ("stuff" strategy)</strong> — top_k chunks are concatenated into a context block and inserted into the prompt template. Llama 3.2 never sees the rest of your document — only what retrieval selected. Temperature = 0.1.
              </div>
              <div style={{ fontSize: 12, fontWeight: 500, color: 'var(--muted)', marginBottom: 6 }}>Exact prompt sent to Llama 3.2</div>
              <pre style={{ ...s.pre, marginBottom: 16 }}>
                {streamPrompt || d.stage_7_generation.prompt}
              </pre>
              <div style={s.row}>
                <div style={s.stat}><div style={s.statVal}>{d.stage_7_generation.model}</div><div style={s.statLabel}>Model</div></div>
                <div style={s.stat}><div style={s.statVal}>{d.stage_7_generation.temperature}</div><div style={s.statLabel}>Temperature</div></div>
                {d.stage_7_generation.generation_ms > 0 && (
                  <div style={s.stat}><div style={s.statVal}>{(d.stage_7_generation.generation_ms/1000).toFixed(1)}s</div><div style={s.statLabel}>Generation time</div></div>
                )}
              </div>
              <div style={{ fontSize: 12, fontWeight: 500, color: 'var(--muted)', margin: '12px 0 6px' }}>Answer</div>
              <div style={{ ...s.pre, borderLeft: '3px solid var(--green)', background: 'var(--green-dim)', color: 'var(--text)', minHeight: 60 }}>
                {streamedAnswer || d.stage_7_generation.answer}
                {streaming && <span style={{ opacity: 0.5 }}>▌</span>}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
