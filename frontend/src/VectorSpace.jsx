import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'

const COLORS = [
  0x4a9eff, 0x4ade80, 0xfbbf24, 0xa78bfa,
  0xf87171, 0x2dd4bf, 0xfb923c, 0xe879f9,
]

export default function VectorSpace({ API, sources }) {
  const mountRef = useRef(null)
  const sceneRef = useRef({})
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [hovered, setHovered] = useState(null)
  const [selected, setSelected] = useState(null)
  const [sourceFilter, setSourceFilter] = useState('')

  const load = async () => {
    setLoading(true)
    setHovered(null)
    setSelected(null)
    try {
      const url = `${API}/debug/embeddings` + (sourceFilter ? `?source=${encodeURIComponent(sourceFilter)}` : '')
      const r = await fetch(url)
      const d = await r.json()
      setData(d)
    } catch (e) {
      alert('Cannot load embeddings: ' + e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!data?.chunks?.length || !mountRef.current) return
    initScene(data.chunks)
    return () => cleanupScene()
  }, [data])

  const initScene = (chunks) => {
    const el = mountRef.current
    const W = el.clientWidth, H = el.clientHeight

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(W, H)
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.setClearColor(0x08080f, 1)
    el.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(60, W / H, 0.01, 100)
    camera.position.set(0, 0, 3.5)

    // Axes
    const axMat = new THREE.LineBasicMaterial({ color: 0x333355, linewidth: 1 })
    ;[
      [[-1.5,0,0],[1.5,0,0]],
      [[0,-1.5,0],[0,1.5,0]],
      [[0,0,-1.5],[0,0,1.5]],
    ].forEach(([a, b]) => {
      const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a), new THREE.Vector3(...b)])
      scene.add(new THREE.Line(geo, axMat))
    })

    // Axis labels (sprites)
    const makeLabel = (text, pos, color) => {
      const canvas = document.createElement('canvas')
      canvas.width = 128; canvas.height = 48
      const ctx = canvas.getContext('2d')
      ctx.fillStyle = color
      ctx.font = 'bold 28px monospace'
      ctx.textAlign = 'center'
      ctx.fillText(text, 64, 34)
      const tex = new THREE.CanvasTexture(canvas)
      const mat = new THREE.SpriteMaterial({ map: tex, transparent: true })
      const sprite = new THREE.Sprite(mat)
      sprite.position.set(...pos)
      sprite.scale.set(0.4, 0.15, 1)
      scene.add(sprite)
    }
    makeLabel('PC1', [1.7, 0, 0], '#4a9eff')
    makeLabel('PC2', [0, 1.7, 0], '#4ade80')
    makeLabel('PC3', [0, 0, 1.7], '#a78bfa')

    // Grid dot at origin
    const ogeo = new THREE.SphereGeometry(0.025, 8, 8)
    const omat = new THREE.MeshBasicMaterial({ color: 0x444466 })
    scene.add(new THREE.Mesh(ogeo, omat))

    // Chunk points
    const sourceNames = [...new Set(chunks.map(c => c.source))]
    const meshes = []

    chunks.forEach((chunk, i) => {
      const colorIdx = sourceNames.indexOf(chunk.source) % COLORS.length
      const color = COLORS[colorIdx]

      const geo = new THREE.SphereGeometry(0.045, 12, 12)
      const mat = new THREE.MeshBasicMaterial({ color })
      const mesh = new THREE.Mesh(geo, mat)
      mesh.position.set(chunk.x, chunk.y, chunk.z)
      mesh.userData = { chunk, index: i, baseColor: color }
      scene.add(mesh)
      meshes.push(mesh)
    })

    sceneRef.current = { renderer, scene, camera, meshes, chunks, sourceNames }

    // Orbit controls (manual)
    let isDragging = false, prevMouse = { x: 0, y: 0 }
    const spherical = { theta: 0, phi: Math.PI / 2, r: 3.5 }

    const updateCamera = () => {
      camera.position.set(
        spherical.r * Math.sin(spherical.phi) * Math.sin(spherical.theta),
        spherical.r * Math.cos(spherical.phi),
        spherical.r * Math.sin(spherical.phi) * Math.cos(spherical.theta),
      )
      camera.lookAt(0, 0, 0)
    }

    el.addEventListener('mousedown', e => { isDragging = true; prevMouse = { x: e.clientX, y: e.clientY } })
    window.addEventListener('mouseup', () => { isDragging = false })
    el.addEventListener('mousemove', e => {
      if (isDragging) {
        spherical.theta -= (e.clientX - prevMouse.x) * 0.008
        spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi - (e.clientY - prevMouse.y) * 0.008))
        prevMouse = { x: e.clientX, y: e.clientY }
        updateCamera()
      } else {
        // Raycast hover
        const rect = el.getBoundingClientRect()
        const mouse = new THREE.Vector2(
          ((e.clientX - rect.left) / rect.width) * 2 - 1,
          -((e.clientY - rect.top) / rect.height) * 2 + 1,
        )
        const raycaster = new THREE.Raycaster()
        raycaster.setFromCamera(mouse, camera)
        const hits = raycaster.intersectObjects(meshes)
        if (hits.length) {
          setHovered(hits[0].object.userData.chunk)
          el.style.cursor = 'pointer'
        } else {
          setHovered(null)
          el.style.cursor = 'grab'
        }
      }
    })
    el.addEventListener('wheel', e => {
      e.preventDefault()
      spherical.r = Math.max(1.5, Math.min(8, spherical.r + e.deltaY * 0.005))
      updateCamera()
    }, { passive: false })
    el.addEventListener('click', e => {
      const rect = el.getBoundingClientRect()
      const mouse = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1,
      )
      const raycaster = new THREE.Raycaster()
      raycaster.setFromCamera(mouse, camera)
      const hits = raycaster.intersectObjects(meshes)
      if (hits.length) {
        const chunk = hits[0].object.userData.chunk
        setSelected(prev => prev?.id === chunk.id ? null : chunk)
      }
    })

    updateCamera()

    let animId
    const animate = () => {
      animId = requestAnimationFrame(animate)
      // Update point sizes based on selection
      meshes.forEach(m => {
        const chunk = m.userData.chunk
        const isSelected = selected?.id === chunk.id
        const scale = isSelected ? 1.8 : 1
        m.scale.setScalar(scale)
      })
      renderer.render(scene, camera)
    }
    animate()

    const onResize = () => {
      const W = el.clientWidth, H = el.clientHeight
      camera.aspect = W / H
      camera.updateProjectionMatrix()
      renderer.setSize(W, H)
    }
    window.addEventListener('resize', onResize)
    sceneRef.current.animId = animId
    sceneRef.current.onResize = onResize
  }

  const cleanupScene = () => {
    const { renderer, animId, onResize } = sceneRef.current
    if (animId) cancelAnimationFrame(animId)
    if (renderer) {
      renderer.dispose()
      renderer.domElement.remove()
    }
    if (onResize) window.removeEventListener('resize', onResize)
  }

  const sourceNames = data ? [...new Set(data.chunks.map(c => c.source))] : []

  return (
    <div style={{ maxWidth: 900 }}>
      <div style={{ marginBottom: 16, display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
        <div style={{ fontSize: 13, color: 'var(--muted)', flex: 1 }}>
          Real PCA projection of MiniLM embeddings. Each point = one chunk. Position = semantic location in 384-dim space compressed to 3D. Drag to rotate · scroll to zoom · click to inspect.
        </div>
        {sources.length > 1 && (
          <select
            value={sourceFilter}
            onChange={e => setSourceFilter(e.target.value)}
            style={{ padding: '7px 12px', background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', color: 'var(--text)', fontSize: 13 }}
          >
            <option value="">All sources</option>
            {sources.map(s => <option key={s.source} value={s.source}>{s.source}</option>)}
          </select>
        )}
        <button
          onClick={load}
          disabled={loading || !sources.length}
          style={{ padding: '7px 18px', background: 'var(--blue)', color: '#fff', border: 'none', borderRadius: 'var(--radius)', cursor: 'pointer', fontSize: 13, fontWeight: 500 }}
        >
          {loading ? 'Loading…' : data ? 'Refresh' : 'Load embeddings'}
        </button>
      </div>

      {!sources.length && (
        <div style={{ color: 'var(--muted)', fontSize: 13 }}>Upload a document first.</div>
      )}

      {data && (
        <>
          <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap', fontSize: 12 }}>
            <span style={{ color: 'var(--muted)' }}>{data.total} chunks</span>
            {data.pca && <>
              <span style={{ color: 'var(--muted)' }}>·</span>
              <span style={{ color: 'var(--muted)' }}>{(data.pca.total_variance_explained * 100).toFixed(1)}% variance explained by 3 PCs</span>
              <span style={{ color: 'var(--muted)' }}>·</span>
              <span style={{ color: 'var(--muted)' }}>PC1={((data.pca.explained_variance?.[0]||0)*100).toFixed(1)}% PC2={((data.pca.explained_variance?.[1]||0)*100).toFixed(1)}% PC3={((data.pca.explained_variance?.[2]||0)*100).toFixed(1)}%</span>
            </>}
          </div>

          <div style={{ display: 'flex', gap: 12 }}>
            <div
              ref={mountRef}
              style={{ flex: 1, height: 480, borderRadius: 'var(--radius-lg)', overflow: 'hidden', border: '1px solid var(--border)', cursor: 'grab', background: '#08080f' }}
            />
            <div style={{ width: 240, display: 'flex', flexDirection: 'column', gap: 10 }}>
              {/* Legend */}
              <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius-lg)', padding: '12px 14px' }}>
                <div style={{ fontSize: 11, fontWeight: 500, color: 'var(--muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Sources</div>
                {sourceNames.map((name, i) => (
                  <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, fontSize: 12 }}>
                    <div style={{ width: 10, height: 10, borderRadius: '50%', background: `#${COLORS[i % COLORS.length].toString(16).padStart(6,'0')}`, flexShrink: 0 }} />
                    <span style={{ color: 'var(--muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={name}>{name}</span>
                  </div>
                ))}
                <div style={{ marginTop: 10, fontSize: 11, color: 'var(--muted)', borderTop: '1px solid var(--border)', paddingTop: 8 }}>
                  <div>x = PC1 ({((data.pca?.explained_variance?.[0]||0)*100).toFixed(1)}%)</div>
                  <div>y = PC2 ({((data.pca?.explained_variance?.[1]||0)*100).toFixed(1)}%)</div>
                  <div>z = PC3 ({((data.pca?.explained_variance?.[2]||0)*100).toFixed(1)}%)</div>
                </div>
              </div>

              {/* Hover / selected info */}
              {(hovered || selected) && (() => {
                const chunk = selected || hovered
                return (
                  <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius-lg)', padding: '12px 14px', fontSize: 12 }}>
                    <div style={{ fontWeight: 500, marginBottom: 6, color: 'var(--text)' }}>
                      {selected ? 'Selected' : 'Hovered'} chunk
                    </div>
                    <div style={{ color: 'var(--muted)', marginBottom: 4, fontSize: 11 }}>
                      {chunk.source} · index {chunk.chunk_index} · {chunk.token_count} tokens
                    </div>
                    <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--muted)', marginBottom: 8 }}>
                      ({chunk.x}, {chunk.y}, {chunk.z})
                    </div>
                    <div style={{ lineHeight: 1.6, color: 'var(--text)', fontSize: 12 }}>
                      {chunk.text_preview || chunk.text}
                    </div>
                    {chunk.embedding_preview && (
                      <div style={{ marginTop: 8, fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--muted)' }}>
                        emb[0:5]: [{chunk.embedding_preview.slice(0,5).map(v=>v.toFixed(3)).join(', ')}…]
                      </div>
                    )}
                  </div>
                )
              })()}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
