import { Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Environment, ContactShadows } from '@react-three/drei'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { useLoader } from '@react-three/fiber'

function Model({ url }: { url: string }) {
  const gltf = useLoader(GLTFLoader, url)
  return <primitive object={gltf.scene} />
}

interface Props {
  url: string
}

export function Preview3D({ url }: Props) {
  return (
    <div style={{ width: '100%', height: '100%' }}>
      <Suspense fallback={
        <div className="preview-empty">Loading 3D model...</div>
      }>
        <Canvas camera={{ position: [0, 1, 3], fov: 50 }}>
          <ambientLight intensity={0.5} />
          <directionalLight position={[5, 5, 5]} intensity={1} castShadow />
          <Suspense fallback={null}>
            <Model url={url} />
          </Suspense>
          <ContactShadows position={[0, -1, 0]} opacity={0.5} scale={10} blur={2} />
          <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
          <Environment preset="studio" />
        </Canvas>
      </Suspense>
    </div>
  )
}
