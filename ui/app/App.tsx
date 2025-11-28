import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from '@animaldet/app/components/layout/Layout'
import Inference from '@animaldet/app/pages/Inference'
import ExperimentComparison from '@animaldet/app/experiments/pages/ExperimentComparison'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Inference />} />
          <Route path="/experiments" element={<ExperimentComparison />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
