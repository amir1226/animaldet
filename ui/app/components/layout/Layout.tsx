import { Outlet } from 'react-router-dom'
import Sidebar from '@animaldet/app/components/layout/Sidebar'

export default function Layout() {
  return (
    <div className="flex h-screen overflow-hidden p-4 gap-4 bg-gray-50 dark:bg-gray-950">
      {/* Sidebar */}
      <div className="w-72 flex-shrink-0">
        <Sidebar />
      </div>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto bg-white dark:bg-gray-900 rounded-lg">
        <Outlet />
      </main>
    </div>
  )
}
