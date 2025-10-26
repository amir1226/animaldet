import { Outlet } from 'react-router-dom'
import Sidebar from '@animaldet/app/components/layout/Sidebar'

export default function Layout() {
  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <div className="hidden lg:fixed lg:inset-y-0 lg:left-0 lg:z-50 lg:flex lg:w-72 lg:flex-col">
        <Sidebar />
      </div>

      {/* Main content */}
        <main className="py-10 px-4 sm:px-6 lg:px-8">
          <Outlet />
        </main>
    </div>
  )
}
