import { useState } from 'react'
import { Outlet } from 'react-router-dom'
import Sidebar from '@animaldet/app/components/layout/Sidebar'

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(true)

  return (
    <div className="flex h-screen">
      {/* Hamburger Toggle Button */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className={`fixed top-4 z-50 p-2 bg-gray-800 text-white rounded hover:bg-gray-700 transition-all duration-300 ${sidebarOpen ? 'left-[17rem]' : 'left-4'}`}
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          {sidebarOpen ? (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          ) : (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          )}
        </svg>
      </button>

      {/* Sidebar */}
      {sidebarOpen && (
        <div className="fixed inset-y-0 left-0 z-40 w-72">
          <Sidebar />
        </div>
      )}

      {/* Main content */}
      <main className="flex-1 w-full">
        <Outlet />
      </main>
    </div>
  )
}
