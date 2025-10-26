import { Link, useLocation } from 'react-router-dom'
import { ChartPieIcon, BeakerIcon } from '@heroicons/react/24/outline'

const navigation = [
  { name: 'Inference', href: '/', icon: ChartPieIcon },
  { name: 'Experiments', href: '/experiments', icon: BeakerIcon },
]

function classNames(...classes: string[]) {
  return classes.filter(Boolean).join(' ')
}

export default function Sidebar() {
  const location = useLocation()

  return (
    <div className="relative flex grow flex-col gap-y-5 overflow-y-auto border-r border-gray-200 bg-white px-6 dark:border-white/10 dark:bg-gray-900 dark:before:pointer-events-none dark:before:absolute dark:before:inset-0 dark:before:bg-black/10">
      <div className="relative flex h-16 shrink-0 items-center">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">
          AnimalDet
        </h2>
      </div>
      <nav className="relative flex flex-1 flex-col">
        <ul role="list" className="flex flex-1 flex-col gap-y-7">
          <li>
            <ul role="list" className="-mx-2 space-y-1">
              {navigation.map((item) => {
                const current = location.pathname === item.href
                return (
                  <li key={item.name}>
                    <Link
                      to={item.href}
                      className={classNames(
                        current
                          ? 'bg-gray-50 text-indigo-600 dark:bg-white/5 dark:text-white'
                          : 'text-gray-700 hover:bg-gray-50 hover:text-indigo-600 dark:text-gray-400 dark:hover:bg-white/5 dark:hover:text-white',
                        'group flex gap-x-3 rounded-md p-2 text-sm/6 font-semibold',
                      )}
                    >
                      <item.icon
                        aria-hidden="true"
                        className={classNames(
                          current
                            ? 'text-indigo-600 dark:text-white'
                            : 'text-gray-400 group-hover:text-indigo-600 dark:group-hover:text-white',
                          'size-6 shrink-0',
                        )}
                      />
                      {item.name}
                    </Link>
                  </li>
                )
              })}
            </ul>
          </li>
        </ul>
      </nav>
    </div>
  )
}
