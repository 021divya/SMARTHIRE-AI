"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navbar() {
  const pathname = usePathname();
  
  // Don't render navbar on admin page
  if (pathname.startsWith('/admin')) {
    return null;
  }
  
  return (
    <nav className="bg-gradient-to-r from-blue-700 to-blue-900 shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex-shrink-0 flex items-center">
            <Link href="/" className="text-white text-xl font-bold">
              Candidate Portal
            </Link>
          </div>
          <div className="flex items-center">
            <Link 
              href="/admin" 
              className="ml-4 px-4 py-2 bg-white text-blue-800 font-medium rounded-md hover:bg-blue-50 transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              Admin
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}
