import Link from 'next/link';
import CandidateForm from '../components/CandidateForm';

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold text-blue-800 mb-4">Welcome to the Candidate Portal</h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Submit your application using the form below to be considered for our open positions.
          </p>
        </div>
        
        <div className="bg-white shadow-xl rounded-lg overflow-hidden">
          <div className="bg-gradient-to-r from-blue-700 to-blue-900 py-6 px-8">
            <h2 className="text-2xl font-semibold text-white">Application Form</h2>
            <p className="text-blue-100">Fill out all required fields</p>
          </div>
          <div className="p-1">
            <CandidateForm />
          </div>
        </div>
      </div>
    </main>
  );
}