"use client";

import { useState, useEffect } from "react";
import ResumeSearch from "@/components/ResumeSearch";
import Link from "next/link";

// Define interfaces for your data types
interface Candidate {
  id: number;
  name: string;
  email: string;
  skills: string;
  linkedin?: string;
  experience: string;
  resumePath: string;
  resumeText?: string;
  createdAt: string;
}

interface Job {
  id: number;
  title: string;
  description: string;
  requirements: string;
  createdAt: string;
}

interface Evaluation {
  candidateId: string;
  candidateName: string;
  similarity: string;
  score: number;
  feedback: string;
  recommendations?: string;
}

// Modal component for displaying full text
function Modal({
  isOpen,
  title,
  content,
  onClose,
}: {
  isOpen: boolean;
  title: string;
  content: string;
  onClose: () => void;
}) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 z-50 flex justify-center items-center p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[80vh] flex flex-col">
        <div className="p-4 border-b border-gray-200 flex justify-between items-center">
          <h2 className="text-xl font-semibold text-gray-800">{title}</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 focus:outline-none"
            aria-label="Close"
          >
            <svg
              className="h-6 w-6"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>
        <div className="p-6 overflow-y-auto whitespace-pre-wrap">{content}</div>
        <div className="p-4 border-t border-gray-200 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

export default function AdminPage() {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJob, setSelectedJob] = useState("");
  const [evaluations, setEvaluations] = useState<Evaluation[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showJobForm, setShowJobForm] = useState(false);
  const [newJob, setNewJob] = useState({
    title: "",
    description: "",
    requirements: "",
  });
  const [jobFormLoading, setJobFormLoading] = useState(false);
  const [jobFormSuccess, setJobFormSuccess] = useState(false);

  // Modal state
  const [modalOpen, setModalOpen] = useState(false);
  const [modalTitle, setModalTitle] = useState("");
  const [modalContent, setModalContent] = useState("");

  // Function to open modal with content
  const openModal = (title: string, content: string) => {
    setModalTitle(title);
    setModalContent(content);
    setModalOpen(true);
  };

  useEffect(() => {
    // Fetch candidates and jobs on component mount
    setIsLoading(true);
    setError(null);

    const fetchCandidates = fetch("/api/candidates")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch candidates");
        return res.json();
      })
      .catch((err) => {
        console.error("Error fetching candidates:", err);
        setError("Failed to load candidates");
        return [];
      });

    const fetchJobs = fetch("/api/jobs")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch jobs");
        return res.json();
      })
      .catch((err) => {
        console.error("Error fetching jobs:", err);
        setError("Failed to load jobs");
        return [];
      });

    Promise.all([fetchCandidates, fetchJobs])
      .then(([candidatesData, jobsData]) => {
        setCandidates(candidatesData || []);
        setJobs(jobsData || []);

        // After setting jobs, check for previously selected job in localStorage
        const savedJobId = localStorage.getItem("selectedJobId");
        if (
          savedJobId &&
          (jobsData as Job[]).some(
            (job: Job) => job.id.toString() === savedJobId
          )
        ) {
          setSelectedJob(savedJobId);

          // Load saved evaluations for this job
          const savedEvaluations: string | null = localStorage.getItem(
            `evaluations_${savedJobId}`
          );
          if (savedEvaluations) {
            try {
              const parsedEvaluations: Evaluation[] =
                JSON.parse(savedEvaluations);
              setEvaluations(parsedEvaluations);
            } catch (e: unknown) {
              console.error("Error parsing saved evaluations:", e);
            }
          }
        }
      })
      .finally(() => setIsLoading(false));
  }, []);

  const evaluateCandidates = async () => {
    if (!selectedJob) return;

    // Save the selected job ID
    localStorage.setItem("selectedJobId", selectedJob);

    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jobId: parseInt(selectedJob, 10) }),
      });

      const data = await response.json();

      if (!response.ok) {
        const errorMessage =
          data.details || data.error || "Failed to evaluate candidates";
        throw new Error(errorMessage);
      }

      // Save the evaluations in state
      setEvaluations(data.evaluations || []);

      // Also save them in localStorage
      localStorage.setItem(
        `evaluations_${selectedJob}`,
        JSON.stringify(data.evaluations || [])
      );
    } catch (error) {
      console.error("Evaluation error:", error);
      setError(
        error instanceof Error ? error.message : "Failed to evaluate candidates"
      );
    } finally {
      setIsLoading(false);
    }
  };

  // Add an effect to update localStorage when selected job changes
  useEffect(() => {
    if (selectedJob) {
      localStorage.setItem("selectedJobId", selectedJob);
    }
  }, [selectedJob]);

  const handleCreateJob = async (e: React.FormEvent) => {
    e.preventDefault();
    setJobFormLoading(true);
    setJobFormSuccess(false);

    try {
      const response = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newJob),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Failed to create job");
      }

      // Add the new job to the local state
      setJobs([...jobs, data.job]);

      // Reset the form
      setNewJob({
        title: "",
        description: "",
        requirements: "",
      });

      setJobFormSuccess(true);
      setTimeout(() => setJobFormSuccess(false), 3000);
    } catch (error) {
      console.error("Error creating job:", error);
      setError(error instanceof Error ? error.message : "Failed to create job");
    } finally {
      setJobFormLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Modal for displaying full text */}
      <Modal
        isOpen={modalOpen}
        title={modalTitle}
        content={modalContent}
        onClose={() => setModalOpen(false)}
      />

      {/* Dashboard Header */}
      {/* <header className="bg-gradient-to-r from-blue-600 to-indigo-700 shadow-lg">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-white">Admin Dashboard</h1>
        </div>
      </header> */}

      <header className="bg-gradient-to-r from-blue-700 to-blue-900 shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div>
              <h1 className="text-3xl font-semibold text-white">
                Admin Dashboard
              </h1>
            </div>

            <div className="flex-shrink-0 font-bold">
              <Link
                href="/"
                className="text-white text-lg hover:text-blue-200 transition-colors"
              >
                Candidate Portal
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        {/* Error Alert */}
        {error && (
          <div className="mb-6 rounded-lg bg-red-50 p-4 border-l-4 border-red-500 shadow-sm">
            <div className="flex items-center">
              <svg
                className="w-6 h-6 text-red-500 mr-3"
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
              </svg>
              <span className="text-red-700 font-medium">{error}</span>
            </div>
          </div>
        )}

        {/* Job Management and Evaluation Section */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 mb-8">
          {/* Left Column - Job Management & Evaluation */}
          <div className="lg:col-span-5 space-y-8">
            {/* Job Management Card */}
            <section className="bg-white rounded-lg shadow overflow-hidden">
              <div className="p-6 border-b border-gray-200 flex justify-between items-center">
                <h2 className="text-xl font-semibold text-gray-800">
                  Job Management
                </h2>
                <button
                  onClick={() => setShowJobForm(!showJobForm)}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                    showJobForm
                      ? "bg-gray-200 text-gray-800 hover:bg-gray-300"
                      : "bg-green-600 text-white hover:bg-green-700"
                  }`}
                >
                  {showJobForm ? "Hide Form" : "Create New Job"}
                </button>
              </div>

              {showJobForm && (
                <div className="p-6 border-b border-gray-100">
                  {jobFormSuccess && (
                    <div className="mb-4 rounded-md bg-green-50 p-3 border border-green-200">
                      <div className="flex items-center">
                        <svg
                          className="w-5 h-5 text-green-500 mr-2"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                            clipRule="evenodd"
                          ></path>
                        </svg>
                        <span className="text-green-700 font-medium">
                          Job created successfully!
                        </span>
                      </div>
                    </div>
                  )}

                  <form onSubmit={handleCreateJob} className="space-y-5">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Job Title
                      </label>
                      <input
                        type="text"
                        value={newJob.title}
                        onChange={(e) =>
                          setNewJob({ ...newJob, title: e.target.value })
                        }
                        className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 py-2 px-3"
                        required
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Description
                      </label>
                      <textarea
                        value={newJob.description}
                        onChange={(e) =>
                          setNewJob({ ...newJob, description: e.target.value })
                        }
                        className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 py-2 px-3"
                        rows={3}
                        required
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Requirements
                      </label>
                      <textarea
                        value={newJob.requirements}
                        onChange={(e) =>
                          setNewJob({ ...newJob, requirements: e.target.value })
                        }
                        className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 py-2 px-3"
                        rows={3}
                        required
                      />
                    </div>

                    <div className="flex justify-end">
                      <button
                        type="submit"
                        disabled={jobFormLoading}
                        className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-blue-400 disabled:cursor-not-allowed"
                      >
                        {jobFormLoading ? (
                          <>
                            <svg
                              className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                              xmlns="http://www.w3.org/2000/svg"
                              fill="none"
                              viewBox="0 0 24 24"
                            >
                              <circle
                                className="opacity-25"
                                cx="12"
                                cy="12"
                                r="10"
                                stroke="currentColor"
                                strokeWidth="4"
                              ></circle>
                              <path
                                className="opacity-75"
                                fill="currentColor"
                                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                              ></path>
                            </svg>
                            Creating...
                          </>
                        ) : (
                          "Create Job"
                        )}
                      </button>
                    </div>
                  </form>
                </div>
              )}

              <div className="px-6 py-4">
                <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-3">
                  Available Jobs
                </h3>
                <div className="max-h-60 overflow-y-auto bg-gray-50 rounded-md">
                  {jobs.length === 0 ? (
                    <p className="text-gray-500 text-center py-4">
                      No jobs available.
                    </p>
                  ) : (
                    <ul className="divide-y divide-gray-200">
                      {jobs.map((job) => (
                        <li
                          key={job.id}
                          className="p-3 hover:bg-gray-100 transition-colors"
                        >
                          <p className="font-medium text-blue-700">
                            {job.title}
                          </p>
                          <p className="text-xs text-gray-500 mt-1">
                            Created:{" "}
                            {new Date(job.createdAt).toLocaleDateString()}
                          </p>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            </section>

            {/* Evaluate Candidates Card */}
            <section className="bg-white rounded-lg shadow overflow-hidden">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-xl font-semibold text-gray-800">
                  Evaluate Candidates
                </h2>
              </div>

              <div className="p-6">
                <div className="flex flex-col md:flex-row gap-4 items-center">
                  <div className="w-full md:w-2/3">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Select Job
                    </label>
                    <select
                      value={selectedJob}
                      onChange={(e) => {
                        const newJobId = e.target.value;
                        setSelectedJob(newJobId);

                        if (newJobId !== selectedJob) {
                          setEvaluations([]);
                        }

                        if (newJobId) {
                          const savedEvaluations = localStorage.getItem(
                            `evaluations_${newJobId}`
                          );
                          if (savedEvaluations) {
                            try {
                              setEvaluations(JSON.parse(savedEvaluations));
                            } catch (e) {
                              console.error(
                                "Error parsing saved evaluations:",
                                e
                              );
                            }
                          }
                        }
                      }}
                      className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                      disabled={isLoading}
                    >
                      <option value="">Select a job...</option>
                      {jobs.map((job) => (
                        <option key={job.id} value={job.id.toString()}>
                          {job.title}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="w-full md:w-1/3 md:self-end">
                    <button
                      onClick={evaluateCandidates}
                      disabled={!selectedJob || isLoading}
                      className={`w-full inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-indigo-400 disabled:cursor-not-allowed transition-colors ${
                        isLoading ? "animate-pulse" : ""
                      }`}
                    >
                      {isLoading ? (
                        <>
                          <svg
                            className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                          >
                            <circle
                              className="opacity-25"
                              cx="12"
                              cy="12"
                              r="10"
                              stroke="currentColor"
                              strokeWidth="4"
                            ></circle>
                            <path
                              className="opacity-75"
                              fill="currentColor"
                              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                            ></path>
                          </svg>
                          Evaluating...
                        </>
                      ) : (
                        "Evaluate Candidates"
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </section>
          </div>

          {/* Right Column - Evaluation Results */}
          <div className="lg:col-span-7">
            {/* Evaluation Results */}
            {evaluations.length > 0 && (
              <section className="bg-white rounded-lg shadow overflow-hidden">
                <div className="p-6 border-b border-gray-200 bg-gradient-to-r from-indigo-50 to-blue-50">
                  <h2 className="text-xl font-semibold text-gray-800">
                    Evaluation Results
                  </h2>
                  <p className="text-sm text-gray-600 mt-1">
                    Top candidates for the selected job position
                  </p>
                </div>

                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th
                          scope="col"
                          className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                        >
                          Rank
                        </th>
                        <th
                          scope="col"
                          className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                        >
                          Candidate
                        </th>
                        <th
                          scope="col"
                          className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                        >
                          Fit Score
                        </th>
                        <th
                          scope="col"
                          className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                        >
                          AI Feedback
                        </th>
                        <th
                          scope="col"
                          className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                        >
                          Recommendations
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {evaluations.map((evaluation, index) => (
                        <tr
                          key={evaluation.candidateId}
                          className={`${
                            index === 0
                              ? "bg-green-50"
                              : index < 3
                              ? "bg-blue-50"
                              : ""
                          } hover:bg-gray-50 transition-colors`}
                        >
                          <td className="px-6 py-4 whitespace-nowrap">
                            {index < 3 ? (
                              <span
                                className={`inline-flex items-center justify-center w-6 h-6 rounded-full ${
                                  index === 0
                                    ? "bg-yellow-400"
                                    : index === 1
                                    ? "bg-gray-300"
                                    : "bg-amber-600"
                                } text-white font-bold`}
                              >
                                {index + 1}
                              </span>
                            ) : (
                              <span className="text-gray-500">{index + 1}</span>
                            )}
                          </td>
                          <td className="px-6 py-4">
                            <div className="font-medium text-gray-900">
                              {evaluation.candidateName}
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <div className="flex items-center">
                              <div className="w-full bg-gray-200 rounded-full h-2.5 mr-2 w-24">
                                <div
                                  className={`h-2.5 rounded-full ${
                                    evaluation.score > 80
                                      ? "bg-green-500"
                                      : evaluation.score > 60
                                      ? "bg-blue-500"
                                      : evaluation.score > 40
                                      ? "bg-yellow-500"
                                      : "bg-red-500"
                                  }`}
                                  style={{ width: `${evaluation.score}%` }}
                                ></div>
                              </div>
                              <span className="text-sm font-medium">
                                {evaluation.score}%
                              </span>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <div className="bg-gray-50 p-3 rounded border border-gray-100">
                              <div className="text-sm text-gray-900 line-clamp-3">
                                {evaluation.feedback}
                              </div>
                              <button
                                onClick={() =>
                                  openModal(
                                    `Feedback for ${evaluation.candidateName}`,
                                    evaluation.feedback
                                  )
                                }
                                className="text-sm text-blue-600 hover:text-blue-800 mt-2 font-medium inline-flex items-center"
                              >
                                <svg
                                  className="w-4 h-4 mr-1"
                                  fill="none"
                                  stroke="currentColor"
                                  viewBox="0 0 24 24"
                                >
                                  <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth="2"
                                    d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                                  />
                                  <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth="2"
                                    d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                                  />
                                </svg>
                                Read full feedback
                              </button>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <div className="bg-gray-50 p-3 rounded border border-gray-100">
                              <div className="text-sm text-gray-900 line-clamp-3">
                                {evaluation.recommendations ||
                                  "No specific recommendations"}
                              </div>
                              {evaluation.recommendations && (
                                <button
                                  onClick={() =>
                                    openModal(
                                      `Recommendations for ${evaluation.candidateName}`,
                                      evaluation.recommendations || ""
                                    )
                                  }
                                  className="text-sm text-blue-600 hover:text-blue-800 mt-2 font-medium inline-flex items-center"
                                >
                                  <svg
                                    className="w-4 h-4 mr-1"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                  >
                                    <path
                                      strokeLinecap="round"
                                      strokeLinejoin="round"
                                      strokeWidth="2"
                                      d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                                    />
                                    <path
                                      strokeLinecap="round"
                                      strokeLinejoin="round"
                                      strokeWidth="2"
                                      d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                                    />
                                  </svg>
                                  Read full recommendations
                                </button>
                              )}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            )}
          </div>
        </div>

        {/* Resume Search Section - Given full width for more space */}
        <section className="bg-white rounded-lg shadow overflow-hidden mb-8">
          <div className="p-6 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-indigo-50">
            <h2 className="text-xl font-semibold text-gray-800">
              Resume Search
            </h2>
            <p className="text-sm text-gray-600 mt-1">
              Search for candidates by skills, experience, or job requirements
            </p>
          </div>
          <div className="p-6">
            <ResumeSearch openModal={openModal} />
          </div>
        </section>

        {/* All Candidates Section - modify this part of your existing code */}
        <section className="bg-white rounded-lg shadow overflow-hidden">
          <div className="p-6 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-green-50">
            <h2 className="text-xl font-semibold text-gray-800">
              All Candidates
            </h2>
            <p className="text-sm text-gray-600 mt-1">
              Total: {candidates.length}{" "}
              {candidates.length === 1 ? "candidate" : "candidates"}
            </p>
          </div>

          {candidates.length === 0 ? (
            <div className="p-6 text-center">
              <p className="text-gray-500">No candidates available.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th
                      scope="col"
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    >
                      Name
                    </th>
                    <th
                      scope="col"
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    >
                      Email
                    </th>
                    <th
                      scope="col"
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    >
                      Skills
                    </th>
                    <th
                      scope="col"
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    >
                      Resume Preview
                    </th>
                    <th
                      scope="col"
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    >
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {candidates.map((candidate) => (
                    <tr
                      key={candidate.id}
                      className="hover:bg-gray-50 transition-colors"
                    >
                      <td className="px-6 py-4">
                        <div className="font-medium text-gray-900">
                          {candidate.name}
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="text-sm text-gray-900">
                          {candidate.email}
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="text-sm">
                          {candidate.skills.split(",").map((skill, i) => (
                            <span
                              key={i}
                              className="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-0.5 rounded-full mr-1 mb-1"
                            >
                              {skill.trim()}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="text-xs text-gray-900 bg-gray-50 p-3 rounded">
                          {candidate.resumeText ? (
                            <>
                              <span className="font-medium text-gray-500 block mb-1">
                                {Math.ceil(candidate.resumeText.length / 1000)}k
                                chars
                              </span>
                              <div className="whitespace-pre-wrap break-words line-clamp-3">
                                {candidate.resumeText.substring(0, 150)}...
                              </div>
                              <button
                                onClick={() =>
                                  openModal(
                                    `Resume: ${candidate.name}`,
                                    candidate.resumeText || ""
                                  )
                                }
                                className="text-sm text-blue-600 hover:text-blue-800 mt-2 font-medium inline-flex items-center"
                              >
                                <svg
                                  className="w-4 h-4 mr-1"
                                  fill="none"
                                  stroke="currentColor"
                                  viewBox="0 0 24 24"
                                >
                                  <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth="2"
                                    d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                                  />
                                  <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth="2"
                                    d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                                  />
                                </svg>
                                View full resume
                              </button>
                            </>
                          ) : (
                            <span className="text-red-500">
                              No resume text available
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <a
                          href={candidate.resumePath}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md text-indigo-700 bg-indigo-100 hover:bg-indigo-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                        >
                          <svg
                            className="w-4 h-4 mr-1"
                            fill="currentColor"
                            viewBox="0 0 20 20"
                            xmlns="http://www.w3.org/2000/svg"
                          >
                            <path d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z"></path>
                          </svg>
                          Download Resume
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
