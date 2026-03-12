"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";

const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB in bytes
const ALLOWED_FILE_TYPE = "application/pdf";

interface CandidateFormData {
  name: string;
  email: string;
  linkedin: string;
  skills: string;
  experience: string;
  resume: FileList;
}

interface APIResponse {
  message?: string;
  error?: string;
  details?: string;
}

export default function CandidateForm() {
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<CandidateFormData>();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const onSubmit = async (data: CandidateFormData): Promise<void> => {
    setIsSubmitting(true);
    setMessage("");
    setError("");

    try {
      const formData = new FormData();
      formData.append("name", data.name);
      formData.append("email", data.email);
      formData.append("linkedin", data.linkedin || "");
      formData.append("skills", data.skills);
      formData.append("experience", data.experience);
      
      // Append the file
      if (data.resume && data.resume[0]) {
        formData.append("resume", data.resume[0]);
      }

      // Make the API request
      const response = await fetch("/api/candidates", {
        method: "POST",
        body: formData,
        // Don't set Content-Type header - browser will set it with boundary for multipart/form-data
      });

      const result: APIResponse = await response.json();

      if (!response.ok) {
        throw new Error(result.error || result.details || "Failed to submit application");
      }

      // Success
      setMessage(result.message || "Application submitted successfully!");
      reset(); // Clear the form fields
    } catch (err) {
      console.error("Error submitting form:", err);
      setError(err instanceof Error ? err.message : "Failed to submit application");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto p-8 bg-gradient-to-br from-white to-blue-50 rounded-xl shadow-xl border border-blue-100">
      <div className="text-center mb-10">
        <h2 className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-700 mb-2">Submit Your Application</h2>
        <p className="text-gray-600 text-lg">Join our team and be part of something great</p>
        <div className="mt-4 w-24 h-1 bg-gradient-to-r from-blue-500 to-indigo-600 mx-auto rounded-full"></div>
      </div>
      
      {message && (
        <div className="mb-8 p-5 bg-green-50 border border-green-200 text-green-700 rounded-lg shadow-inner flex items-center animate-fadeIn transition-all duration-300">
          <svg className="h-6 w-6 mr-3 text-green-500" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"></path>
          </svg>
          <span className="font-medium">{message}</span>
        </div>
      )}
      
      {error && (
        <div className="mb-8 p-5 bg-red-50 border border-red-200 text-red-700 rounded-lg shadow-inner flex items-center animate-fadeIn transition-all duration-300">
          <svg className="h-6 w-6 mr-3 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span className="font-medium">{error}</span>
        </div>
      )}
      
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-7">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-7">
          {/* Full Name */}
          <div className="group transition-all duration-300">
            <label className="block text-sm font-medium text-gray-700 mb-2 group-focus-within:text-blue-700 transition-colors" htmlFor="name">
              Full Name <span className="text-blue-600">*</span>
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-400">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
                </svg>
              </div>
              <input
                id="name"
                type="text"
                placeholder="John Doe"
                className="w-full pl-10 pr-4 py-3.5 rounded-lg border border-gray-300 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all hover:border-blue-300"
                {...register("name", { required: "Name is required" })}
              />
            </div>
            {errors.name && (
              <p className="text-red-500 text-sm mt-2 flex items-center">
                <svg className="h-4 w-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
                {errors.name.message}
              </p>
            )}
          </div>
          
          {/* Email */}
          <div className="group transition-all duration-300">
            <label className="block text-sm font-medium text-gray-700 mb-2 group-focus-within:text-blue-700 transition-colors" htmlFor="email">
              Email <span className="text-blue-600">*</span>
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-400">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path d="M2.003 5.884L10 9.882l7.997-3.998A2 2 0 0016 4H4a2 2 0 00-1.997 1.884z" />
                  <path d="M18 8.118l-8 4-8-4V14a2 2 0 002 2h12a2 2 0 002-2V8.118z" />
                </svg>
              </div>
              <input
                id="email"
                type="email"
                placeholder="your.email@example.com"
                className="w-full pl-10 pr-4 py-3.5 rounded-lg border border-gray-300 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all hover:border-blue-300"
                {...register("email", { 
                  required: "Email is required",
                  pattern: {
                    value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                    message: "Invalid email address"
                  }
                })}
              />
            </div>
            {errors.email && (
              <p className="text-red-500 text-sm mt-2 flex items-center">
                <svg className="h-4 w-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
                {errors.email.message}
              </p>
            )}
          </div>
        </div>
        
        {/* LinkedIn */}
        <div className="group transition-all duration-300">
          <label className="block text-sm font-medium text-gray-700 mb-2 group-focus-within:text-blue-700 transition-colors" htmlFor="linkedin">
            LinkedIn URL <span className="text-gray-500 text-xs">(optional)</span>
          </label>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-400">
              <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.338 16.338H13.67V12.16c0-1.005-.019-2.299-1.4-2.299-1.4 0-1.616 1.095-1.616 2.223v4.254H8.084V8h2.543v1.164h.037c.36-.683 1.241-1.4 2.556-1.4 2.733 0 3.236 1.799 3.236 4.137v4.437zM5.5 6.833a1.56 1.56 0 11.017-3.123 1.56 1.56 0 01-.017 3.123zm1.335 9.505H4.166V8H6.835v8.338zM18 0H2C.9 0 0 .9 0 2v16c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V2c0-1.1-.9-2-2-2z" clipRule="evenodd" />
              </svg>
            </div>
            <input
              id="linkedin"
              type="text"
              placeholder="https://linkedin.com/in/username"
              className="w-full pl-10 pr-4 py-3.5 rounded-lg border border-gray-300 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all hover:border-blue-300"
              {...register("linkedin")}
            />
          </div>
        </div>
        
        {/* Skills */}
        <div className="group transition-all duration-300">
          <label className="block text-sm font-medium text-gray-700 mb-2 group-focus-within:text-blue-700 transition-colors" htmlFor="skills">
            Skills <span className="text-blue-600">*</span>
          </label>
          <div className="relative">
            <div className="absolute top-3.5 left-3 flex items-center pointer-events-none text-gray-400">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path d="M7 3a1 1 0 000 2h6a1 1 0 100-2H7zM4 7a1 1 0 011-1h10a1 1 0 110 2H5a1 1 0 01-1-1zM2 11a2 2 0 012-2h12a2 2 0 012 2v4a2 2 0 01-2 2H4a2 2 0 01-2-2v-4z" />
              </svg>
            </div>
            <textarea
              id="skills"
              className="w-full pl-10 pr-4 py-3.5 rounded-lg border border-gray-300 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all hover:border-blue-300"
              rows={3}
              placeholder="JavaScript, React, Node.js, CSS, etc. (separate with commas)"
              {...register("skills", { required: "Skills are required" })}
            ></textarea>
          </div>
          {errors.skills && (
            <p className="text-red-500 text-sm mt-2 flex items-center">
              <svg className="h-4 w-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              {errors.skills.message}
            </p>
          )}
        </div>
        
        {/* Experience */}
        <div className="group transition-all duration-300">
          <label className="block text-sm font-medium text-gray-700 mb-2 group-focus-within:text-blue-700 transition-colors" htmlFor="experience">
            Experience <span className="text-blue-600">*</span>
          </label>
          <div className="relative">
            <div className="absolute top-3.5 left-3 flex items-center pointer-events-none text-gray-400">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M6 6V5a3 3 0 013-3h2a3 3 0 013 3v1h2a2 2 0 012 2v3.57A22.952 22.952 0 0110 13a22.95 22.95 0 01-8-1.43V8a2 2 0 012-2h2zm2-1a1 1 0 011-1h2a1 1 0 011 1v1H8V5zm1 5a1 1 0 011-1h.01a1 1 0 110 2H10a1 1 0 01-1-1z" clipRule="evenodd" />
                <path d="M2 13.692V16a2 2 0 002 2h12a2 2 0 002-2v-2.308A24.974 24.974 0 0110 15c-2.796 0-5.487-.46-8-1.308z" />
              </svg>
            </div>
            <textarea
              id="experience"
              className="w-full pl-10 pr-4 py-3.5 rounded-lg border border-gray-300 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all hover:border-blue-300"
              rows={4}
              placeholder="Describe your relevant experience, including years of experience, previous roles, and significant achievements..."
              {...register("experience", { required: "Experience is required" })}
            ></textarea>
          </div>
          {errors.experience && (
            <p className="text-red-500 text-sm mt-2 flex items-center">
              <svg className="h-4 w-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              {errors.experience.message}
            </p>
          )}
        </div>
        
        {/* Resume Upload */}
        <div className="group transition-all duration-300">
          <label className="block text-sm font-medium text-gray-700 mb-2 group-focus-within:text-blue-700 transition-colors" htmlFor="resume">
            Resume (PDF) <span className="text-blue-600">*</span>
          </label>
          <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-dashed border-blue-200 rounded-lg bg-blue-50/50 transition-colors hover:bg-blue-50 hover:border-blue-300">
            <div className="space-y-2 text-center">
              <svg className="mx-auto h-12 w-12 text-blue-400" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <div className="flex text-sm text-gray-600 justify-center">
                <label htmlFor="resume" className="relative cursor-pointer bg-white py-2 px-4 rounded-md font-medium text-blue-600 hover:text-blue-500 hover:bg-blue-50 border border-blue-200 transition-colors shadow-sm">
                  <span>Select your resume</span>
                  <input
                    id="resume"
                    type="file"
                    accept="application/pdf"
                    className="sr-only"
                    {...register("resume", { 
                      required: "Resume is required",
                      validate: {
                        fileType: (value) => {
                          if (!value[0]) return true;
                          return value[0].type === ALLOWED_FILE_TYPE || "Only PDF files are allowed";
                        },
                        fileSize: (value) => {
                          if (!value[0]) return true;
                          return value[0].size <= MAX_FILE_SIZE || "File size must be less than 5MB";
                        }
                      }
                    })}
                  />
                </label>
              </div>
              <p className="text-xs text-gray-500">PDF up to 5MB</p>
            </div>
          </div>
          {errors.resume && (
            <p className="text-red-500 text-sm mt-2 flex items-center">
              <svg className="h-4 w-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              {errors.resume.message}
            </p>
          )}
        </div>
        
        <div className="pt-4">
          <button
            type="submit"
            disabled={isSubmitting}
            className={`w-full py-4 px-6 bg-gradient-to-r from-blue-600 to-indigo-700 text-white font-bold rounded-lg hover:from-blue-700 hover:to-indigo-800 focus:outline-none focus:ring-4 focus:ring-blue-300 focus:ring-opacity-50 shadow-lg transform transition-all hover:scale-[1.02] active:scale-[0.98] ${
              isSubmitting ? 'opacity-70 cursor-not-allowed' : ''
            }`}
          >
            {isSubmitting ? (
              <div className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing Your Application...
              </div>
            ) : (
              <span className="inline-flex items-center">
                Submit Application
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 ml-2" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L12.586 11H5a1 1 0 110-2h7.586l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </span>
            )}
          </button>
        </div>
        
        <p className="text-center text-sm text-gray-500 mt-4">
          By submitting, you agree to our <a href="#" className="text-blue-600 hover:text-blue-800 hover:underline font-medium">Terms of Service</a> and <a href="#" className="text-blue-600 hover:text-blue-800 hover:underline font-medium">Privacy Policy</a>.
        </p>
      </form>
    </div>
  );
}