import { NextResponse } from 'next/server';
import { GoogleGenerativeAI } from '@google/generative-ai';
import parsePDF from '@/lib/pdfParser';
import fs from 'fs/promises';
import path from 'path';
import { generateAIEvaluation } from '@/lib/embeddings';

// Initialize Google Generative AI
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "YOUR_API_KEY_HERE";
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

export async function POST(request) {
  try {
    const { jobId } = await request.json();
    
    if (!jobId) {
      return NextResponse.json(
        { error: 'Job ID is required' },
        { status: 400 }
      );
    }

    console.log(`Evaluating candidates for job ID: ${jobId}`);
    
    // Get candidates
    const candidatesResponse = await fetch(new URL('/api/candidates', request.url));
    if (!candidatesResponse.ok) {
      throw new Error('Failed to fetch candidates');
    }
    
    const candidates = await candidatesResponse.json();
    
    if (!candidates || candidates.length === 0) {
      return NextResponse.json({ 
        evaluations: [],
        message: 'No candidates found in the system'
      });
    }
    
    console.log(`Found ${candidates.length} candidates to evaluate`);
    
    // Get job details with better error handling
    let job;
    try {
      const jobResponse = await fetch(`${request.nextUrl.origin}/api/jobs/${jobId}`);
      
      if (!jobResponse.ok) {
        console.error(`Failed to fetch job with ID ${jobId}: ${jobResponse.status}`);
        // Return a fallback job if the API call fails
        job = {
          title: "Unknown Position",
          description: "Job details not available",
          requirements: "Unknown requirements"
        };
      } else {
        job = await jobResponse.json();
      }
    } catch (jobError) {
      console.error(`Error fetching job with ID ${jobId}:`, jobError);
      // Return a fallback job if the API call fails
      job = {
        title: "Unknown Position",
        description: "Job details not available",
        requirements: "Unknown requirements"
      };
    }
    
    // Format job text
    const jobText = `
      Title: ${job.title || 'No title'}
      Description: ${job.description || 'No description'}
      Requirements: ${job.requirements || 'No requirements'}
    `;
    
    console.log('Processing job:', job.title);
    
    // Evaluate each candidate
    const evaluationPromises = candidates.map(async (candidate) => {
      try {
        console.log(`Evaluating candidate: ${candidate.name}`);
        
        // Create candidate profile
        const candidateProfile = `
          Name: ${candidate.name || 'Unknown'}
          Email: ${candidate.email || 'Not provided'}
          LinkedIn: ${candidate.linkedin || 'Not provided'}
          Skills: ${candidate.skills || 'Not specified'}
          Experience: ${candidate.experience || 'Not specified'}
          Resume Text: ${candidate.resumeText || 'Not available'}
        `;

        console.log("Resume text length being sent to Gemini:", (candidate.resumeText?.length || 0));
        if (candidate.resumeText?.length > 100) {
          console.log("Resume text preview:", candidate.resumeText.substring(0, 100) + "...");
        }
        
        // Generate AI evaluation
        const evaluation = await generateAIEvaluation(candidateProfile, jobText);
        
        return {
          candidateId: candidate.id.toString(),
          candidateName: candidate.name,
          similarity: "N/A", 
          score: evaluation.score || 0,
          feedback: evaluation.feedback || 'No feedback available',
          recommendations: evaluation.recommendations || 'No recommendations available'
        };
      } catch (evalError) {
        console.error(`Error evaluating candidate ${candidate.name}:`, evalError);
        return {
          candidateId: candidate.id.toString(),
          candidateName: candidate.name,
          similarity: "0.00",
          score: 0,
          feedback: `Error during evaluation: ${evalError.message}`
        };
      }
    });
    
    // Wait for all evaluations to complete
    const evaluations = await Promise.all(evaluationPromises);
    
    // Sort by score (highest first)
    evaluations.sort((a, b) => b.score - a.score);
    
    return NextResponse.json({ evaluations });
    
  } catch (error) {
    console.error('Error evaluating candidates:', error);
    return NextResponse.json(
      { error: 'Error evaluating candidates', details: error.message },
      { status: 500 }
    );
  }
}