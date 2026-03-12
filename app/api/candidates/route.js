import { NextResponse } from "next/server";
import { writeFile, unlink } from "fs/promises";
import path from "path";
import fs from "fs";
import parsePDF from "@/lib/pdfParser";
import { getCandidates, saveCandidates, addCandidate, getNextCandidateId } from "@/lib/dataStore";

const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
const ALLOWED_FILE_TYPE = "application/pdf";

// Ensure upload directory exists
const uploadDir = path.join(process.cwd(), "public/uploads");
fs.promises.mkdir(uploadDir, { recursive: true }).catch(console.error);

// GET all candidates
export async function GET() {
  const candidates = await getCandidates();
  return NextResponse.json(candidates);
}

// POST (Add a new candidate with resume upload)
export async function POST(request) {
  console.log("POST request received at /api/candidates");
  
  try {
    // Parse the form data from the request
    const formData = await request.formData();
    console.log("Form data parsed successfully");

    // Extract fields from the form data
    const name = formData.get("name");
    const email = formData.get("email");
    const linkedin = formData.get("linkedin") || "";
    const skills = formData.get("skills");
    const experience = formData.get("experience");
    const resumeFile = formData.get("resume");

    console.log("Extracted form fields:", { name, email, skills });

    // Validate required fields
    if (!name || !email || !skills || !experience || !resumeFile) {
      return NextResponse.json({ error: "Missing required fields" }, { status: 400 });
    }

    // Validate file format
    if (!(resumeFile instanceof Blob)) {
      return NextResponse.json({ error: "Invalid file format" }, { status: 400 });
    }

    // Validate file type
    if (resumeFile.type !== ALLOWED_FILE_TYPE) {
      console.log("Invalid file type:", resumeFile.type);
      return NextResponse.json({ error: "Invalid file type. Please upload a PDF" }, { status: 400 });
    }

    // Validate file size
    if (resumeFile.size > MAX_FILE_SIZE) {
      return NextResponse.json({ error: "File size too large. Maximum 5MB" }, { status: 400 });
    }

    // Process the candidate data and file
    const id = await getNextCandidateId();
    const filePath = path.join(uploadDir, `resume_${id}.pdf`);
    console.log("Saving file to:", filePath);

    // Convert Blob to Buffer
    const bytes = await resumeFile.arrayBuffer();
    const buffer = Buffer.from(bytes);
    await writeFile(filePath, buffer);
    console.log("File saved successfully");

    // Parse the PDF to extract text and job title
    let resumeText = "Resume content";
    let jobTitle = "Software Developer"; // Default job title

    try {
      console.log("Attempting to process resume");
      
      // Try to parse the PDF, but don't let failures stop the process
      const parsedResume = await parsePDF(filePath).catch(error => {
        console.error("Safe PDF parsing failed:", error);
        return { text: "Resume processing failed", jobTitle: null };
      });
      
      // Use the parsed results if available
      if (parsedResume) {
        console.log(parsedResume.text);
        resumeText = parsedResume.text || resumeText;
        jobTitle = parsedResume.jobTitle || jobTitle;
      }
      
      console.log("Resume processing complete:", { 
        textLength: resumeText?.length || 0,
        jobTitle 
      });
    } catch (error) {
      // This catch should never be reached due to the .catch() above,
      // but added for extra safety
      console.error("Unexpected error in resume processing:", error);
    }

    // Create candidate object
    const candidate = {
      id,
      name,
      email,
      linkedin,
      skills,
      experience,
      resumePath: `/uploads/resume_${id}.pdf`,
      resumeText: resumeText,
      jobTitle: jobTitle || "Unspecified Position",
      createdAt: new Date().toISOString(),
    };

    // Save the candidate to our persistent storage
    await addCandidate(candidate);
    console.log("Candidate added:", candidate.id);

    // Return success response
    return NextResponse.json({ 
      success: true, 
      message: "Application submitted successfully!", 
      candidate 
    });

  } catch (error) {
    console.error("Error processing candidate application:", error);
    return NextResponse.json({ 
      error: "Error processing application", 
      details: error.message 
    }, { status: 500 });
  }
}

// DELETE (Remove a candidate by ID)
export async function DELETE(request) {
  try {
    const { id } = await request.json();
    const candidates = await getCandidates();
    const candidateIndex = candidates.findIndex((c) => c.id === id);

    if (candidateIndex === -1) {
      return NextResponse.json({ error: "Candidate not found" }, { status: 404 });
    }

    const [removedCandidate] = candidates.splice(candidateIndex, 1);
    await saveCandidates(candidates);

    if (removedCandidate.resumePath) {
      const filePath = path.join(process.cwd(), "public", removedCandidate.resumePath);
      await unlink(filePath).catch(console.error);
    }

    return NextResponse.json({ success: true, message: "Candidate deleted successfully!" });
  } catch (error) {
    return NextResponse.json({ error: "Error deleting candidate", details: error.message }, { status: 500 });
  }
}

// PUT (Update candidate details)
export async function PUT(request) {
  try {
    const { id, name, email, linkedin, skills, experience } = await request.json();
    const candidates = await getCandidates();
    const candidateIndex = candidates.findIndex((c) => c.id === id);

    if (candidateIndex === -1) {
      return NextResponse.json({ error: "Candidate not found" }, { status: 404 });
    }

    candidates[candidateIndex] = {
      ...candidates[candidateIndex],
      name: name || candidates[candidateIndex].name,
      email: email || candidates[candidateIndex].email,
      linkedin: linkedin || candidates[candidateIndex].linkedin,
      skills: skills || candidates[candidateIndex].skills,
      experience: experience || candidates[candidateIndex].experience,
    };

    await saveCandidates(candidates);

    return NextResponse.json({ 
      success: true, 
      message: "Candidate updated successfully!", 
      candidate: candidates[candidateIndex] 
    });
  } catch (error) {
    return NextResponse.json({ 
      error: "Error updating candidate", 
      details: error.message 
    }, { status: 500 });
  }
}
