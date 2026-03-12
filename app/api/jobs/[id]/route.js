import { NextResponse } from "next/server";
import { getJobs } from "@/lib/dataStore";

export async function GET(request, { params }) {
  try {
    const { id } = await params;
    const jobId = parseInt(id, 10);
    const jobs = await getJobs();
    const job = jobs.find(j => j.id === id);
    
    if (!job) {
      return NextResponse.json({ error: "Job not found" }, { status: 404 });
    }
    
    return NextResponse.json(job);
  } catch (error) {
    return NextResponse.json(
      { error: "Error retrieving job", details: error.message },
      { status: 500 }
    );
  }
}