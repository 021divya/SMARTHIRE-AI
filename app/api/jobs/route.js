import { NextResponse } from "next/server";
import { getJobs, addJob, getNextJobId } from "@/lib/dataStore";
import { getEmbedding } from "@/lib/embeddings";
import pineconeIndex from "@/lib/pinecone";

/* ---------------------------------------------------
   Initialize jobs inside Pinecone (vector database)
--------------------------------------------------- */

async function initializeJobsInPinecone() {
  try {
    const jobs = await getJobs();

    if (!jobs || jobs.length === 0) {
      console.log("No jobs found to initialize.");
      return;
    }

    console.log("Initializing jobs in Pinecone...");

    for (const job of jobs) {
      const jobText = `
Title: ${job.title}
Description: ${job.description}
Requirements: ${job.requirements}
`;

      const embedding = await getEmbedding(jobText);

      // Skip if embedding failed
      if (!embedding || embedding.length === 0) {
        console.warn(`Skipping job ${job.id} due to missing embedding`);
        continue;
      }

      await pineconeIndex.upsert([
        {
          id: `job_${job.id}`,
          values: embedding,
          metadata: {
            type: "job",
            title: job.title
          }
        }
      ]);
    }

    console.log("Jobs successfully initialized in Pinecone");

  } catch (error) {
    console.error("Error initializing jobs:", error.message);
  }
}

// Run initialization once when server loads
initializeJobsInPinecone();


/* ---------------------------------------------------
   GET all jobs
--------------------------------------------------- */

export async function GET() {
  try {
    const jobs = await getJobs();

    return NextResponse.json(jobs);

  } catch (error) {
    return NextResponse.json(
      { error: "Failed to fetch jobs", details: error.message },
      { status: 500 }
    );
  }
}


/* ---------------------------------------------------
   POST create new job
--------------------------------------------------- */

export async function POST(request) {
  try {
    const { title, description, requirements } = await request.json();

    if (!title || !description || !requirements) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 }
      );
    }

    const id = await getNextJobId();

    const job = {
      id,
      title,
      description,
      requirements,
      createdAt: new Date().toISOString()
    };

    await addJob(job);

    /* --- store job embedding in Pinecone --- */

    const jobText = `
Title: ${title}
Description: ${description}
Requirements: ${requirements}
`;

    const embedding = await getEmbedding(jobText);

    if (embedding && embedding.length > 0) {
      await pineconeIndex.upsert([
        {
          id: `job_${id}`,
          values: embedding,
          metadata: {
            type: "job",
            title
          }
        }
      ]);
    }

    return NextResponse.json({
      success: true,
      message: "Job created successfully",
      job
    });

  } catch (error) {
    return NextResponse.json(
      { error: "Error creating job", details: error.message },
      { status: 500 }
    );
  }
}