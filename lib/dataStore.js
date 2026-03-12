import fs from "fs";
import path from "path";
import { promises as fsPromises } from "fs";
import { getEmbedding } from "./embeddings";
import pineconeIndex from "./pinecone";

/* ----------------------------------------
   File paths
----------------------------------------- */

const dataDir = path.join(process.cwd(), "data");
const candidatesFile = path.join(dataDir, "candidates.json");
const jobsFile = path.join(dataDir, "jobs.json");

/* ----------------------------------------
   Ensure data directory exists
----------------------------------------- */

try {
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
    console.log("Data directory created");
  }
} catch (error) {
  console.error("Error creating data directory:", error);
}

/* ----------------------------------------
   Helper: read JSON data
----------------------------------------- */

async function readData(filePath, defaultData = []) {
  try {
    if (!fs.existsSync(filePath)) {
      await fsPromises.writeFile(filePath, JSON.stringify(defaultData, null, 2));
      return defaultData;
    }

    const data = await fsPromises.readFile(filePath, "utf-8");

    if (!data) return defaultData;

    return JSON.parse(data);
  } catch (error) {
    console.error(`Error reading ${filePath}:`, error);
    return defaultData;
  }
}

/* ----------------------------------------
   Helper: write JSON data
----------------------------------------- */

async function writeData(filePath, data) {
  try {
    await fsPromises.writeFile(filePath, JSON.stringify(data, null, 2));
  } catch (error) {
    console.error(`Error writing ${filePath}:`, error);
    throw error;
  }
}

/* ----------------------------------------
   Candidate Operations
----------------------------------------- */

export async function getCandidates() {
  return readData(candidatesFile, []);
}

export async function saveCandidates(candidates) {
  await writeData(candidatesFile, candidates);
}

export async function addCandidate(candidate) {
  const candidates = await getCandidates();

  candidates.push(candidate);

  await saveCandidates(candidates);

  // index candidate in Pinecone
  await indexCandidateInPinecone(candidate);

  return candidate;
}

/* ----------------------------------------
   Pinecone Candidate Indexing
----------------------------------------- */

export async function indexCandidateInPinecone(candidate) {
  try {
    const candidateText = `
Name: ${candidate.name || ""}
Skills: ${candidate.skills || ""}
Experience: ${candidate.experience || ""}
Resume: ${candidate.resumeText || ""}
`;

    const embedding = await getEmbedding(candidateText);

    if (!embedding || embedding.length === 0) {
      console.warn(`Skipping Pinecone indexing for candidate ${candidate.id}`);
      return false;
    }

    await pineconeIndex.upsert([
      {
        id: `candidate_${candidate.id}`,
        values: embedding,
        metadata: {
          type: "candidate",
          name: candidate.name || "",
          skills: candidate.skills || "",
          experience: candidate.experience || ""
        }
      }
    ]);

    console.log(`Candidate ${candidate.id} indexed in Pinecone`);

    return true;

  } catch (error) {
    console.error(`Error indexing candidate ${candidate.id}:`, error.message);
    return false;
  }
}

/* ----------------------------------------
   Job Operations
----------------------------------------- */

export async function getJobs() {
  return readData(jobsFile, [
    {
      id: 1,
      title: "Frontend Developer",
      description: "Looking for a React developer to build modern web apps.",
      requirements:
        "3+ years React experience, TypeScript, Next.js, REST APIs.",
      createdAt: new Date().toISOString()
    },
    {
      id: 2,
      title: "Backend Engineer",
      description: "Node.js developer to design scalable backend services.",
      requirements:
        "Experience with Node.js, Express, databases, and cloud platforms.",
      createdAt: new Date().toISOString()
    }
  ]);
}

export async function saveJobs(jobs) {
  await writeData(jobsFile, jobs);
}

export async function addJob(job) {
  const jobs = await getJobs();

  jobs.push(job);

  await saveJobs(jobs);

  return job;
}

/* ----------------------------------------
   ID Generators
----------------------------------------- */

export async function getNextCandidateId() {
  const candidates = await getCandidates();

  return candidates.length > 0
    ? Math.max(...candidates.map((c) => c.id)) + 1
    : 1;
}

export async function getNextJobId() {
  const jobs = await getJobs();

  return jobs.length > 0
    ? Math.max(...jobs.map((j) => j.id)) + 1
    : 1;
}