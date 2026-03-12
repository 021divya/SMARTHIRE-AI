import { NextResponse } from "next/server";
import { getCandidates } from "@/lib/dataStore";
import { getEmbedding } from "@/lib/embeddings";
import pineconeIndex from "@/lib/pinecone";
import { GoogleGenerativeAI } from "@google/generative-ai";

// Initialize Gemini
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

export async function POST(request) {
  try {
    console.log("Starting candidate search...");

    const { query, jobId, limit = 10 } = await request.json();
    console.log("Request params:", { query, jobId, limit });

    if (!query && !jobId) {
      return NextResponse.json(
        { error: "Either search query or job ID is required" },
        { status: 400 }
      );
    }

    let searchText = query;
    let jobRequirements = "";

    // Fetch job requirements if jobId provided
    if (jobId) {
      try {
        const jobResponse = await fetch(
          `${request.nextUrl.origin}/api/jobs/${jobId}`
        );

        if (jobResponse.ok) {
          const job = await jobResponse.json();
          jobRequirements = job.requirements || "";
          searchText = `${jobRequirements} ${query || ""}`.trim();
          console.log("Job requirements fetched");
        }
      } catch (err) {
        console.error("Job fetch failed:", err);
      }
    }

    if (!searchText) {
      return NextResponse.json(
        { error: "No valid search text available" },
        { status: 400 }
      );
    }

    console.log("Search text:", searchText);

    const allCandidates = await getCandidates();

    try {
      // VECTOR SEARCH
      console.log("Generating embedding...");
      const searchEmbedding = await getEmbedding(searchText);

      console.log("Querying Pinecone...");

      const searchResults = await pineconeIndex.query({
        vector: searchEmbedding,
        topK: limit * 2,
        filter: { type: "candidate" },
        includeMetadata: true,
      });

      if (searchResults?.matches?.length > 0) {
        console.log(`Vector matches: ${searchResults.matches.length}`);

        let matchedCandidates = searchResults.matches
          .map((match) => {
            const candidateId = parseInt(
              match.id.replace("candidate_", ""),
              10
            );

            const candidate = allCandidates.find(
              (c) => c.id === candidateId
            );

            return candidate
              ? {
                  ...candidate,
                  vectorScore: match.score,
                }
              : null;
          })
          .filter(Boolean);

        const ranked = await rerankWithGemini(
          matchedCandidates,
          jobRequirements || query,
          limit
        );

        return NextResponse.json({
          results: ranked,
          searchMethod: "vector-and-gemini",
        });
      }
    } catch (vectorError) {
      console.log("Vector search failed, fallback to Gemini only");
      console.error(vectorError);
    }

    // FALLBACK SEARCH
    const candidatesSubset = allCandidates.slice(
      0,
      Math.min(allCandidates.length, 15)
    );

    const rankedCandidates = await rerankWithGemini(
      candidatesSubset,
      jobRequirements || query,
      limit
    );

    return NextResponse.json({
      results: rankedCandidates,
      searchMethod: "gemini-only",
    });
  } catch (error) {
    console.error("Search error:", error);

    return NextResponse.json(
      {
        error: "Search failed",
        details: error.message,
      },
      { status: 500 }
    );
  }
}

async function rerankWithGemini(candidates, requirements, limit) {
  if (!candidates || candidates.length === 0) {
    return [];
  }

  try {
    console.log(`Reranking ${candidates.length} candidates`);

    const model = genAI.getGenerativeModel({
      model: "gemini-1.5-flash",
    });

    const prompt = `
You are an expert recruiter evaluating candidates.

JOB REQUIREMENTS:
${requirements || "General technical skills"}

CANDIDATES:
${candidates
  .map(
    (c, i) => `
Candidate ${i + 1}
ID: ${c.id}
Name: ${c.name}
Skills: ${c.skills}
Experience: ${c.experience}
Resume: ${(c.resumeText || "").substring(0, 800)}
`
  )
  .join("\n")}

Return ONLY valid JSON in this format:

{
 "rankings":[
  {
   "id":1,
   "score":0.9,
   "explanation":"Strong skill match"
  }
 ]
}
`;

    const result = await model.generateContent({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: {
        temperature: 0.2,
        maxOutputTokens: 1024,
      },
    });

    const text = result.response.text();

    console.log("Gemini response received");

    // Clean markdown blocks if present
    const cleaned = text.replace(/```json|```/g, "");

    const jsonStart = cleaned.indexOf("{");
    const jsonEnd = cleaned.lastIndexOf("}") + 1;

    if (jsonStart === -1) {
      console.log("JSON not found in response");
      return candidates.slice(0, limit);
    }

    const jsonText = cleaned.substring(jsonStart, jsonEnd);

    const rankings = JSON.parse(jsonText);

    let ranked = candidates.map((candidate) => {
      const match = rankings.rankings?.find(
        (r) => r.id === candidate.id
      );

      return {
        ...candidate,
        score: match?.score ?? 0.1,
        explanation: match?.explanation ?? "Not ranked by AI",
      };
    });

    ranked.sort((a, b) => b.score - a.score);

    console.log(
      "Final ranking:",
      ranked.map((c) => ({
        id: c.id,
        score: c.score,
      }))
    );

    return ranked.slice(0, limit);
  } catch (error) {
    console.error("Gemini ranking error:", error);

    return candidates.slice(0, limit);
  }
}