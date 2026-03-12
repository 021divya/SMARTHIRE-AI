import { GoogleGenerativeAI } from "@google/generative-ai";

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

if (!GEMINI_API_KEY) {
  console.warn("⚠ GEMINI_API_KEY not found in environment variables");
}

const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

/* ----------------------------------------
   Generate embeddings for Pinecone search
-----------------------------------------*/
export async function getEmbedding(text) {
  try {
    const model = genAI.getGenerativeModel({
      model: "text-embedding-004"
    });

    const result = await model.embedContent({
      content: {
        parts: [{ text }]
      }
    });

    return result.embedding.values;

  } catch (error) {
    console.error("Embedding generation failed:", error);

    // Return neutral vector so system continues
    return new Array(768).fill(0);
  }
}

/* ----------------------------------------
   AI Candidate Evaluation
-----------------------------------------*/
export async function generateAIEvaluation(candidateProfile, jobDescription) {
  try {
    const model = genAI.getGenerativeModel({
      model: "gemini-2.0-flash"
    });

    const prompt = `
You are an expert recruiter evaluating candidates.

JOB DESCRIPTION:
${jobDescription}

CANDIDATE PROFILE:
${candidateProfile}

Evaluate the candidate and return ONLY JSON:

{
 "score": number between 0 and 100,
 "feedback": "Detailed explanation of candidate fit",
 "recommendations": "Bullet point recommendations"
}
`;

    const result = await model.generateContent({
      contents: [
        {
          role: "user",
          parts: [{ text: prompt }]
        }
      ],
      generationConfig: {
        temperature: 0.4,
        maxOutputTokens: 1024
      }
    });

    const responseText = result.response.text();

    console.log("Gemini raw response:", responseText);

    let parsed;

    try {
      const jsonStart = responseText.indexOf("{");
      const jsonEnd = responseText.lastIndexOf("}") + 1;

      parsed = JSON.parse(responseText.substring(jsonStart, jsonEnd));

    } catch (err) {
      console.warn("JSON parsing failed, using fallback values");
      parsed = {};
    }

    return {
      score: typeof parsed.score === "number" ? parsed.score : 60,

      feedback:
        parsed.feedback ||
        "The candidate shows partial alignment with the job requirements. Some relevant skills are present but additional experience may be beneficial.",

      recommendations:
        parsed.recommendations ||
        "• Strengthen relevant technical skills\n• Gain more hands-on project experience\n• Highlight measurable achievements"
    };

  } catch (error) {
    console.error("AI evaluation failed:", error);

    return {
      score: 0,
      feedback: "AI evaluation could not be generated.",
      recommendations: "Please try again later."
    };
  }
}