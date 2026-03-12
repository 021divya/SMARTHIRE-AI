import { promises as fs } from 'fs';
import path from 'path';
import PDFParser from 'pdf2json';

export default async function parsePDF(filePath) {
  console.log("Processing file with pdf2json:", filePath);
  
  try {
    // Check if file path exists
    if (!filePath) {
      throw new Error("No file path provided");
    }
    
    // Ensure path is absolute
    const absolutePath = path.isAbsolute(filePath) ? filePath : path.join(process.cwd(), filePath);
    console.log("Absolute path:", absolutePath);
    
    // Check if the file exists
    try {
      await fs.access(absolutePath);
    } catch (error) {
      console.error(`File does not exist at path: ${absolutePath}`);
      throw new Error(`File not found: ${absolutePath}`);
    }

    // Create a new PDFParser instance
    const pdfParser = new PDFParser(null, 1);
    
    // Create a promise to handle the parsing
    const parsingPromise = new Promise((resolve, reject) => {
      pdfParser.on("pdfParser_dataError", (errData) => {
        console.error("PDF parsing error:", errData.parserError);
        reject(errData.parserError);
      });
      
      pdfParser.on("pdfParser_dataReady", () => {
        // Get raw text content
        const text = pdfParser.getRawTextContent();
        console.log(`Successfully extracted ${text.length} characters from PDF`);
        resolve(text);
      });
    });
    
    // Load and parse the PDF
    pdfParser.loadPDF(absolutePath);
    const text = await parsingPromise;
    
    // Try to extract a potential job title from the first few lines
    let jobTitle = null;
    const firstLines = text.split('\n').slice(0, 10).join(' ');
    
    // Look for common job title patterns in the beginning of the resume
    const titleMatches = firstLines.match(/(?:senior|junior|lead)?\s*(?:software|frontend|backend|fullstack|web)\s*(?:developer|engineer|architect)/i);
    if (titleMatches) {
      jobTitle = titleMatches[0];
    }
    
    return {
      text: text,
      jobTitle: jobTitle || "Unspecified Position"
    };
    
  } catch (error) {
    console.error("PDF processing error:", error);
    
    // Return an error message that clearly indicates the issue
    return {
      text: `Error processing PDF file: ${error.message}. Please check server logs for details.`,
      jobTitle: "Candidate"
    };
  }
}
