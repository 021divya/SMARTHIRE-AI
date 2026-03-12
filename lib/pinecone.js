// lib/pinecone.js
import { Pinecone } from '@pinecone-database/pinecone';

// Initialize Pinecone client with updated API format
const pinecone = new Pinecone({
  apiKey: "pcsk_gDMz6_7Dzwxtk64vWgUsuFgfaBbrRDCd8gzRPCG2DyFDnSEHcMrRQ4jn5x1RjtjnHm3KN"
  // The 'environment' parameter is not supported in the newer SDK versions
});

// Get the index name from environment variables or use a default
const indexName = process.env.PINECONE_INDEX || "candidate-index";
const index = pinecone.index(indexName);

export default index;