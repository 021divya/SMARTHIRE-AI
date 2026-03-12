# SmartHire-AI

A modern web application for managing candidate information built with Next.js, featuring AI-powered resume parsing, candidate evaluation, and job matching.

## Project Description

This application serves as a comprehensive platform for tracking and managing job candidates throughout the recruitment process. Users can:

- View a dashboard of all candidates with sorting and filtering capabilities
- Add new candidates with detailed profile information
- Generate reports on hiring metrics
- Perform AI-powered candidate evaluations and job matching
- Receive intelligent resume summaries and candidate rankings

The application uses a modern tech stack with server-side rendering for optimal performance and SEO, while providing a seamless user experience through client-side interactions.

## Features

- Candidate profile management and submission
- Resume parsing and analysis
- AI-powered candidate evaluation and matching
- Vector-based semantic search for relevant candidates
- Interactive admin dashboard with candidate analytics
- Automated feedback generation for candidates
- User authentication and role-based access

## Technologies Used

- [Next.js](https://nextjs.org/) - React framework
- TypeScript
- Tailwind CSS for styling
- API Routes for backend functionality
- Next.js App Router
- Pinecone for vector database and similarity search
- Google Gemini API for AI-powered features
- RAG (Retrieval-Augmented Generation) architecture for candidate evaluation

## Getting Started

### Prerequisites

- Node.js 18.x or higher
- npm or yarn
- Pinecone account
- Google Gemini API Key

### Environment Variables

Create a `.env.local` file in the root directory with the following variables:

```
# API Keys
PINECONE_API_KEY=YOUR_API_KEY
PINECONE_ENVIRONMENT=YOUR_ENVIRONMENT
PINECONE_INDEX=YOUR_INDEX
GEMINI_API_KEY=YOUR_API_KEY
```

Make sure to replace the placeholder values with your actual credentials. Never commit the `.env.local` file to version control.

### Installation

```bash
# Clone the repository
git clone <your-repo-url>

# Navigate to the project directory
cd candidate-app

# Install dependencies
npm install
# or
yarn install
```

### Development

Run the development server:

```bash
npm run dev
# or
yarn dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the application.

<!-- ## How It Works

1. **Authentication**: Users sign in through NextAuth.js integration
2. **Dashboard**: The main interface displays candidate cards with key information
3. **Candidate Management**:
   - Add new candidates through a multi-step form
   - Update candidate status with a drag-and-drop interface
   - View detailed profiles with application history
4. **Interview Scheduling**: Calendar integration for scheduling and notifications
5. **Analytics**: Real-time metrics on hiring pipeline and conversion rates -->

# How It Works

Our candidate evaluation system streamlines the recruitment process through advanced AI and vector database technologies:

1. **Candidate Submission**: Candidates submit their name, email, LinkedIn URL, resume (PDF or text), and manually enter skills and experience

2. **Resume Processing**: The system extracts text from PDFs using pdf-parse and processes it to identify keywords like skills, experience, and education

3. **Vector Database Storage**: Job descriptions and candidate profiles are stored in Pinecone as vector embeddings

4. **RAG Implementation**: The application uses Google's textembedding-gecko to create searchable resume vectors

5. **Gemini API Integration**: Leverages Google Gemini API to:

   - Summarize resumes and generate candidate profiles
   - Match job descriptions with candidates using semantic search
   - Generate AI-powered evaluation scores

6. **Candidate Ranking**: Scores candidates based on relevance to job descriptions

7. **AI Feedback**: Provides AI-generated feedback on missing skills and suggests next steps

## Project Structure

```
candidate-app/
├── app/               # Next.js App Router
│   ├── admin/         # Admin dashboard pages
│   ├── api/           # API routes for backend functionality
│   ├── layout.tsx     # Root layout component
│   └── page.tsx       # Homepage component
├── components/        # Reusable UI components
│   ├── CandidateForm.tsx  # Form for candidate submissions
│   ├── Navbar.tsx     # Navigation component
│   └── ResumeSearch.tsx   # AI-powered resume search component
├── data/              # Sample data and schemas
│   ├── candidates.json  # Candidate information
│   └── jobs.json      # Job listings
├── lib/               # Utility functions and shared code
│   ├── dataStore.js   # Data persistence functions
│   ├── embeddings.js  # Vector embedding generation
│   └── pinecone.js    # Pinecone database client
├── public/            # Static assets
└── .env.local         # Environment variables (not committed to git)
```

## Core Functionalities

### Resume Parsing

Extracts text from PDF resumes using pdf-parse and processes the content to identify key information like skills, experience, and education.

### Vector Search

Implements RAG architecture using Pinecone to store and search vector embeddings of candidate resumes and job descriptions, enabling semantic matching between candidates and positions.

### AI Integration

Google Gemini API powers intelligent features including:

- Resume summarization
- Candidate-job matching using semantic search
- Evaluation score generation
- Customized feedback on candidate qualifications

### Candidate Ranking

Scores and ranks candidates based on their relevance to specific job descriptions, providing recruiters with AI-assisted candidate recommendations.

## Deployment

The application can be deployed on [Vercel](https://vercel.com/new?utm_medium=default-template&filter=next.js) with minimal configuration:

1. Push your code to a GitHub repository
2. Import your repository to Vercel
3. Configure environment variables in the Vercel dashboard
4. Deploy

## Troubleshooting

1. Missing Environment Variables: Ensure all required variables in .**env.local** are correctly set
2. Pinecone Connection Issues: Verify your Pinecone API key and environment settings
3. Build Errors: Make sure all dependencies are installed with **npm install**

## Learn More

To learn more about the technologies used in this project:

- [Next.js Documentation](https://nextjs.org/docs)
- [Google Gemini API Documentation](https://ai.google.dev/docs/gemini_api)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [RAG Architecture Overview](https://www.pinecone.io/learn/retrieval-augmented-generation/)
