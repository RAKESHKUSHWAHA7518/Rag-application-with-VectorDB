import { GoogleGenAI } from "@google/genai";
import { EMBEDDING_MODEL_NAME, GENERATIVE_MODEL_NAME } from '../constants.ts';

if (!process.env.API_KEY) {
    throw new Error("API_KEY environment variable not set");
}

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
const generativeModel = GENERATIVE_MODEL_NAME;

const handleError = (error: any, context: string): never => {
    console.error(`Failed to ${context}:`, error);
    if (error.message && error.message.includes('RESOURCE_EXHAUSTED')) {
        throw new Error("API quota exceeded. The document is too large or requests are too frequent. Please wait and try again.");
    }
    throw new Error(`Failed to ${context}. Please check the console for details.`);
};


/**
 * Generates a numerical embedding for a single text string.
 *
 * @param text The string to embed.
 * @param taskType The type of task for which the embedding is generated.
 * @returns A promise that resolves to an array of numbers representing the embedding.
 */
export const generateEmbedding = async (
    text: string,
    taskType: 'RETRIEVAL_DOCUMENT' | 'RETRIEVAL_QUERY'
): Promise<number[]> => {
    try {
        const response = await ai.models.embedContent({
            model: EMBEDDING_MODEL_NAME,
            contents: text,
            taskType: taskType,
        });

        const embedding = response.embeddings && response.embeddings[0] ? response.embeddings[0].values : undefined;

        if (embedding && Array.isArray(embedding)) {
            return embedding;
        }
        
        throw new Error("Invalid embedding format received from API.");
    } catch (error) {
        handleError(error, 'generate single embedding');
    }
};

/**
 * Generates embeddings for an array of text strings in a single batch API call.
 *
 * @param texts An array of strings to embed.
 * @param taskType The type of task for the embeddings.
 * @returns A promise that resolves to an array of number arrays (embeddings).
 */
export const generateEmbeddings = async (
    texts: string[],
    taskType: 'RETRIEVAL_DOCUMENT'
): Promise<number[][]> => {
    try {
        const response = await ai.models.embedContent({
            model: EMBEDDING_MODEL_NAME,
            contents: texts,
            taskType: taskType,
        });

        const embeddings = response.embeddings.map(e => e.values);
        if (embeddings && embeddings.length === texts.length) {
            return embeddings;
        }

        throw new Error("Mismatch between number of texts and embeddings received.");
    } catch(error) {
        handleError(error, 'generate batch embeddings');
    }
}

export const generateAnswerStream = async (context: string, question: string) => {
    const prompt = `Based on the following context, please provide a comprehensive answer to the user's question. If the context does not contain the answer, state that you cannot find the answer in the provided document.

Context:
---
${context}
---

Question: ${question}
`;

    try {
        const responseStream = await ai.models.generateContentStream({
            model: generativeModel,
            contents: prompt,
            config: {
                systemInstruction: "You are a helpful AI assistant that answers questions based on a given document context.",
            }
        });
        return responseStream;
    } catch (error) {
        console.error("Failed to generate answer:", error);
        throw new Error("Could not get an answer from the AI model.");
    }
};