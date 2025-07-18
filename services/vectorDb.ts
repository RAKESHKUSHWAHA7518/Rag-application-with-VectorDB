
import { DocumentChunk } from '../types.ts';

export class InMemoryVectorDB {
  private chunks: DocumentChunk[] = [];

  add(newChunks: DocumentChunk[]): void {
    this.chunks.push(...newChunks);
  }

  search(queryEmbedding: number[], topK: number): DocumentChunk[] {
    if (this.chunks.length === 0) {
      return [];
    }

    const similarities = this.chunks.map(chunk => ({
      chunk,
      similarity: this.cosineSimilarity(queryEmbedding, chunk.embedding),
    }));

    similarities.sort((a, b) => b.similarity - a.similarity);
    
    return similarities.slice(0, topK).map(s => s.chunk);
  }

  private cosineSimilarity(vecA: number[], vecB: number[]): number {
    if (vecA.length !== vecB.length) {
      // Return a very low similarity score for mismatched dimensions.
      // In a real scenario, this should probably throw an error.
      console.error("Vector dimensions do not match!");
      return -1;
    }
    
    let dotProduct = 0;
    let magA = 0;
    let magB = 0;
    
    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      magA += vecA[i] * vecA[i];
      magB += vecB[i] * vecB[i];
    }
    
    magA = Math.sqrt(magA);
    magB = Math.sqrt(magB);
    
    if (magA === 0 || magB === 0) {
      return 0;
    }
    
    return dotProduct / (magA * magB);
  }

  reset(): void {
    this.chunks = [];
  }
  
  isReady(): boolean {
    return this.chunks.length > 0;
  }
}