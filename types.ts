
export interface DocumentChunk {
  id: number;
  text: string;
  embedding: number[];
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export enum AppState {
  INITIAL,
  PROCESSING,
  READY,
  ERROR,
}
