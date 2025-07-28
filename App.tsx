import React, { useState,  useRef, useEffect } from 'react';
import { AppState, DocumentChunk, ChatMessage } from './types.ts';
import { CHUNK_SIZE, CHUNK_OVERLAP } from './constants.ts';
import { InMemoryVectorDB } from './services/vectorDb.ts';
import { generateEmbedding, generateAnswerStream, generateEmbeddings } from './services/geminiService.ts';
import { UploadIcon, SendIcon, AiIcon, UserIcon } from './components/icons.tsx';

// Import and setup PDF.js
import * as pdfjsLibProxy from "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.3.136/pdf.min.mjs";

const pdfjsLib: any = pdfjsLibProxy;
if (pdfjsLib.GlobalWorkerOptions) {
  pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.3.136/pdf.worker.min.mjs`;
}

const vectorDB = new InMemoryVectorDB();
const API_BATCH_SIZE = 50; // Process embeddings in batches of 50 (max 100 for API).
const BATCH_DELAY_MS = 1000; // Delay between batch API calls to avoid rate limits.

const App: React.FC = () => {
    const [appState, setAppState] = useState<AppState>(AppState.INITIAL);
    const [progress, setProgress] = useState({ percentage: 0, message: '' });
    const [error, setError] = useState<string | null>(null);
    const [fileName, setFileName] = useState<string>('');

    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file || !file.name.toLowerCase().endsWith('.pdf')) {
            setError('Please select a PDF file.');
            return;
        }
        if (!pdfjsLib.getDocument) {
             setError('PDF processing library is not available. Please refresh the page.');
             setAppState(AppState.ERROR);
             return;
        }
        
        setError(null);
        setFileName(file.name);
        setAppState(AppState.PROCESSING);
        vectorDB.reset();

        try {
            // 1. Read and Chunk PDF
            setProgress({ percentage: 0, message: 'Reading PDF...' });
            const fileBuffer = await file.arrayBuffer();
            const pdf = await pdfjsLib.getDocument(fileBuffer).promise;
            let fullText = '';
            for (let i = 1; i <= pdf.numPages; i++) {
                const page = await pdf.getPage(i);
                const textContent = await page.getTextContent();
                fullText += textContent.items.map((item: any) => item.str).join(' ') + '\n';
                setProgress({ percentage: Math.round((i / pdf.numPages) * 15), message: `Reading page ${i}/${pdf.numPages}...` });
            }

            const rawChunks: string[] = [];
            for (let i = 0; i < fullText.length; i += CHUNK_SIZE - CHUNK_OVERLAP) {
                rawChunks.push(fullText.substring(i, i + CHUNK_SIZE));
            }
            // Filter out empty or whitespace-only chunks
            const chunks = rawChunks.filter(c => c.trim().length > 0);
            
            // 2. Embed Chunks and store in DB using batch API
            let allEmbeddedChunks: DocumentChunk[] = [];
            let processedCount = 0;

            for (let i = 0; i < chunks.length; i += API_BATCH_SIZE) {
                const batchChunksText = chunks.slice(i, i + API_BATCH_SIZE);
                
                const batchEmbeddings = await generateEmbeddings(batchChunksText, 'RETRIEVAL_DOCUMENT');

                const embeddedChunks = batchChunksText.map((text, index) => ({
                    id: i + index,
                    text,
                    embedding: batchEmbeddings[index],
                }));

                allEmbeddedChunks.push(...embeddedChunks);
                vectorDB.add(embeddedChunks); // Add to DB incrementally

                processedCount += batchChunksText.length;
                setProgress({
                    percentage: 15 + Math.round((processedCount / chunks.length) * 85),
                    message: `Embedding chunk ${processedCount}/${chunks.length}...`
                });

                // Add a polite delay between batches to stay under API rate limits
                if (i + API_BATCH_SIZE < chunks.length) {
                    await new Promise(resolve => setTimeout(resolve, BATCH_DELAY_MS));
                }
            }
            
            setProgress({ percentage: 100, message: 'Processing complete!' });
            setAppState(AppState.READY);

        } catch (err: any) {
            console.error("Processing failed:", err);
            setError(`An error occurred: ${err.message}`);
            setAppState(AppState.ERROR);
        }
    };

    const handleReset = () => {
        setAppState(AppState.INITIAL);
        setError(null);
        setFileName('');
        setProgress({ percentage: 0, message: '' });
        vectorDB.reset();
    };

    return (
        <div className="min-h-screen flex flex-col items-center justify-center p-4 sm:p-6 bg-base-100 font-sans">
            <div className="w-full max-w-4xl mx-auto">
                <header className="text-center mb-8">
                    <h1 className="text-4xl sm:text-5xl font-bold text-text-primary tracking-tight">AskMyDoc</h1>
                    <p className="mt-2 text-lg text-text-secondary">Upload a PDF and ask questions about its content.</p>
                </header>

                <main className="bg-base-200 rounded-2xl shadow-2xl p-6 sm:p-8 min-h-[60vh] flex flex-col">
                    {appState === AppState.INITIAL && <PdfUploader onChange={handleFileChange} />}
                    
                    {appState === AppState.PROCESSING && <ProcessingView progress={progress} />}

                    {appState === AppState.READY && <ChatView fileName={fileName} onReset={handleReset} />}
                    
                    {appState === AppState.ERROR && (
                        <div className="text-center text-red-400">
                            <h3 className="text-2xl font-semibold mb-4">Processing Failed</h3>
                            <p className="mb-6">{error}</p>
                            <button onClick={handleReset} className="px-6 py-2 bg-brand-primary hover:bg-brand-secondary text-white font-semibold rounded-lg transition-colors">
                                Try Again
                            </button>
                        </div>
                    )}
                </main>
                <footer className="text-center mt-8 text-text-secondary text-sm">
                    <p>Powered by Google Gemini & React</p>
                </footer>
            </div>
        </div>
    );
};

// --- UI Components ---

const PdfUploader = ({ onChange }: { onChange: (e: React.ChangeEvent<HTMLInputElement>) => void }) => (
    <div className="flex flex-col items-center justify-center h-full text-center p-8 border-4 border-dashed border-base-300 rounded-xl">
        <UploadIcon className="w-16 h-16 text-brand-secondary mb-4" />
        <h2 className="text-2xl font-bold text-text-primary mb-2">Upload Your Document</h2>
        <p className="text-text-secondary mb-6 max-w-sm">Select a PDF file to begin. The content will be processed and embedded for querying.</p>
        <label htmlFor="file-upload" className="cursor-pointer px-8 py-3 bg-brand-primary hover:bg-brand-secondary text-white font-bold rounded-lg shadow-lg transition-transform transform hover:scale-105">
            Select PDF
        </label>
        <input id="file-upload" type="file" className="hidden" accept=".pdf" onChange={onChange} />
    </div>
);

const ProcessingView = ({ progress }: { progress: { percentage: number; message: string } }) => (
    <div className="flex flex-col items-center justify-center h-full">
        <h2 className="text-3xl font-bold text-text-primary mb-4">Processing Document...</h2>
        <div className="w-full bg-base-300 rounded-full h-4 mb-2 overflow-hidden">
            <div className="bg-brand-secondary h-4 rounded-full transition-all duration-300" style={{ width: `${progress.percentage}%` }}></div>
        </div>
        <p className="text-text-secondary font-medium">{progress.message}</p>
    </div>
);

const ChatView = ({ fileName, onReset }: { fileName: string; onReset: () => void; }) => {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(scrollToBottom, [messages]);

    const handleSendMessage = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage: ChatMessage = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const queryEmbedding = await generateEmbedding(input, 'RETRIEVAL_QUERY');
            const contextChunks = vectorDB.search(queryEmbedding, 5);
            const context = contextChunks.map(c => c.text).join('\n---\n');

            const stream = await generateAnswerStream(context, input);
            
            const assistantMessage: ChatMessage = { role: 'assistant', content: '' };
            setMessages(prev => [...prev, assistantMessage]);

            for await (const chunk of stream) {
                const chunkText = chunk.text;
                setMessages(prev => prev.map((msg, index) => 
                    index === prev.length - 1 
                        ? { ...msg, content: msg.content + chunkText } 
                        : msg
                ));
            }
        } catch (error) {
            console.error(error);
            const errorMessageContent = error instanceof Error ? error.message : 'Sorry, I encountered an error. Please try again.';
            const errorMessage: ChatMessage = { role: 'assistant', content: errorMessageContent };
            setMessages(prev => {
                // If the last message was the placeholder, replace it. Otherwise, add new.
                const lastMessage = prev[prev.length -1];
                if (lastMessage && lastMessage.role === 'assistant' && lastMessage.content === '') {
                    const newMessages = [...prev];
                    newMessages[newMessages.length - 1] = errorMessage;
                    return newMessages;
                }
                return [...prev, errorMessage];
            });
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full max-h-[70vh]">
            <div className="flex-shrink-0 flex justify-between items-center pb-4 border-b border-base-300">
                <div>
                    <h3 className="text-xl font-bold">Chatting with:</h3>
                    <p className="text-text-secondary truncate">{fileName}</p>
                </div>
                <button onClick={onReset} className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white font-semibold rounded-lg transition-colors text-sm">
                    Upload New
                </button>
            </div>

            <div className="flex-grow overflow-y-auto my-4 pr-4 space-y-6">
                {messages.map((msg, index) => (
                    <div key={index} className={`flex items-start gap-4 ${msg.role === 'user' ? 'justify-end' : ''}`}>
                        {msg.role === 'assistant' && <div className="flex-shrink-0 w-8 h-8 rounded-full bg-brand-primary flex items-center justify-center text-white"><AiIcon className="w-5 h-5"/></div>}
                        <div className={`max-w-xl p-4 rounded-2xl ${msg.role === 'user' ? 'bg-brand-secondary text-white rounded-br-none' : 'bg-base-300 text-text-primary rounded-bl-none'}`}>
                            <p className="whitespace-pre-wrap">{msg.content}</p>
                        </div>
                         {msg.role === 'user' && <div className="flex-shrink-0 w-8 h-8 rounded-full bg-base-300 flex items-center justify-center"><UserIcon className="w-5 h-5 text-text-secondary"/></div>}
                    </div>
                ))}
                {isLoading && messages[messages.length-1]?.role === 'user' && (
                     <div className="flex items-start gap-4">
                        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-brand-primary flex items-center justify-center text-white"><AiIcon className="w-5 h-5"/></div>
                        <div className="max-w-xl p-4 rounded-2xl bg-base-300 text-text-primary rounded-bl-none">
                            <div className="flex items-center space-x-2">
                                <div className="w-2 h-2 bg-brand-secondary rounded-full animate-pulse delay-75"></div>
                                <div className="w-2 h-2 bg-brand-secondary rounded-full animate-pulse delay-150"></div>
                                <div className="w-2 h-2 bg-brand-secondary rounded-full animate-pulse delay-300"></div>
                            </div>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <form onSubmit={handleSendMessage} className="flex-shrink-0 flex gap-4 mt-auto pt-4 border-t border-base-300">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask a question about the document..."
                    className="w-full px-4 py-3 bg-base-100 text-text-primary rounded-lg border-2 border-base-300 focus:border-brand-secondary focus:ring-0 outline-none transition"
                    disabled={isLoading}
                />
                <button type="submit" className="px-5 py-3 bg-brand-primary hover:bg-brand-secondary text-white font-bold rounded-lg disabled:bg-base-300 disabled:cursor-not-allowed transition-colors flex items-center justify-center" disabled={isLoading || !input.trim()}>
                    <SendIcon className="w-6 h-6" />
                </button>
            </form>
        </div>
    );
};


export default App;