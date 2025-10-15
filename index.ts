import * as dotenv from 'dotenv';
dotenv.config();

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { TextLoader } from "langchain/document_loaders/fs/text"; // Ví dụ
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
// import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { InMemoryChatMessageHistory  } from "@langchain/core/chat_history";
// import { ChatMessageHistory } from "@langchain/community/stores/message/in_memory"; 
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from "@langchain/pinecone";
import { BaseMessage } from '@langchain/core/messages';

// Tải tài liệu
const loader = new TextLoader("document.txt");
const rawDocs = await loader.load();

// Chia nhỏ tài liệu
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 100,
  chunkOverlap: 20,
});
const splitDocs = await splitter.splitDocuments(rawDocs);
console.log("RRR", splitDocs.length);

// const embeddings = new OpenAIEmbeddings(
//     {
//         model:'text-embedding-3-small',
//     }
// );
const embeddings = new GoogleGenerativeAIEmbeddings({
  // Sử dụng mô hình mới nhất và hiệu quả nhất cho Embedding
  model: "text-embedding-004", 
  apiKey: process.env.GEMINI_API_KEY || 'AIzaSyC3LNv1UUMd0FAIH_Vx2DYffe6s2r6RFHI'
  // API Key sẽ được tự động lấy từ biến môi trường GEMINI_API_KEY
});
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY!,
});
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME!);

await PineconeStore.fromDocuments(
  splitDocs,
  embeddings,
  {
    pineconeIndex,
    // namespace: "my-namespace", // Tùy chọn: sử dụng namespace để phân chia dữ liệu
  }
);
// 4. Kết nối với Index để tạo Retriever
const vectorStore = new PineconeStore(embeddings, { pineconeIndex });
// // Tạo Retriever
const retriever = vectorStore.asRetriever();
// // client = chromadb.CloudClient(
//   api_key='ck-D3cvNih9An2ZGDQTZ19L9Hd32bbtemmCgrKWqc6HKyVB',
//   tenant='e957c441-3c50-4a3e-b0a2-3f84308d9e4b',
//   database='Development'
// )

// Khởi tạo và lưu trữ vào Vector Store
// const vectorStore = await Chroma.fromDocuments(splitDocs, embeddings, {
//   collectionName: "my-docs",
//   chromaCloudAPIKey:'ck-D3cvNih9An2ZGDQTZ19L9Hd32bbtemmCgrKWqc6HKyVB',
// });
// const vectorStore = await FaissStore.fromDocuments(
//   splitDocs, // Dữ liệu đã chia nhỏ
//   embeddings
// );
// // // Tạo Retriever để truy xuất tài liệu liên quan
// const retriever = vectorStore.asRetriever();

const LLM = new ChatGoogleGenerativeAI({
    // openAIApiKey:'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',
     model: 'gemini-2.5-flash', 
     apiKey: process.env.GEMINI_API_KEY || 'AIzaSyC3LNv1UUMd0FAIH_Vx2DYffe6s2r6RFHI',
    temperature: 0.7, // Nhiệt độ tùy chọn
}); // Mô hình ngôn ngữ lớn

// 1. Prompt để rephrase câu hỏi (tạo ngữ cảnh từ lịch sử)
const REPHRASE_PROMPT = ChatPromptTemplate.fromMessages([
  new MessagesPlaceholder("chat_history"), // Nơi chèn lịch sử chat
  ["human", "Dựa vào lịch sử hội thoại trên, hãy diễn đạt lại câu hỏi sau để nó độc lập: {input}"],
]);

// 2. Prompt chính cho RAG
const QA_PROMPT = ChatPromptTemplate.fromMessages([
  ["system", "Bạn là một trợ lý hữu ích. Hãy trả lời câu hỏi dựa trên ngữ cảnh sau: {context}"],
  new MessagesPlaceholder("chat_history"),
  ["human", "{input}"],
]);
const historyAwareRetriever = await createHistoryAwareRetriever({
  llm: LLM,
  retriever,
  rephrasePrompt: REPHRASE_PROMPT,
});

const combineDocsChain = await createStuffDocumentsChain({
  llm: LLM,
  prompt: QA_PROMPT,
});

const ragChain = await createRetrievalChain({
  retriever: historyAwareRetriever,
  combineDocsChain,
});


// Store lịch sử tạm thời trong bộ nhớ (InMemory)
const messageHistory: Record<string, InMemoryChatMessageHistory > = {};

const finalChatbot = new RunnableWithMessageHistory({
  runnable: ragChain,
  getMessageHistory: async (sessionId: string) => {
    if (messageHistory[sessionId] === undefined) {
      messageHistory[sessionId] = new InMemoryChatMessageHistory ();
    }
    return messageHistory[sessionId];
  },
  outputMessagesKey: "answer", // <-- ADD THIS LINE
  inputMessagesKey: "input", // Key cho input của người dùng
  historyMessagesKey: "chat_history", // Key cho lịch sử chat trong prompt
});

// Sử dụng Chatbot
const sessionId = "user123";

// await finalChatbot.invoke(
//   { input: "Tên tôi là gì và nó được dùng để làm gì?" }, // inputMessagesKey = "input"
//   { configurable: { sessionId } }
// );
//     console.log("History for session", sessionId, await messageHistory[sessionId]?.getMessages());

// Lượt 1
let result = await finalChatbot.invoke(
  { input: "Tên tôi là An. Nodejs là gì?" },
  { configurable: { sessionId } }
);
console.log("Lượt 1:", result.answer);
// console.log("History for session", sessionId,await messageHistory[sessionId]?.getMessages());
// Lượt 2 - Hỏi một câu hỏi cần ngữ cảnh
result = await finalChatbot.invoke(
  { input: "Tên tôi là gì và nó được dùng để làm gì?" },
  { configurable: { sessionId } }
);
console.log("Lượt 2:", result.answer);


console.log("Test");
