import dotenv from "dotenv";
import path from "path";
import { ChatOpenAI } from "@langchain/openai";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";
import { UnstructuredLoader } from "@langchain/community/document_loaders/fs/unstructured";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { WikipediaQueryRun } from "@langchain/community/tools/wikipedia_query_run";
import { ChatPromptTemplate } from "@langchain/core/prompts";

// Load environment variables
dotenv.config({
    path: path.resolve(__dirname, "../../.env"), // Adjust path to your .env if needed
});

// Ensure OpenAI API Key Exists
const openaiApiKey = process.env.OPENAI_API_KEY;
if (!openaiApiKey) {
    throw new Error("No OpenAI API key found. Make sure the .env file is loaded correctly.");
}

async function main() {
    // Initialize Chat Model
    const chatModel = new ChatOpenAI({
        openAIApiKey: openaiApiKey,
        model: "gpt-4o", // Can also use gpt-4 or other models if needed
    });

    console.log("\n----------\n");

    /**
     * Loading TXT File
     */
    const textLoader = new TextLoader("../../data/be-good.txt");
    const textData = await textLoader.load();
    console.log("Loaded TXT file content:");
    console.log("\n----------\n");
    console.log(textData);

    console.log("\n----------\n");

    /**
     * Loading CSV File
     */
    const csvLoader = new CSVLoader("../../data/Street_Tree_List.csv");
    const csvData = await csvLoader.load();
    console.log("Loaded CSV file content:");
    console.log("\n----------\n");
    console.log(csvData);

    console.log("\n----------\n");

    /**
     * Loading HTML File
     */
    const htmlLoader = new UnstructuredLoader("../../data/100-startups.html", {
        apiKey: process.env.UNSTRUCTURED_API_KEY,
        apiUrl: 'https://api.unstructuredapp.io/general/v0/general'
    });
    const htmlData = await htmlLoader.load();
    console.log("Loaded HTML page content:");
    console.log("\n----------\n");
    console.log(htmlData);

    console.log("\n----------\n");

    /**
     * Loading PDF File
     */
    const pdfLoader = new PDFLoader("../../data/5pages.pdf");
    const pdfData = await pdfLoader.load();
    console.log("Loaded PDF pages content:");
    console.log("\n----------\n");
    console.log(pdfData[0].pageContent);
    console.log("\n----------\n");

    /**
     * Loading Wikipedia Content
     */

    const tool = new WikipediaQueryRun({
        topKResults: 3,
        maxDocContentLength: 4000,
    });

    const wikiPageContent = await tool.invoke("JFK");

    /**
     * Chat Prompt with Wikipedia Data
     */
    const chatTemplate = ChatPromptTemplate.fromMessages([
        { role: "human", content: "Answer this question: {question}, here is some extra context: {context}" },
    ]);

    const messages = await chatTemplate.formatMessages({
        question: "What was the full name of JFK?",
        context: wikiPageContent, // Inject Wikipedia context
    });

    const response = await chatModel.invoke(messages);

    console.log("\n----------\n");
    console.log("Response from Wikipedia: What was the full name of JFK?");
    console.log("\n----------\n");
    console.log(response.content);
    console.log("\n----------\n");
}

// Run the script
main().catch((error) => {
    console.error("An error occurred:");
    console.error(error);
});