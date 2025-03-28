import dotenv from "dotenv";
import path from "path";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { CharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { FaissStore  } from "@langchain/community/vectorstores/faiss";

// Load environment variables
dotenv.config({
    path: path.resolve(__dirname, "../../.env"),
});

// Ensure OpenAI API Key Exists
const openaiApiKey = process.env.OPENAI_API_KEY;
if (!openaiApiKey) {
    throw new Error("No OpenAI API key found. Make sure the .env file is loaded correctly.");
}

async function runChromaWorkflow() {
    console.log("\n----------\n");
    console.log("Running Chroma Workflow");
    console.log("\n----------\n");

    /**
     * Step 1: Load the Document
     */
    const textLoader = new TextLoader("../../data/state_of_the_union.txt");
    const loadedDocument = await textLoader.load();

    console.log("Document loaded successfully!");
    console.log("\n----------\n");

    /**
     * Step 2: Split the Document into Chunks
     */
    const textSplitter = new CharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 0,
    });

    const chunksOfText = await textSplitter.splitText(loadedDocument[0].pageContent);

    console.log("Document successfully split into chunks!");
    console.log("\n----------\n");

    /**
     * Step 3: Embed Chunks and Create a Vector Store with Chroma
     */
    const embeddingsModel = new OpenAIEmbeddings({
        openAIApiKey: openaiApiKey,
    });

    const vectorDb = await Chroma.fromTexts(chunksOfText,[], embeddingsModel, {});

    console.log("Vector Store created with embedded chunks (Chroma)!");
    console.log("\n----------\n");

    /**
     * Step 4: Perform Similarity Search
     */
    const question = "What did the president say about the John Lewis Voting Rights Act?";
    const response = await vectorDb.similaritySearch(question);

    console.log("\n----------\n");
    console.log("Ask the RAG App: What did the president say about the John Lewis Voting Rights Act?");
    console.log("\n----------\n");

    if (response.length > 0) {
        console.log(response[0].pageContent); // Output the most relevant chunk of text
    } else {
        console.log("No relevant content found.");
    }

    console.log("\n----------\n");
}

async function runFAISSWorkflow() {
    console.log("\n----------\n");
    console.log("Running FAISS Workflow");
    console.log("\n----------\n");

    /**
     * Step 1: Load the Document
     */
    const textLoader = new TextLoader("../../data/state_of_the_union.txt");
    const loadedDocument = await textLoader.load();

    console.log("Document loaded successfully!");
    console.log("\n----------\n");

    /**
     * Step 2: Split the Document into Chunks
     */
    const textSplitter = new CharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 0,
    });

    const chunksOfText = await textSplitter.splitText(loadedDocument[0].pageContent);


    console.log("Document successfully split into chunks!");
    console.log("\n----------\n");

    /**
     * Step 3: Embed Chunks and Create a Vector Store with FAISS
     */
    const embeddingsModel = new OpenAIEmbeddings({
        openAIApiKey: openaiApiKey,
    });

    const vectorDb = await FaissStore.fromTexts(chunksOfText,[], embeddingsModel, {});

    console.log("Vector Store created with embedded chunks (FAISS)!");
    console.log("\n----------\n");

    /**
     * Step 4: Perform Search with the Retriever API
     */
    const retriever = vectorDb.asRetriever({
        k:1,
    });
    const response = await retriever.invoke("what did he say about ketanji brown jackson?");

    console.log("\n----------\n");
    console.log("Ask the RAG App with Retriever: What did he say about ketanji brown jackson?");
    console.log("\n----------\n");

    if (response.length > 0) {
        console.log(response); // Output the most relevant chunk of text
    } else {
        console.log("No relevant content found.");
    }

    console.log("\n----------\n");
}

async function main() {
    try {
        await runChromaWorkflow();
        await runFAISSWorkflow();
    } catch (error) {
        console.error("An error occurred:");
        console.error(error);
    }
}

main();