import dotenv from "dotenv";
import path from "path";
import { OpenAIEmbeddings } from "@langchain/openai";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { CharacterTextSplitter } from "@langchain/textsplitters";

// Load environment variables
dotenv.config({
    path: path.resolve(__dirname, "../../.env"),
});

// Ensure OpenAI API Key Exists
const openaiApiKey = process.env.OPENAI_API_KEY;
if (!openaiApiKey) {
    throw new Error("No OpenAI API key found. Make sure the .env file is loaded correctly.");
}

async function main() {

    /**
     * Load a TXT File
     */
    const textLoader = new TextLoader("../../data/be-good.txt");
    const loadedData = await textLoader.load();

    console.log("TXT file loaded:");
    console.log("\n----------\n");
    console.log("Loaded data:");
    console.log(loadedData);

    console.log("\n----------\n");
    console.log("Content of the first page loaded:");
    console.log(loadedData[0].pageContent);
    console.log("\n----------\n");

    /**
     * Initialize Text Splitter
     */
    const textSplitter = new CharacterTextSplitter({
        separator: "\n\n", // Split on double newlines
        chunkSize: 1000, // Maximum chunk size in characters
        chunkOverlap: 200, // Characters overlapping between chunks
    });

    // Split text into chunks
    const texts = await textSplitter.splitText(loadedData[0].pageContent);

    console.log("\n----------\n");
    console.log("How many chunks of text were created by the splitter?");
    console.log("\n----------\n");
    console.log(texts.length);

    console.log("\n----------\n");
    console.log("Print the first chunk of text:");
    console.log("\n----------\n");
    console.log(texts[0]);
    console.log("\n----------\n");

    /**
     * Initialize Embeddings Model
     */
    const embeddingsModel = new OpenAIEmbeddings({
        openAIApiKey: openaiApiKey,
    });

    const chunksOfText = [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!",
    ];

    // Generate embeddings for multiple documents
    const embeddings = await embeddingsModel.embedDocuments(chunksOfText);

    console.log("\n----------\n");
    console.log("How many embeddings were created?");
    console.log("\n----------\n");
    console.log(embeddings.length);

    console.log("\n----------\n");
    console.log("How long is the first embedding?");
    console.log("\n----------\n");
    console.log(embeddings[0].length); // Length of the first embedding vector

    console.log("\n----------\n");
    console.log("Print the first 5 elements of the first embedding:");
    console.log("\n----------\n");
    console.log(embeddings[0].slice(0, 5)); // First 5 numbers in the embedding vector

    console.log("\n----------\n");

    /**
     * Create Embedding for a Query
     */
    const embeddedQuery = await embeddingsModel.embedQuery("What was the name mentioned in the conversation?");
    console.log("Query embedding complete. Example length of the embedded query:");
    console.log(embeddedQuery.length);
}

main().catch((error) => {
    console.error("An error occurred:");
    console.error(error);
});