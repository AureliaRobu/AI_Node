import dotenv from "dotenv";
import path from "path";
import { ChatOpenAI } from "@langchain/openai";
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
    // Initialize Chat Model
    const chatModel = new ChatOpenAI({
        openAIApiKey: openaiApiKey,
        model: "gpt-3.5-turbo-0125",
    });

    console.log("\n----------\n");

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
        chunkSize: 1000, // Size of each chunk in characters
        chunkOverlap: 200, // Overlapping characters between chunks
    });

    // Use the `splitText` method to split the text into chunks
    const textChunks = await textSplitter.splitText(loadedData[0].pageContent);

    // Manually attach metadata to each chunk
    const metadata = { source: "be-good.txt" }; // Example metadata
    const chunksWithMetadata = textChunks.map((chunk, index) => ({
        chunkContent: chunk,
        metadata: {
            ...metadata,
            chunkIndex: index, // Add index of the chunk
        },
    }));

    console.log("\n----------\n");
    console.log("How many chunks of text were created by the splitter?");
    console.log("\n----------\n");
    console.log(chunksWithMetadata.length);

    console.log("\n----------\n");
    console.log("Print the first chunk of text with metadata:");
    console.log("\n----------\n");
    console.log(chunksWithMetadata[0]);
    console.log("\n----------\n");
}

main().catch((error) => {
    console.error("An error occurred:");
    console.error(error);
});