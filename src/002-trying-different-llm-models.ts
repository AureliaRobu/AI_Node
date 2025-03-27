// Import required libraries
import dotenv from "dotenv";
import path from "path";
import { ChatGroq } from "@langchain/groq";

// Load environment variables from .env
dotenv.config({
    path: path.resolve(__dirname, "../.env"), // Adjust the path to your .env file
});

// Ensure the required environment variable is available
const groqApiKey = process.env.GROQ_API_KEY;
if (!groqApiKey) {
    throw new Error("No Groq API key found. Make sure the .env file is loaded correctly.");
}

console.log("Your Groq API Key has been loaded successfully:", groqApiKey);

async function main() {
    // Instantiate the ChatGroq models
    const llamaChatModel = new ChatGroq({
        model: "llama-3.3-70b-versatile",
        apiKey: groqApiKey,
    });

    const mistralChatModel = new ChatGroq({
        model: "mistral-saba-24b",
        apiKey: groqApiKey,
    });

    // Define the conversation messages
    const messages = [
        { role: "system", content: "You are an historian expert in the Kennedy family." },
        { role: "human", content: "How many members of the family died tragically?" },
    ];

    console.log("\n----------\n");
    console.log("How many members of the family died tragically? - LLama3 Response:");
    console.log("\n----------\n");

    // Use the first model (Llama3) to generate a response
    const llamaResponse = await llamaChatModel.invoke(messages);
    console.log(llamaResponse.content);

    console.log("\n----------\n");
    console.log("How many members of the family died tragically? - Mistral Response:");
    console.log("\n----------\n");

    // Use the second model (Mistral) to generate a response
    const mistralResponse = await mistralChatModel.invoke(messages);
    console.log(mistralResponse.content);

    console.log("\n----------\n");
}

// Execute the main function and handle errors
main().catch((error) => {
    console.error("An error occurred:", error);
});