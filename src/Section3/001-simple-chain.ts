import dotenv from "dotenv";
import path from "path";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";

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
     * Step 1: Initialize the Chat Model
     */
    const chatModel = new ChatOpenAI({
        openAIApiKey: openaiApiKey,
        model: "gpt-4o",
    });

    console.log("\n----------\n");

    /**
     * Step 2: Define the Prompt Template
     */
    const prompt = ChatPromptTemplate.fromTemplate("tell me a curious fact about {politician}");

    console.log("Prompt Template initialized!");
    console.log("\n----------\n");

    /**
     * Step 3: Chain Together Prompt, Model, and Parser
     */
    const chain = prompt.pipe(chatModel).pipe(new StringOutputParser());

    console.log("Chain initialized!");
    console.log("\n----------\n");

    /**
     * Step 4: Invoke the Chain
     */
    const input = { politician: "JFK" }; // Template variable: {politician}
    const response = await chain.invoke(input);

    console.log("Chain invoked successfully!");
    console.log("\n----------\n");

    /**
     * Step 5: Log the Result
     */
    console.log("Result from invoking the chain:");
    console.log("\n----------\n");
    console.log(response);
    console.log("\n----------\n");
}

main().catch((error) => {
    console.error("An error occurred:");
    console.error(error);
});