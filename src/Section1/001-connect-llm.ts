// Load environment variables
import dotenv from 'dotenv';
import {OpenAI} from '@langchain/openai'
import { ChatOpenAI } from "@langchain/openai";
import path from "path";


dotenv.config({
    path: path.resolve(__dirname, "../.env"), // Adjust this path to where your .env actually resides
});

// Validate that the environment variables were loaded
const openaiApiKey = process.env.OPENAI_API_KEY;

if (!openaiApiKey) {
    throw new Error("No OpenAI API key found. Make sure the .env file is loaded correctly.");
}

console.log("Your OpenAI API Key has been loaded successfully:", openaiApiKey);


async function main() {
    // Create standard OpenAI model instance
    const llmModel = new OpenAI({
        openAIApiKey: openaiApiKey,
    });

    console.log("\n----------\n");

    // Basic completion
    const response = await llmModel.invoke(
        "Tell me one fun fact about the Kennedy family."
    );

    console.log("Tell me one fun fact about the Kennedy family:");
    console.log(response);

    console.log("\n----------\n");

    // Streaming completion
    console.log("Streaming:");

    const stream = await llmModel.stream(
        "Tell me one fun fact about the Kennedy family."
    );

    for await (const chunk of stream) {
        process.stdout.write(chunk);
    }

    console.log("\n----------\n");

    // Creative model with higher temperature
    const creativeLlmModel = new OpenAI({
        openAIApiKey: openaiApiKey,
        temperature: 0.9,
    });

    const poemResponse = await llmModel.invoke(
        "Write a short 5 line poem about JFK"
    );

    console.log("Write a short 5 line poem about JFK:");
    console.log(poemResponse);

    console.log("\n----------\n");

    // Chat model
    const chatModel = new ChatOpenAI({
        openAIApiKey: openaiApiKey,
        model: "gpt-4o",
    });

    const messages = [
        {
            role: "system",
            content: "You are a helpful assistant that translates English to French.",
        },
        {
            role: "user",
            content: "I love programming.",
        },
    ];


    const chatResponse = await chatModel.invoke(messages);

    console.log(chatResponse.content);

    console.log("\n----------\n");

    // Streaming chat
    console.log("Streaming:");

    const chatStream = await chatModel.stream(messages);

    for await (const chunk of chatStream) {
        process.stdout.write(chunk.content);
    }

    console.log("\n----------\n");
}

main().catch(console.error);