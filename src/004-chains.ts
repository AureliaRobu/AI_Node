import dotenv from "dotenv";
import path from "path";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate, FewShotChatMessagePromptTemplate } from "@langchain/core/prompts";

// Load environment variables
dotenv.config({
    path: path.resolve(__dirname, "../.env"),
});

// Ensure the OpenAI API key is available
const openaiApiKey = process.env.OPENAI_API_KEY;
if (!openaiApiKey) {
    throw new Error("No OpenAI API key found. Make sure the .env file is loaded correctly.");
}

async function main() {
    // Create a ChatOpenAI instance
    const chatModel = new ChatOpenAI({
        openAIApiKey: openaiApiKey,
        model: "gpt-4o",
    });

    // Define a Few-Shot Prompt Example with clear translation examples
    const examples = [
        { input: "hi!", output: "¡hola!" },
        { input: "bye!", output: "¡adiós!" },
        { input: "How are you?", output: "¿Cómo estás?" },
        { input: "My name is John", output: "Me llamo John" },
    ];

    // Define an Example Chat Prompt Template
    const examplePrompt = ChatPromptTemplate.fromMessages([
        { role: "human", content: "Translate this English text to Spanish: {input}" },
        { role: "ai", content: "{output}" },
    ]);

    // Create a FewShotChatMessagePromptTemplate
    const fewShotPrompt = new FewShotChatMessagePromptTemplate({
        examplePrompt: examplePrompt,
        examples: examples,
        inputVariables: ["input"],
    });

    // Define the Final Prompt Template with stronger translation instruction
    const finalPrompt = ChatPromptTemplate.fromMessages([
        {
            role: "system",
            content: "You are an English to Spanish translator. Your only task is to translate the user's English text into Spanish. Do not explain or answer questions - only provide the direct Spanish translation."
        },
        fewShotPrompt,
        { role: "human", content: "Translate this English text to Spanish: {input}" },
    ]);

    // Create a chain using the pipe method
    const chain = finalPrompt.pipe(chatModel);

    // Use the chain to invoke with an input
    const response = await chain.invoke({ input: "Who was JFK?" });

    // Print the response
    console.log("\n----------\n");
    console.log("Translate: Who was JFK?");
    console.log(response.content);
    console.log("\n----------\n");
}

// Execute the main function
main().catch((error) => {
    console.error("An error occurred:", error);
});