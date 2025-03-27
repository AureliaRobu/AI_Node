// Import required libraries
import dotenv from "dotenv";
import path from "path";
import { OpenAI } from "@langchain/openai";
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate } from "@langchain/core/prompts"

// Load environment variables
dotenv.config({
    path: path.resolve(__dirname, "../.env"), // Adjust the path if necessary
});

// Ensure the OpenAI API key is available
const openaiApiKey = process.env.OPENAI_API_KEY;
if (!openaiApiKey) {
    throw new Error("No OpenAI API key found. Make sure the .env file is loaded correctly.");
}

console.log("Your OpenAI API Key has been loaded successfully:", openaiApiKey);

async function main() {
    // Part 1: LLM with a Prompt Template
    const llmModel = new OpenAI({
        openAIApiKey: openaiApiKey,
    });

    const promptTemplate = new PromptTemplate({
        template: "Tell me a {adjective} story about {topic}.",
        inputVariables: ["adjective", "topic"],
    });

    const llmModelPrompt = await promptTemplate.format({
        adjective: "curious",
        topic: "the Kennedy family",
    });

    const llmResponse = await llmModel.invoke(llmModelPrompt);

    console.log("Tell me one curious thing about the Kennedy family:");
    console.log(llmResponse);

    console.log("\n----------\n");

    // Part 2: Chat Model with a Chat Prompt Template
    const chatModel = new ChatOpenAI({
        openAIApiKey: openaiApiKey,
        model: "gpt-4o",
    });

    const chatTemplate = ChatPromptTemplate.fromMessages([
        { role: "system", content: "You are an {profession} expert on {topic}." },
        { role: "human", content: "Hello, Mr. {profession}, can you please answer a question?" },
        { role: "ai", content: "Sure!" },
        { role: "human", content: "{user_input}" },
    ]);

    const formattedMessages = await chatTemplate.formatMessages({
        profession: "Historian",
        topic: "The Kennedy family",
        user_input: "How many grandchildren had Joseph P. Kennedy?",
    });

    const chatResponse = await chatModel.invoke(formattedMessages);

    console.log("How many grandchildren had Joseph P. Kennedy?:");
    console.log(chatResponse.content);

    console.log("\n----------\n");

    // Part 3: Few-Shot Example Using ChatPromptTemplate
    const examples = [
        { input: "hi!", output: "¡hola!" },
        { input: "bye!", output: "¡adiós!" },
    ];

    const examplePrompt = ChatPromptTemplate.fromMessages([
        { role: "human", content: "{input}" },
        { role: "ai", content: "{output}" },
    ]);

    const fewShotPrompt = new FewShotChatMessagePromptTemplate({
        examplePrompt: examplePrompt,
        examples: examples,
        inputVariables: ["input"],

    });

    const finalPrompt = ChatPromptTemplate.fromMessages([
        { role: "system", content: "You are an English-Spanish translator." },
        fewShotPrompt,
        { role: "human", content: "{input}" },
    ]);

    // Example of formatting a prompt (input can be dynamic)
    const finalMessages = await finalPrompt.formatMessages({
        input: "How do you say 'I love programming' in Spanish?",
    });

    const fewShotResponse = await chatModel.invoke(finalMessages);

    console.log("Translation of 'I love programming' in Spanish:");
    console.log(fewShotResponse.content);

    console.log("\n----------\n");
}

// Execute the function
main().catch((error) => {
    console.error("An error occurred:", error);
});