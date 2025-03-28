import dotenv from "dotenv";
import path from "path";
import { OpenAI } from "@langchain/openai";
import { ChatOpenAI } from "@langchain/openai";
import { RunnableSequence } from "@langchain/core/runnables";
import { ChatPromptTemplate, PromptTemplate } from "@langchain/core/prompts";
import { StructuredOutputParser } from "@langchain/core/output_parsers";
import { z } from "zod";

// Load environment variables
dotenv.config({
    path: path.resolve(__dirname, "../.env"),
});

// Ensure OpenAI API key exists
const openAIApiKey = process.env.OPENAI_API_KEY;
if (!openAIApiKey) {
    throw new Error("No OpenAI API key found. Make sure the .env file is loaded correctly.");
}

async function main() {
    /**
     * JSON Object Chain
     */
        // Create an OpenAI model instance
    const llmModel = new OpenAI({
            openAIApiKey
        });

    // Create a prompt template for JSON
    const jsonPrompt = PromptTemplate.fromTemplate(
        "Return a JSON object with an `answer` key that answers the following question: {question}"
    );

    // Define the schema with Zod for structured output
    const jsonSchema = z.object({
        answer: z.string().describe("answer to the user's question"),
    });

    // Create a structured output parser
    const jsonParser = StructuredOutputParser.fromZodSchema(jsonSchema);

    // Create a chain for generating JSON answers
    const jsonChain = RunnableSequence.from([jsonPrompt, llmModel, jsonParser]);

    // Invoke the chain
    const jsonResponse = await jsonChain.invoke({
        question: "What is the biggest country?",
    });

    console.log("What is the biggest country?");
    console.log(jsonResponse);
    console.log("\n----------\n");

    /**
     * Custom Joke Chain with Pydantic-equivalent schema
     */
        // Define your "Joke" schema using Zod
    const jokeSchema = z.object({
            setup: z.string().describe("question to set up a joke"),
            punchline: z.string().describe("answer to resolve the joke"),
        });

    // Create a joke parser
    const jokeParser = StructuredOutputParser.fromZodSchema(jokeSchema);

    // Get formatted instructions for joke output
    const jokeFormatInstructions = jokeParser.getFormatInstructions();

    // Create a prompt for generating jokes
    const jokePrompt = ChatPromptTemplate.fromTemplate(
        `Answer the user query below in the form of a JSON object with properties "setup" and "punchline".
{format_instructions}

{query}`
    );

    // Create a chain for jokes
    const jokeChain = RunnableSequence.from([jokePrompt, new ChatOpenAI({ openAIApiKey, model: "gpt-4o" }), jokeParser]);

    // Invoke the joke chain
    const jokeResponse = await jokeChain.invoke({
        query: "Tell me a joke.",
        format_instructions: jokeFormatInstructions,
    });

    console.log("Tell me a joke in custom format defined by Zod:");
    console.log(jokeResponse);
    console.log("\n----------\n");
}

// Execute the main function
main().catch((error) => {
    console.error("An error occurred:", error);
});