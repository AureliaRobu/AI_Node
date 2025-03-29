import dotenv from "dotenv";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
} from "@langchain/core/runnables";
import path from "path";
import { z } from "zod";

// Load environment variables
dotenv.config({
    path: path.resolve(__dirname, "../../.env"),
});

// Ensure OpenAI API key exists
const openAIApiKey = process.env.OPENAI_API_KEY;
if (!openAIApiKey) {
    throw new Error("No OpenAI API key found. Make sure the .env file is loaded correctly.");
}

async function main() {
    // Initialize model
    const model = new ChatOpenAI({
        openAIApiKey,
        model: "gpt-4o"
    });

    // Example 1: Basic LCEL chain
    const prompt = ChatPromptTemplate.fromTemplate("tell me a curious fact about {soccer_player}");
    const outputParser = new StringOutputParser();

    let chain = prompt.pipe(model).pipe(outputParser);

    console.log("\n----------\n");
    console.log("Basic LCEL chain:");
    console.log("\n----------\n");

    let response = await chain.invoke({ soccer_player: "Ronaldo" });
    console.log(response);

    console.log("\n----------\n");

    // Example 2: Using bind() to set parameters
    chain = prompt.pipe(model.bind({ stop: ["Ronaldo"] })).pipe(outputParser);

    console.log("\n----------\n");
    console.log("Basic LCEL chain with .bind():");
    console.log("\n----------\n");

    response = await chain.invoke({ soccer_player: "Ronaldo" });
    console.log(response);

    console.log("\n----------\n");

    // Example 3: Using OpenAI function calling with bind()
    const tools = [
        {
            type: "function",
            function: {
                name: "soccerfacts",
                description: "Curious facts about a soccer player",
                parameters: {
                    type: "object",
                    properties: {
                        question: {
                            type: "string",
                            description: "The question for the curious facts about a soccer player"
                        },
                    }
                }
            }

        }
    ];



    const chain2 = model.bind({tools}).pipe(new StringOutputParser())

    console.log("\n----------\n");
    console.log("Call OpenAI Function in LCEL chain with .bind():");
    console.log("\n----------\n");

    response = await chain2.invoke("Mbappe?" );
    console.log(response);

    console.log("\n----------\n");

    // Example 4: Using RunnableParallel with assign
    const makeUppercase = (arg: any) => {
        return arg["original_input"].toUpperCase();
    };

    chain =  RunnableParallel.from({
        original_input: new RunnablePassthrough()
    }).assign({
        uppercase: RunnableLambda.from(makeUppercase)
    });

    console.log("\n----------\n");
    console.log("Basic LCEL chain with .assign():");
    console.log("\n----------\n");

    response = await chain.invoke({
        soccer_player: "Mbappe"
    });
    console.log(response);

    console.log("\n----------\n");
}

main().catch((error) => {
    console.error("An error occurred:");
    console.error(error);
});