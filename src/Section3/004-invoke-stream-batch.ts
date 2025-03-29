import dotenv from "dotenv";
import path from "path";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

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
    // Initialize model
    const model = new ChatOpenAI({
        openAIApiKey: openaiApiKey,
        model: "gpt-4o"
    });

    // Create prompt template and chain
    const prompt = ChatPromptTemplate.fromTemplate("Tell me one sentence about {politician}.");
    const parser = new StringOutputParser();
    const chain = prompt.pipe(model).pipe(parser);

    console.log("\n----------\n");
    console.log("Response with invoke:");
    console.log("\n----------\n");

    // Using invoke
    const response = await chain.invoke({ politician: "Churchill" });
    console.log(response);

    console.log("\n----------\n");
    console.log("Response with stream:");
    console.log("\n----------\n");

    // Using stream
    const stream = await chain.stream({ politician: "F.D. Roosevelt" });
    for await (const chunk of stream) {
        console.log(`${chunk}`);
    }

    console.log("\n\n----------\n");
    console.log("Response with batch:");
    console.log("\n----------\n");

    // Using batch
    const batchResponses = await chain.batch([
        { politician: "Lenin" },
        { politician: "Stalin" }
    ]);

    console.log(batchResponses);
    console.log("\n----------\n");
}

main().catch((error) => {
    console.error("An error occurred:");
    console.error(error);
});