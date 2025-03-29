import dotenv from "dotenv";
import path from "path";
import { ChatOpenAI } from "@langchain/openai";
import { OpenAIEmbeddings } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate, PromptTemplate } from "@langchain/core/prompts";
import {
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
    RunnableBranch
} from "@langchain/core/runnables";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { StructuredOutputParser } from "@langchain/core/output_parsers";
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

    // Example 1: RunnablePassthrough
    let chain = new RunnablePassthrough();

    console.log("\n----------\n");
    console.log("Chain with RunnablePassthrough:");
    console.log("\n----------\n");

    let response = await chain.invoke("Abram");
    console.log(response);

    console.log("\n----------\n");

    // Example 2: RunnableLambda
    const russianLastname = (name: string): string => {
        return `${name}ovich`;
    };

    chain = new RunnablePassthrough().pipe(RunnableLambda.from(russianLastname));

    console.log("\n----------\n");
    console.log("Chain with RunnableLambda:");
    console.log("\n----------\n");

    response = await chain.invoke("Abram");
    console.log(response);

    console.log("\n----------\n");

    // Example 3: RunnableParallel - First example
    chain = RunnableParallel.from({
        operation_a: new RunnablePassthrough(),
        operation_b: RunnableLambda.from(russianLastname)
    });

    console.log("\n----------\n");
    console.log("Chain with RunnableParallel:");
    console.log("\n----------\n");

    response = await chain.invoke("Abram");
    console.log(response);

    console.log("\n----------\n");

    // Example 4: RunnableParallel - Second example with inline lambda
    chain =  RunnableParallel.from({
        operation_a: new RunnablePassthrough(),
        soccer_player: (x: any) => x["name"] + "ovich"
    });

    console.log("\n----------\n");
    console.log("Chain with RunnableParallel:");
    console.log("\n----------\n");

    response = await chain.invoke({
        name1: "Jordam",
        name: "Abram"
    });
    console.log(response);

    console.log("\n----------\n");

    // Example 5: More complex chain with RunnableParallel
    const prompt = ChatPromptTemplate.fromTemplate("tell me a curious fact about {soccer_player}");
    const outputParser = new StringOutputParser();

    const russianLastnameFromDictionary = (person: any) => {
        return person["name"] + "ovich";
    };

    chain = RunnableParallel.from({
        operation_a: new RunnablePassthrough(),
        soccer_player: RunnableLambda.from(russianLastnameFromDictionary),
        operation_c: new RunnablePassthrough()
    }).pipe(prompt).pipe(model).pipe(outputParser);

    console.log("\n----------\n");
    console.log("Chain with RunnableParallel:");
    console.log("\n----------\n");

    response = await chain.invoke({
        name1: "Jordam",
        name: "Abram"
    });
     console.log(response);

    console.log("\n----------\n");

    // Example 6: Advanced use of RunnableParallel with retrieval
    const embeddings = new OpenAIEmbeddings({
        openAIApiKey
    });

    const vectorstore = await FaissStore.fromTexts(
        ["AI Accelera has trained more than 10,000 Alumni from all continents and top companies"],
        { },
        embeddings
    );

    const retriever = vectorstore.asRetriever();

    const template = `Answer the question based only on the following context:
  {context}
  
  Question: {question}
  `;

    const retrievalPrompt = ChatPromptTemplate.fromTemplate(template);
    const standardModel = new ChatOpenAI({
        openAIApiKey,
        model: "gpt-4o"
    });

    const retrievalChain =  RunnableParallel.from({
        context: retriever,
        question: new RunnablePassthrough()
    })
        .pipe(retrievalPrompt)
        .pipe(standardModel)
        .pipe(new StringOutputParser());

    console.log("\n----------\n");
    console.log("Chain with Advanced Use of RunnableParallel:");
    console.log("\n----------\n");

    response = await retrievalChain.invoke("who are the Alumni of AI Accelera?");
    console.log(response);

    console.log("\n----------\n");

    // Example 7: Using itemgetter equivalent in TypeScript
    const vectorstore2 = await FaissStore.fromTexts(
        ["AI Accelera has trained more than 5,000 Enterprise Alumni."],
        { },
        embeddings
    );

    const retriever2 = vectorstore2.asRetriever();

    const template2 = `Answer the question based only on the following context:
  {context}
  
  Question: {question}
  
  Answer in the following language: {language}
  `;

    const prompt2 = ChatPromptTemplate.fromTemplate(template2);

    // In TypeScript, we use object destructuring and explicit function for itemgetter
    const itemgetterChain = {
        context: async (input: any) => {
            const docs = await retriever2.invoke(input.question);
            return docs;
        },
        question: (input: any) => input.question,
        language: (input: any) => input.language
    };

    const fullChain = RunnableParallel.from(itemgetterChain)
        .pipe(prompt2)
        .pipe(standardModel)
        .pipe(new StringOutputParser());

    console.log("\n----------\n");
    console.log("Chain with RunnableParallel and itemgetter:");
    console.log("\n----------\n");

    response = await fullChain.invoke({
        question: "How many Enterprise Alumni has trained AI Accelera?",
        language: "Pirate English"
    });
    console.log(response);

    console.log("\n----------\n");

    // Example 8: RunnableBranch
    const rockTemplate = `You are a very smart rock and roll professor. \
  You are great at answering questions about rock and roll in a concise\
  and easy to understand manner.
  
  Here is a question:
  {input}`;

    const rockPrompt = PromptTemplate.fromTemplate(rockTemplate);

    const politicsTemplate = `You are a very good politics professor. \
  You are great at answering politics questions..
  
  Here is a question:
  {input}`;

    const politicsPrompt = PromptTemplate.fromTemplate(politicsTemplate);

    const historyTemplate = `You are a very good history teacher. \
  You have an excellent knowledge of and understanding of people,\
  events and contexts from a range of historical periods.
  
  Here is a question:
  {input}`;

    const historyPrompt = PromptTemplate.fromTemplate(historyTemplate);

    const sportsTemplate = `You are a sports teacher.\
  You are great at answering sports questions.
  
  Here is a question:
  {input}`;

    const sportsPrompt = PromptTemplate.fromTemplate(sportsTemplate);

    const generalPrompt = PromptTemplate.fromTemplate(
        "You are a helpful assistant. Answer the question as accurately as you can.\n\n{input}"
    );

    const promptBranch = RunnableBranch.from([
        [(x) => x.topic === "rock", rockPrompt],
        [(x) => x.topic === "politics", politicsPrompt],
        [(x) => x.topic === "history", historyPrompt],
        [(x) => x.topic === "sports", sportsPrompt],
        generalPrompt
    ]);

    // Define topic classifier schema using Zod
    const TopicClassifier = z.object({
        topic: z.enum(["rock", "politics", "history", "sports", "general"])
            .describe("The topic of the user question")
    });

    type TopicClassifierType = z.infer<typeof TopicClassifier>;

    // Set up function calling for classification
    const classifierModel = new ChatOpenAI({
        openAIApiKey
    }).bind({
        functions: [
            {
                name: "TopicClassifier",
                description: "Classify the topic of the user question",
                parameters: {
                    type: "object",
                    properties: {
                        topic: {
                            type: "string",
                            enum: ["rock", "politics", "history", "sports", "general"],
                            description: "The topic of the user question. One of 'rock', 'politics', 'history', 'sports' or 'general'."
                        }
                    },
                    required: ["topic"]
                }
            }
        ],
        function_call: { name: "TopicClassifier" }
    });

    const parser = new StructuredOutputParser<TopicClassifierType>({
        zod_schema: TopicClassifier
    });

    const classifierChain = classifierModel.pipe(parser);

    const finalChain = RunnablePassthrough.assign({
        topic: (input) => classifierChain.invoke(input).then(res => res.topic)
    })
        .pipe(promptBranch)
        .pipe(new ChatOpenAI({ openAIApiKey }))
        .pipe(new StringOutputParser());

    console.log("\n----------\n");
    console.log("Chain with RunnableBranch:");
    console.log("\n----------\n");

    response = await finalChain.invoke({
        input: "Who was Napoleon Bonaparte?"
    });

    console.log(response);
    console.log("\n----------\n");
}

main().catch((error) => {
    console.error("An error occurred:");
    console.error(error);
});