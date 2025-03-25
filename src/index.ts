import {OpenAI} from 'openai';
import {encoding_for_model, TiktokenModel} from "tiktoken";

const openai = new OpenAI();

async function main() {
    const response = await openai.chat.completions.create({
        model: 'gpt-4o',
        messages: [
            {role: 'user', content: 'Hello, how are you?'}
        ]
    })

    console.log(response.choices[0].message.content);
}

function encodePrompt(prompt: string, model: TiktokenModel) {
    const encoder = encoding_for_model(model);
    const words = encoder.encode(prompt);
    console.log(words)
}

encodePrompt('Hello, how are you?', 'gpt-4o');