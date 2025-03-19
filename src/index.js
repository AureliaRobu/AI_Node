import {OpenAI} from 'openai';

const openai = new OpenAI();

async function main() {
    const response = openai.chat.completions.create({
        model: 'gpt-4o',
        messages: [
            {role: 'user', content: 'Hello, how are you?'}
        ]
    })

    console.log(response);
}

main();