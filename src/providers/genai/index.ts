//
//  index.ts
//
//  The MIT License
//  Copyright (c) 2021 - 2026 O2ter Limited. All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.
//

import _ from 'lodash';
import { GoogleGenAI, GoogleGenAIOptions, Content } from "@google/genai";
import { ClientProvider } from '../../client/provider';
import { ChatOptions, EmbedOptions } from '../../client/types';

type GoogleGenAIChatParams = Parameters<GoogleGenAI['models']['generateContent']>[0];
type _GoogleGenAIChatConfig = NonNullable<GoogleGenAIChatParams['config']>;
type GoogleGenAIChatConfig = Omit<_GoogleGenAIChatConfig, 'tools' | 'systemInstruction'> & ChatOptions;

export class GoogleGenAIProvider extends ClientProvider {

  client: GoogleGenAI;

  constructor(options: GoogleGenAIOptions) {
    super();
    this.client = new GoogleGenAI(options);
  };

  async* models() {
    for await (const model of await this.client.models.list()) {
      if (!model.name) continue;
      yield { name: model.name };
    }
  }

  async embeddings({ model, input, signal }: EmbedOptions) {
    const { embeddings } = await this.client.models.embedContent({
      model,
      contents: input,
      config: {
        abortSignal: signal,
        autoTruncate: true,
      },
    });

    return {
      embeddings: embeddings?.map(item => ({
        values: item.values ?? [],
        truncated: item.statistics?.truncated ?? false,
      })) ?? [],
      usage: {
        prompt_tokens: embeddings?.reduce((acc, item) => acc + (item.statistics?.tokenCount ?? 0), 0) ?? 0,
        total_tokens: embeddings?.reduce((acc, item) => acc + (item.statistics?.tokenCount ?? 0), 0) ?? 0,
      },
    };
  }

  #convertMessage(message: ChatOptions['messages'][number]): Content {
    const { role, content } = message;
    switch (role) {
      case 'user':
        return {
          role,
          parts: _.isString(content)
            ? [{ text: content }]
            : content.map(c => {
              switch (c.type) {
                case 'text':
                  return { text: c.text };
                case 'image_url':
                  const url = c.image_url.url;
                  const matches = url.match(/^data:([^;]+);base64,(.+)$/);
                  if (matches) {
                    return {
                      inlineData: {
                        mimeType: matches[1],
                        data: matches[2],
                      },
                    };
                  } else {
                    return {
                      inlineData: {}
                    };
                  }
                default:
                  throw new Error(`Unsupported content type: ${(c as any).type}`);
              }
            }),
        };
      case 'assistant':
        return {
          role: 'model',
          parts: _.compact([
            message.reasoning ? { text: message.reasoning, thought: true } : null,
            { text: message.content },
            ...message.tool_calls?.map(call => ({
              functionCall: {
                name: call.name,
                arguments: call.arguments,
              },
            })) ?? [],
          ]),
        };
      case 'tool':
        let response;
        try {
          response = JSON.parse(content);
        } catch {
          response = { text: content };
        }
        return {
          role: 'tool',
          parts: [{
            functionResponse: {
              id: message.tool_call_id,
              response: response,
            },
          }],
        };
      default:
        throw new Error(`Unsupported message role: ${role}`);
    }
  }

  async* chat({
    model,
    systemMessage,
    messages,
    tools,
    signal,
    ...options
  }: GoogleGenAIChatConfig) {

    const response = await this.client.models.generateContentStream({
      model,
      contents: messages.map(msg => this.#convertMessage(msg)),
      config: {
        ...options,
        abortSignal: signal,
        systemInstruction: systemMessage,
        tools: [{
          functionDeclarations: _.map(tools, tool => ({
            name: tool.name,
            description: tool.description,
            parameters: tool.parameters,
          })),
        }],
      },
    });

    const now = Date.now();
    const toolCallIds = new Map<number, string>();

    for await (const { text: content, functionCalls, usageMetadata: usage } of response) {
      if (content) yield { type: 'content', content } as const;
      if (usage) {
        const total_tokens = usage.totalTokenCount ?? 0;
        const prompt_tokens = usage.promptTokenCount ?? 0;
        const completion_tokens = total_tokens - prompt_tokens;

        yield {
          type: 'usage',
          usage: {
            completion_tokens: completion_tokens,
            prompt_tokens: prompt_tokens,
            total_tokens: total_tokens,
            reasoning_tokens: usage.thoughtsTokenCount,
            cached_tokens: usage.cachedContentTokenCount,
          },
        } as const;
      }
      if (functionCalls) {
        for (const [index, call] of functionCalls.entries()) {
          if (!toolCallIds.has(index)) toolCallIds.set(index, call.id ?? `tool-${now}-${index}`);
          yield {
            type: 'tool_call',
            tool_call_id: toolCallIds.get(index)!,
            name: call.name,
            arguments: call.args as any,
          } as const;
        }
      }
    }
  }
};