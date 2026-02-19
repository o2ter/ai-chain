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
import { GoogleGenAI, GoogleGenAIOptions } from "@google/genai";
import { ClientProvider } from '../../client/provider';
import { ChatOptions, ChatResponse, EmbedOptions } from '../../client/types';

type GoogleGenAIChatParams = Parameters<GoogleGenAI['models']['generateContent']>[0];
type GoogleGenAIChatConfig = NonNullable<GoogleGenAIChatParams['config']>;

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

  async embeddings({ model, input }: EmbedOptions) {
    const { embeddings } = await this.client.models.embedContent({
      model,
      contents: input,
      config: {
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

  chat<S extends boolean = false>({
    model,
    messages,
    tools,
    stream,
    ...options
  }: Omit<GoogleGenAIChatConfig, 'tools' | 'systemInstruction'> & ChatOptions<S>) {

    const systemInstruction = messages.filter(message => message.role === 'system');
    const params: GoogleGenAIChatParams = {
      model,
      contents: _.compact(messages.map(msg => {
        const { role, content } = msg;
        switch (role) {
          case 'user':
            return {
              role: 'user',
              parts: [{ text: content }],
            };
          case 'assistant':
            return {
              role: 'model',
              parts: [{ text: content }],
            };
        }
      })),
      config: {
        systemInstruction: systemInstruction.map(message => message.content),
        tools: [{
          functionDeclarations: _.map(tools, tool => ({
            name: tool.name,
            description: tool.description,
            parameters: tool.parameters,
          })),
        }],
        ...options,
      },
    };

    if (stream) {
      const self = this;

      return (async function* () {

        const response = await self.client.models.generateContentStream({
          ...params,
        });

        let usage;
        const calls: {
          id?: string;
          name: string;
          arguments: any;
        }[] = [];

        for await (const { text: content, functionCalls, usageMetadata: _usage } of response) {
          if (content) yield { content };
          if (_usage) usage = _usage;
          if (functionCalls) {
            for (const [index, call] of functionCalls.entries()) {
              calls[index] = {
                id: call.id,
                name: call.name ?? calls[index]?.name ?? '',
                arguments: call.args ?? calls[index]?.arguments ?? {},
              };
            }
          }
        }
        if (!_.isEmpty(calls)) yield { tool_calls: calls };
        if (usage) {
          const total_tokens = usage.totalTokenCount ?? 0;
          const prompt_tokens = usage.promptTokenCount ?? 0;
          const completion_tokens = total_tokens - prompt_tokens;

          yield {
            usage: {
              completion_tokens: completion_tokens || undefined,
              prompt_tokens: prompt_tokens || undefined,
              total_tokens: total_tokens || undefined,
              reasoning_tokens: usage.thoughtsTokenCount,
              cached_tokens: usage.cachedContentTokenCount,
            },
          };
        }

      })() as ChatResponse<S>;

    } else {

      return (async () => {

        const response = await this.client.models.generateContent({
          ...params,
        });

        const total_tokens = response.usageMetadata?.totalTokenCount ?? 0;
        const prompt_tokens = response.usageMetadata?.promptTokenCount ?? 0;
        const completion_tokens = total_tokens - prompt_tokens;

        return {
          content: response.text ?? '',
          tool_calls: response.functionCalls?.map(call => ({
            id: call.id,
            name: call.name,
            arguments: call.args,
          })),
          usage: {
            completion_tokens: completion_tokens || undefined,
            prompt_tokens: prompt_tokens || undefined,
            total_tokens: total_tokens || undefined,
            reasoning_tokens: response.usageMetadata?.thoughtsTokenCount,
            cached_tokens: response.usageMetadata?.cachedContentTokenCount,
          },
        };

      })() as ChatResponse<S>;
    }
  }
};